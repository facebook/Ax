#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import reduce
from logging import Logger
from random import choice, uniform
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from ax import core
from ax.core.arm import Arm
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    Parameter,
    ParameterType,
    RangeParameter,
    TParamValue,
)
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.core.parameter_distribution import ParameterDistribution
from ax.core.types import TParameterization
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none


logger: Logger = get_logger(__name__)


class SearchSpace(Base):
    """Base object for SearchSpace object.

    Contains a set of Parameter objects, each of which have a
    name, type, and set of valid values. The search space also contains
    a set of ParameterConstraint objects, which can be used to define
    restrictions across parameters (e.g. p_a < p_b).
    """

    def __init__(
        self,
        parameters: List[Parameter],
        parameter_constraints: Optional[List[ParameterConstraint]] = None,
    ) -> None:
        """Initialize SearchSpace

        Args:
            parameters: List of parameter objects for the search space.
            parameter_constraints: List of parameter constraints.
        """
        if len({p.name for p in parameters}) < len(parameters):
            raise ValueError("Parameter names must be unique.")

        self._parameters: Dict[str, Parameter] = {p.name: p for p in parameters}
        self.set_parameter_constraints(parameter_constraints or [])

    @property
    def is_hierarchical(self) -> bool:
        return isinstance(self, HierarchicalSearchSpace)

    @property
    def is_robust(self) -> bool:
        return isinstance(self, RobustSearchSpace)

    @property
    def parameters(self) -> Dict[str, Parameter]:
        return self._parameters

    @property
    def parameter_constraints(self) -> List[ParameterConstraint]:
        return self._parameter_constraints

    @property
    def range_parameters(self) -> Dict[str, Parameter]:
        return {
            name: parameter
            for name, parameter in self.parameters.items()
            if isinstance(parameter, RangeParameter)
        }

    @property
    def tunable_parameters(self) -> Dict[str, Parameter]:
        return {
            name: parameter
            for name, parameter in self.parameters.items()
            if not isinstance(parameter, FixedParameter)
        }

    def __getitem__(self, parameter_name: str) -> Parameter:
        """Retrieves the parameter"""
        if parameter_name in self.parameters:
            return self.parameters[parameter_name]
        raise ValueError(
            f"Parameter '{parameter_name}' is not part of the search space."
        )

    def add_parameter_constraints(
        self, parameter_constraints: List[ParameterConstraint]
    ) -> None:
        self._validate_parameter_constraints(parameter_constraints)
        self._parameter_constraints.extend(parameter_constraints)

    def set_parameter_constraints(
        self, parameter_constraints: List[ParameterConstraint]
    ) -> None:
        # Validate that all parameters in constraints are in search
        # space already.
        self._validate_parameter_constraints(parameter_constraints)
        # Set the parameter on the constraint to be the parameter by
        # the matching name among the search space's parameters, so we
        # are not keeping two copies of the same parameter.
        for constraint in parameter_constraints:
            if isinstance(constraint, OrderConstraint):
                constraint._lower_parameter = self.parameters[
                    constraint._lower_parameter.name
                ]
                constraint._upper_parameter = self.parameters[
                    constraint._upper_parameter.name
                ]
            elif isinstance(constraint, SumConstraint):
                for idx, parameter in enumerate(constraint.parameters):
                    constraint.parameters[idx] = self.parameters[parameter.name]

        self._parameter_constraints: List[ParameterConstraint] = parameter_constraints

    def add_parameter(self, parameter: Parameter) -> None:
        if parameter.name in self.parameters.keys():
            raise ValueError(
                f"Parameter `{parameter.name}` already exists in search space. "
                "Use `update_parameter` to update an existing parameter."
            )
        self._parameters[parameter.name] = parameter

    def update_parameter(self, parameter: Parameter) -> None:
        if parameter.name not in self._parameters.keys():
            raise ValueError(
                f"Parameter `{parameter.name}` does not exist in search space. "
                "Use `add_parameter` to add a new parameter."
            )

        prev_type = self._parameters[parameter.name].parameter_type
        if parameter.parameter_type != prev_type:
            raise ValueError(
                f"Parameter `{parameter.name}` has type {prev_type.name}. "
                f"Cannot update to type {parameter.parameter_type.name}."
            )

        self._parameters[parameter.name] = parameter

    def check_all_parameters_present(
        self,
        parameterization: TParameterization,
        raise_error: bool = False,
    ) -> bool:
        """Whether a given parameterization contains all the parameters in the
        search space.

        Args:
            parameterization: Dict from parameter name to value to validate.
            raise_error: If true parameterization does not belong, raises an error
                with detailed explanation of why.

        Returns:
            Whether the parameterization is contained in the search space.
        """
        parameterization_params = set(parameterization.keys())
        ss_params = set(self.parameters.keys())
        if parameterization_params != ss_params:
            if raise_error:
                raise ValueError(
                    f"Parameterization has parameters: {parameterization_params}, "
                    f"but search space has parameters: {ss_params}."
                )
            return False
        return True

    def check_membership(
        self,
        parameterization: TParameterization,
        raise_error: bool = False,
        check_all_parameters_present: bool = True,
    ) -> bool:
        """Whether the given parameterization belongs in the search space.

        Checks that the given parameter values have the same name/type as
        search space parameters, are contained in the search space domain,
        and satisfy the parameter constraints.

        Args:
            parameterization: Dict from parameter name to value to validate.
            raise_error: If true parameterization does not belong, raises an error
                with detailed explanation of why.
            check_all_parameters_present: Ensure that parameterization specifies
                values for all parameters as expected by the search space.

        Returns:
            Whether the parameterization is contained in the search space.
        """
        if check_all_parameters_present:
            if not self.check_all_parameters_present(
                parameterization=parameterization, raise_error=raise_error
            ):
                return False

        for name, value in parameterization.items():
            if not self.parameters[name].validate(value):
                if raise_error:
                    raise ValueError(
                        f"{value} is not a valid value for "
                        f"parameter {self.parameters[name]}"
                    )
                return False

        # parameter constraints only accept numeric parameters
        numerical_param_dict = {
            # pyre-fixme[6]: Expected `typing.Union[...oat]` but got `unknown`.
            name: float(value)
            for name, value in parameterization.items()
            if self.parameters[name].is_numeric
        }

        for constraint in self._parameter_constraints:
            if not constraint.check(numerical_param_dict):
                if raise_error:
                    raise ValueError(f"Parameter constraint {constraint} is violated.")
                return False

        return True

    def check_types(
        self,
        parameterization: TParameterization,
        allow_none: bool = True,
        raise_error: bool = False,
    ) -> bool:
        """Checks that the given parameterization's types match the search space.

        Args:
            parameterization: Dict from parameter name to value to validate.
            allow_none: Whether None is a valid parameter value.
            raise_error: If true and parameterization does not belong, raises an error
                with detailed explanation of why.

        Returns:
            Whether the parameterization has valid types.
        """
        for name, value in parameterization.items():
            if name not in self.parameters:
                if raise_error:
                    raise ValueError(f"Parameter {name} not defined in search space")
                return False

            if value is None and allow_none:
                continue

            if not self.parameters[name].is_valid_type(value):
                if raise_error:
                    raise ValueError(
                        f"{value} is not a valid value for "
                        f"parameter {self.parameters[name]}"
                    )
                return False

        return True

    def cast_arm(self, arm: Arm) -> Arm:
        """Cast parameterization of given arm to the types in this SearchSpace.

        For each parameter in given arm, cast it to the proper type specified
        in this search space. Throws if there is a mismatch in parameter names. This is
        mostly useful for int/float, which user can be sloppy with when hand written.

        Args:
            arm: Arm to cast.

        Returns:
            New casted arm.
        """
        new_parameters: TParameterization = {}
        for name, value in arm.parameters.items():
            # Allow raw values for out of space parameters.
            if name not in self.parameters:
                new_parameters[name] = value
            else:
                new_parameters[name] = self.parameters[name].cast(value)
        return Arm(new_parameters, arm.name if arm.has_name else None)

    def out_of_design_arm(self) -> Arm:
        """Create a default out-of-design arm.

        An out of design arm contains values for some parameters which are
        outside of the search space. In the modeling conversion, these parameters
        are all stripped down to an empty dictionary, since the point is already
        outside of the modeled space.

        Returns:
            New arm w/ null parameter values.
        """
        return self.construct_arm()

    def construct_arm(
        self, parameters: Optional[TParameterization] = None, name: Optional[str] = None
    ) -> Arm:
        """Construct new arm using given parameters and name. Any missing parameters
        fallback to the experiment defaults, represented as None.
        """
        final_parameters: TParameterization = {k: None for k in self.parameters.keys()}
        if parameters is not None:
            # Validate the param values
            for p_name, p_value in parameters.items():
                if p_name not in self.parameters:
                    raise ValueError(f"`{p_name}` does not exist in search space.")
                if p_value is not None and not self.parameters[p_name].validate(
                    p_value
                ):
                    raise ValueError(
                        f"`{p_value}` is not a valid value for parameter {p_name}."
                    )
            final_parameters.update(not_none(parameters))
        return Arm(parameters=final_parameters, name=name)

    def clone(self) -> SearchSpace:
        return self.__class__(
            parameters=[p.clone() for p in self._parameters.values()],
            parameter_constraints=[pc.clone() for pc in self._parameter_constraints],
        )

    def _validate_parameter_constraints(
        self, parameter_constraints: List[ParameterConstraint]
    ) -> None:
        for constraint in parameter_constraints:
            if isinstance(constraint, OrderConstraint) or isinstance(
                constraint, SumConstraint
            ):
                for parameter in constraint.parameters:
                    if parameter.name not in self._parameters.keys():
                        raise ValueError(
                            f"`{parameter.name}` does not exist in search space."
                        )
                    if parameter != self._parameters[parameter.name]:
                        raise ValueError(
                            f"Parameter constraint's definition of '{parameter.name}' "
                            "does not match the SearchSpace's definition"
                        )
            else:
                for parameter_name in constraint.constraint_dict.keys():
                    if parameter_name not in self._parameters.keys():
                        raise ValueError(
                            f"`{parameter_name}` does not exist in search space."
                        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            "parameters=" + repr(list(self._parameters.values())) + ", "
            "parameter_constraints=" + repr(self._parameter_constraints) + ")"
        )

    def __hash__(self) -> int:
        """Make the class hashable to support grouping of GeneratorRuns."""
        return hash(repr(self))


class HierarchicalSearchSpace(SearchSpace):
    def __init__(
        self,
        parameters: List[Parameter],
        parameter_constraints: Optional[List[ParameterConstraint]] = None,
    ) -> None:
        super().__init__(
            parameters=parameters, parameter_constraints=parameter_constraints
        )
        self._all_parameter_names: Set[str] = set(self.parameters.keys())
        self._root: Parameter = self._find_root()
        self._validate_hierarchical_structure()
        logger.debug(f"Found root: {self.root}.")

    @property
    def root(self) -> Parameter:
        """Root of the hierarchical search space tree, as identified during
        ``HierarchicalSearchSpace`` construction.
        """
        return self._root

    def flatten(self) -> SearchSpace:
        """Returns a flattened ``SearchSpace`` with all the parameters in the
        given ``HierarchicalSearchSpace``; ignores their hierarchical structure.
        """
        return SearchSpace(
            parameters=list(self.parameters.values()),
            parameter_constraints=self.parameter_constraints,
        )

    def cast_observation_features(
        self, observation_features: core.observation.ObservationFeatures
    ) -> core.observation.ObservationFeatures:
        """Cast parameterization of given observation features to the hierarchical
        structure of the given search space; return the newly cast observation features
        with the full parameterization stored in ``metadata`` under
        ``Keys.FULL_PARAMETERIZATION``.

        For each parameter in given parameterization, cast it to the proper type
        specified in this search space and remove it from the parameterization if that
        parameter should not be in the arm within the search space due to its
        hierarchical structure.
        """
        full_parameterization_md = {
            Keys.FULL_PARAMETERIZATION: observation_features.parameters.copy()
        }
        obs_feats = observation_features.clone(
            replace_parameters=self._cast_parameterization(
                parameters=observation_features.parameters,
                check_all_parameters_present=False,
            )
        )
        if not obs_feats.metadata:
            obs_feats.metadata = full_parameterization_md  # pyre-ignore[8]
        else:
            obs_feats.metadata = {**obs_feats.metadata, **full_parameterization_md}

        return obs_feats

    def flatten_observation_features(
        self,
        observation_features: core.observation.ObservationFeatures,
        inject_dummy_values_to_complete_flat_parameterization: bool = False,
    ) -> core.observation.ObservationFeatures:
        """Flatten observation features that were previously cast to the hierarchical
        structure of the given search space; return the newly flattened observation
        features. This method re-injects parameter values that were removed from
        observation features during casting (as they are saved in observation features
        metadata).

        Args:
            observation_features: Observation features corresponding to one point
                to flatten.
            inject_dummy_values_to_complete_flat_parameterization: Whether to inject
                values for parameters that are not in the parameterization if they
                are not recorded in the observation features' metadata (this can
                happen if e.g. the point wasn't generated by Ax but attached manually).
        """
        obs_feats = observation_features
        if obs_feats.metadata and Keys.FULL_PARAMETERIZATION in obs_feats.metadata:
            # NOTE: We could just use the full parameterization as stored;
            # opting for a safer option of only injecting parameters that were
            # removed, but not altering those that are present if they have different
            # values in full parameterization as stored in metadata.
            full_parameterization = not_none(obs_feats.metadata)[
                Keys.FULL_PARAMETERIZATION
            ]
            obs_feats.parameters = {**full_parameterization, **obs_feats.parameters}
            return obs_feats

        if inject_dummy_values_to_complete_flat_parameterization:
            # To cast a parameterization to flattened search space, inject dummy values
            # for parameters that were not present in it.
            dummy_values_to_inject = (
                self._gen_dummy_values_to_complete_flat_parameterization(
                    observation_features=obs_feats
                )
            )
            obs_feats.parameters = {**dummy_values_to_inject, **obs_feats.parameters}
            return obs_feats

        # We did not have the full parameterization stored, so we either return the
        # observation features as given without change, or we inject dummy values if
        # that behavior was requested via the opt-in flag.
        warnings.warn(
            f"Cannot flatten observation features {obs_feats} as full "
            "parameterization is not recorded in metadata."
        )
        return obs_feats

    def check_membership(
        self,
        parameterization: TParameterization,
        raise_error: bool = False,
        check_all_parameters_present: bool = True,
    ) -> bool:
        """Whether the given parameterization belongs in the search space.

        Checks that the given parameter values have the same name/type as
        search space parameters, are contained in the search space domain,
        and satisfy the parameter constraints.

        Args:
            parameterization: Dict from parameter name to value to validate.
            raise_error: If true parameterization does not belong, raises an error
                with detailed explanation of why.
            check_all_parameters_present: Ensure that parameterization specifies
                values for all parameters as expected by the search space and its
                hierarchical structure.

        Returns:
            Whether the parameterization is contained in the search space.
        """
        super().check_membership(
            parameterization=parameterization,
            raise_error=raise_error,
            check_all_parameters_present=False,
        )

        # Check that each arm "belongs" in the hierarchical
        # search space; ensure that it only has the parameters that make sense
        # with each other (and does not contain dependent parameters if the
        # parameter they depend on does not have the correct value).
        try:
            cast_to_hss_params = set(
                self._cast_parameterization(
                    parameters=parameterization,
                    check_all_parameters_present=check_all_parameters_present,
                ).keys()
            )
        except RuntimeError:
            if raise_error:
                raise
            return False
        parameterization_params = set(parameterization.keys())
        if cast_to_hss_params != parameterization_params:
            if raise_error:
                raise ValueError(
                    "Parameterization violates the hierarchical structure of the search"
                    f"space; cast version would have parameters: {cast_to_hss_params},"
                    f" but full version contains parameters: {parameterization_params}."
                )

            return False

        return True

    def hierarchical_structure_str(self, parameter_names_only: bool = False) -> str:
        """String representation of the hierarchical structure.

        Args:
            parameter_names_only: Whether parameter should show up just as names
                (instead of full parameter strings), useful for a more concise
                representation.
        """

        def _hrepr(param: Optional[Parameter], value: Optional[str], level: int) -> str:
            is_level_param = param and not value
            if is_level_param:
                param = not_none(param)
                node_name = f"{param.name if parameter_names_only else param}"
                ret = "\t" * level + node_name + "\n"
                if param.is_hierarchical:
                    for val, deps in param.dependents.items():
                        ret += _hrepr(param=None, value=str(val), level=level + 1)
                        for param_name in deps:
                            ret += _hrepr(
                                param=self[param_name],
                                value=None,
                                level=level + 2,
                            )
            else:
                value = not_none(value)
                node_name = f"({value})"
                ret = "\t" * level + node_name + "\n"

            return ret

        return _hrepr(param=self.root, value=None, level=0)

    def _cast_arm(self, arm: Arm) -> Arm:
        """Cast parameterization of given arm to the types in this search space and to
        its hierarchical structure; return the newly cast arm.

        For each parameter in given arm, cast it to the proper type specified
        in this search space and remove it from the arm if that parameter should not be
        in the arm within the search space due to its hierarchical structure.
        """
        # Validate parameter values in flat search space.
        arm = super().cast_arm(arm=arm)

        return Arm(
            parameters=self._cast_parameterization(parameters=arm.parameters),
            name=arm._name,
        )

    def _cast_parameterization(
        self,
        parameters: TParameterization,
        check_all_parameters_present: bool = True,
    ) -> TParameterization:
        """Cast parameterization (of an arm, observation features, etc.) to the
        hierarchical structure of this search space.

        Args:
            parameters: Parameterization to cast to hierarchical structure.
            check_all_parameters_present: Whether to raise an error if a paramete
                 that is expected to be present (according to values of other
                 parameters and the hierarchical structure of the search space)
                 is not specified.
        """
        error_msg_prefix: str = (
            f"Parameterization {parameters} violates the hierarchical structure "
            f"of the search space: {self.hierarchical_structure_str}."
        )

        def _find_applicable_parameters(root: Parameter) -> Set[str]:
            applicable = {root.name}
            if check_all_parameters_present and root.name not in parameters:
                raise RuntimeError(
                    error_msg_prefix
                    + f"Parameter '{root.name}' not in parameterization to cast."
                )

            if not root.is_hierarchical:
                return applicable

            for val, deps in root.dependents.items():
                if parameters[root.name] == val:
                    for dep in deps:
                        applicable.update(_find_applicable_parameters(root=self[dep]))

            return applicable

        applicable_paramers = _find_applicable_parameters(root=self.root)
        if check_all_parameters_present and not all(
            k in parameters for k in applicable_paramers
        ):
            raise RuntimeError(
                error_msg_prefix
                + f"Parameters {applicable_paramers- set(parameters.keys())} are"
                " missing."
            )

        return {k: v for k, v in parameters.items() if k in applicable_paramers}

    def _find_root(self) -> Parameter:
        """Find the root of hierarchical search space: a parameter that does not depend
        on other parameters.
        """
        dependent_parameter_names = set()
        for parameter in self.parameters.values():
            if parameter.is_hierarchical:
                for deps in parameter.dependents.values():
                    dependent_parameter_names.update(param_name for param_name in deps)

        root_parameters = self._all_parameter_names - dependent_parameter_names
        if len(root_parameters) != 1:
            num_parameters = len(self.parameters)
            # TODO: In the future, do not need to fail here; can add a "unifying" root
            # fixed parameter, on which all independent parameters in the HSS can
            # depend.
            raise NotImplementedError(
                "Could not find the root parameter; found dependent parameters "
                f"{dependent_parameter_names}, with {num_parameters} total parameters."
                f" Root parameter candidates: {root_parameters}. Having multiple "
                "independent parameters is not yet supported."
            )

        return self.parameters[root_parameters.pop()]

    def _validate_hierarchical_structure(self) -> None:
        """Validate the structure of this hierarchical search space, ensuring that all
        subtrees are independent (not sharing any parameters) and that all parameters
        are reachable and part of the tree.
        """

        def _check_subtree(root: Parameter) -> Set[str]:
            logger.debug(f"Verifying subtree with root {root}...")
            visited = {root.name}
            # Base case: validate leaf node.
            if not root.is_hierarchical:
                return visited  # TODO: Should there be other validation?

            # Recursive case: validate each subtree.
            visited_in_subtrees = (  # Generator of sets of visited parameter names.
                _check_subtree(root=self[param_name])
                for deps in root.dependents.values()
                for param_name in deps
            )
            # Check that subtrees are disjoint and return names of visited params.
            visited.update(
                reduce(
                    lambda set1, set2: _disjoint_union(set1=set1, set2=set2),
                    visited_in_subtrees,
                    next(visited_in_subtrees),
                )
            )
            logger.debug(f"Visited parameters {visited} in subtree.")
            return visited

        # Verify that all nodes have been reached.
        visited = _check_subtree(root=self._root)
        if len(self._all_parameter_names - visited) != 0:
            raise UserInputError(
                f"Parameters {self._all_parameter_names - visited} are not reachable "
                "from the root. Please check that the hierachical search space provided"
                " is represented as a valid tree with a single root."
            )
        logger.debug(f"Visited all parameters in the tree: {visited}.")

    def _gen_dummy_values_to_complete_flat_parameterization(
        self, observation_features: core.observation.ObservationFeatures
    ) -> Dict[str, TParamValue]:
        dummy_values_to_inject = {}
        for param_name, param in self.parameters.items():
            if param_name in observation_features.parameters:
                continue
            if isinstance(param, FixedParameter):
                dummy_values_to_inject[param_name] = param.value
            elif isinstance(param, ChoiceParameter):
                dummy_values_to_inject[param_name] = choice(param.values)
            elif isinstance(param, RangeParameter):
                val = uniform(param.lower, param.upper)
                print(val)
                if param.parameter_type is ParameterType.INT:
                    val += 0.5
                dummy_values_to_inject[param_name] = param.cast(val)
            else:
                raise NotImplementedError(
                    f"Unhandled parameter type on parameter {param}."
                )
        return dummy_values_to_inject


class RobustSearchSpace(SearchSpace):
    """Search space for robust optimization that supports environmental variables
    and input noise.

    In addition to the usual search space properties, this allows specifying
    environmental variables (parameters) and input noise distributions.
    """

    def __init__(
        self,
        parameters: List[Parameter],
        parameter_distributions: List[ParameterDistribution],
        num_samples: int,
        environmental_variables: Optional[List[Parameter]] = None,
        parameter_constraints: Optional[List[ParameterConstraint]] = None,
    ) -> None:
        """Initialize the robust search space.

        Args:
            parameters: List of parameter objects for the search space.
            parameter_distributions: List of parameter distributions, each representing
                the distribution of one or more parameters. These can be used to
                specify the distribution of the environmental variables or the input
                noise distribution on the parameters.
            num_samples: Number of samples to draw from the `parameter_distributions`
                for the MC approximation of the posterior risk measure. Must agree with
                the `n_w` of the risk measure in `OptimizationConfig`.
            environmental_variables: List of parameter objects, each denoting an
                environmental variable. These must have associated parameter
                distributions.
            parameter_constraints: List of parameter constraints.
        """
        if len(parameter_distributions) == 0:
            raise UserInputError(
                "RobustSearchSpace requires at least one distributional parameter. "
                "Use SearchSpace instead."
            )
        if num_samples < 1 or int(num_samples) != num_samples:
            raise UserInputError("`num_samples` must be a positive integer!")
        self.num_samples = num_samples
        self.parameter_distributions = parameter_distributions
        # Make sure that the env var names are unique.
        environmental_variables = environmental_variables or []
        all_env_vars: Set[str] = {p.name for p in environmental_variables}
        if len(all_env_vars) < len(environmental_variables):
            raise UserInputError("Environmental variable names must be unique!")
        self._environmental_variables: Dict[str, Parameter] = {
            p.name: p for p in environmental_variables
        }
        # Make sure that the environmental variables and parameters are distinct.
        param_names = {p.name for p in parameters}
        for p_name in self._environmental_variables:
            if p_name in param_names:
                raise UserInputError(
                    f"Environmental variable {p_name} should not be repeated "
                    "in parameters."
                )
        # NOTE: We need `_environmental_variables` set before calling `__init__`.
        super().__init__(
            parameters=parameters, parameter_constraints=parameter_constraints
        )
        self._validate_distributions()

    def _validate_distributions(self) -> None:
        r"""Validate the parameter distributions.

        * All distributional parameters must be range parameters.
        * All environmental variables must have a non-multiplicative distribution.
        * Either all or none of the perturbation distributions must be
        multiplicative.
        * Each parameter can have at most one distribution associated with it.
        """
        distributions = self.parameter_distributions
        # Make sure that there is at most one distribution per parameter.
        self._distributional_parameters: Set[str] = set()
        for dist in distributions:
            duplicates = self._distributional_parameters.intersection(dist.parameters)
            if duplicates:
                raise UserInputError(
                    "Received multiple parameter distributions for parameters "
                    f"{duplicates}. Make sure that there is at most one distribution "
                    "specified for any given parameter / environmental variable."
                )
            self._distributional_parameters.update(dist.parameters)

        all_env_vars = set(self._environmental_variables.keys())
        if not all_env_vars.issubset(self._distributional_parameters):
            raise UserInputError(
                "All environmental variables must have a distribution specified."
            )

        self._environmental_distributions: List[ParameterDistribution] = []
        self._perturbation_distributions: List[ParameterDistribution] = []
        if len(all_env_vars) > 0:
            if all_env_vars != self._distributional_parameters:
                # NOTE: We do not support mixing env var and input noise together
                # in a single `ParameterDistribuion`.
                for dist in distributions:
                    is_env = [p in all_env_vars for p in dist.parameters]
                    if not all(is_env) and any(is_env):
                        raise UnsupportedError(
                            "A `ParameterDistribution` must represent either the "
                            "distribution of a set of environmental variables or "
                            "a set of parameter perturbations. Mixing the distribution "
                            "of both types in a single `ParameterDistribution` is "
                            f"not supported. Offending distribution: {dist}."
                        )
                    if any(is_env):
                        self._environmental_distributions.append(dist)
                    else:
                        self._perturbation_distributions.append(dist)
            else:
                self._environmental_distributions = distributions
            if any(d.multiplicative for d in self._environmental_distributions):
                raise UserInputError(
                    "Distributions of environmental variables must have "
                    "`multiplicative=False`."
                )
        else:
            self._perturbation_distributions = distributions

        if not all(
            isinstance(self.parameters[p], RangeParameter)
            for p in self._distributional_parameters
        ):
            raise UserInputError(
                "All parameters with an associated distribution must be "
                "range parameters."
            )

        # Make sure that all or none of perturbation distributions are multiplicative.
        mul_flags = [d.multiplicative for d in self._perturbation_distributions]
        if not (all(mul_flags) or not any(mul_flags)):
            raise UnsupportedError(
                "Non-environmental parameter distributions must be either all "
                "multiplicative or all additive (not multiplicative)."
            )
        self.multiplicative = any(mul_flags)

    def is_environmental_variable(self, parameter_name: str) -> bool:
        r"""Check if a given parameter is an environmental variable.

        Args:
            parameter: A string denoting the name of the parameter.

        Returns:
            A boolean denoting whether the given `parameter_name` corresponds
            to an environmental variable of this search space.
        """
        return parameter_name in self._environmental_variables

    @property
    def parameters(self) -> Dict[str, Parameter]:
        """Get all parameters and environmental variables.

        We include environmental variables here to support `transform_search_space`
        and other similar functionality. It also helps avoid having to overwrite a
        bunch of parent methods.
        """
        return {**self._parameters, **self._environmental_variables}

    def update_parameter(self, parameter: Parameter) -> None:
        raise UnsupportedError("RobustSearchSpace does not support `update_parameter`.")

    def clone(self) -> RobustSearchSpace:
        return self.__class__(
            parameters=[p.clone() for p in self._parameters.values()],
            parameter_distributions=[d.clone() for d in self.parameter_distributions],
            num_samples=self.num_samples,
            environmental_variables=[
                p.clone() for p in self._environmental_variables.values()
            ],
            parameter_constraints=[pc.clone() for pc in self._parameter_constraints],
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            "parameters=" + repr(list(self._parameters.values())) + ", "
            "parameter_distributions=" + repr(self.parameter_distributions) + ", "
            "num_samples=" + repr(self.num_samples) + ", "
            "environmental_variables="
            + repr(list(self._environmental_variables.values()))
            + ", "
            "parameter_constraints=" + repr(self._parameter_constraints) + ")"
        )


@dataclass
class SearchSpaceDigest:
    """Container for lightweight representation of search space properties.

    This is used for communicating between modelbridge and models. This is
    an ephemeral object and not meant to be stored / serialized.

    Attributes:
        feature_names: A list of parameter names.
        bounds: A list [(l_0, u_0), ..., (l_d, u_d)] of tuples representing the
            lower and upper bounds on the respective parameter (both inclusive).
        ordinal_features: A list of indices corresponding to the parameters
            to be considered as ordinal discrete parameters. The corresponding
            bounds are assumed to be integers, and parameter `i` is assumed
            to take on values `l_i, l_i+1, ..., u_i`.
        categorical_features: A list of indices corresponding to the parameters
            to be considered as categorical discrete parameters. The corresponding
            bounds are assumed to be integers, and parameter `i` is assumed
            to take on values `l_i, l_i+1, ..., u_i`.
        discrete_choices: A dictionary mapping indices of discrete (ordinal
            or categorical) parameters to their respective sets of values
            provided as a list.
        task_features: A list of parameter indices to be considered as
            task parameters.
        fidelity_features: A list of parameter indices to be considered as
            fidelity parameters.
        target_fidelities: A dictionary mapping parameter indices (of fidelity
            parameters) to their respective target fidelity value. Only used
            when generating candidates.
        robust_digest: An optional `RobustSearchSpaceDigest` that carries the
            additional attributes if using a `RobustSearchSpace`.
    """

    feature_names: List[str]
    bounds: List[Tuple[Union[int, float], Union[int, float]]]
    ordinal_features: List[int] = field(default_factory=list)
    categorical_features: List[int] = field(default_factory=list)
    discrete_choices: Dict[int, List[Union[int, float]]] = field(default_factory=dict)
    task_features: List[int] = field(default_factory=list)
    fidelity_features: List[int] = field(default_factory=list)
    target_fidelities: Dict[int, Union[int, float]] = field(default_factory=dict)
    robust_digest: Optional[RobustSearchSpaceDigest] = None


@dataclass
class RobustSearchSpaceDigest:
    """Container for lightweight representation of properties that are unique
    to the `RobustSearchSpace`. This is used to append the `SearchSpaceDigest`.

    NOTE: Both `sample_param_perturbations` and `sample_environmental` should
    require no inputs and return a `num_samples x d`-dim array of samples from
    the corresponding parameter distributions, where `d` is the number of
    non-environmental parameters for `distribution_sampler` and the number of
    environmental variables for `environmental_sampler`.

    Attributes:
        sample_param_perturbations: An optional callable for sampling from the
            parameter distributions representing input perturbations.
        sample_environmental: An optional callable for sampling from the
            distributions of the environmental variables.
        environmental_variables: A list of environmental variable names.
        multiplicative: Denotes whether the distribution is multiplicative.
            Only relevant if paired with a `distribution_sampler`.
    """

    sample_param_perturbations: Optional[Callable[[], np.ndarray]] = None
    sample_environmental: Optional[Callable[[], np.ndarray]] = None
    environmental_variables: List[str] = field(default_factory=list)
    multiplicative: bool = False

    def __post_init__(self) -> None:
        if (
            self.sample_param_perturbations is None
            and self.sample_environmental is None
        ):
            raise UserInputError(
                "`RobustSearchSpaceDigest` must be initialized with at least one of "
                "`distribution_sampler` and `environmental_sampler`."
            )


def _disjoint_union(set1: Set[str], set2: Set[str]) -> Set[str]:
    if not set1.isdisjoint(set2):
        raise UserInputError(
            "Two subtrees in the search space contain the same parameters: "
            f"{set1.intersection(set2)}."
        )
    logger.debug(f"Subtrees {set1} and {set2} are disjoint.")
    return set1.union(set2)

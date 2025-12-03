#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from logging import Logger

import pandas as pd
from ax import core
from ax.core.arm import Arm
from ax.core.parameter import (
    ChoiceParameter,
    DerivedParameter,
    FixedParameter,
    get_dummy_value_for_parameter,
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
from ax.core.types import TParameterization
from ax.exceptions.core import AxWarning, UnsupportedError, UserInputError
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from pyre_extensions import none_throws


logger: Logger = get_logger(__name__)
PARAMETER_DF_COLNAMES: dict[str, str] = {
    "name": "Name",
    "type": "Type",
    "domain": "Domain",
    "parameter_type": "Datatype",
    "flags": "Flags",
    "target_value": "Target Value",
    "dependents": "Dependent Parameters",
}


class SearchSpace(Base):
    """Base object for SearchSpace object.

    Contains a set of Parameter objects, each of which have a
    name, type, and set of valid values. The search space also contains
    a set of ParameterConstraint objects, which can be used to define
    restrictions across parameters (e.g. p_a < p_b).
    """

    def __init__(
        self,
        parameters: Sequence[Parameter],
        parameter_constraints: list[ParameterConstraint] | None = None,
    ) -> None:
        """Initialize SearchSpace

        Args:
            parameters: List of parameter objects for the search space.
            parameter_constraints: List of parameter constraints.
        """
        if len({p.name for p in parameters}) < len(parameters):
            raise ValueError("Parameter names must be unique.")

        self._parameters: dict[str, Parameter] = {p.name: p for p in parameters}

        for p in parameters:
            if isinstance(p, DerivedParameter):
                self._validate_derived_parameter(parameter=p)
        self.set_parameter_constraints(parameter_constraints or [])

        if self.is_hierarchical:
            self._validate_hierarchical_structure()

    @property
    def parameters(self) -> dict[str, Parameter]:
        return self._parameters

    @property
    def parameter_constraints(self) -> list[ParameterConstraint]:
        return self._parameter_constraints

    @property
    def range_parameters(self) -> dict[str, RangeParameter]:
        return {
            name: parameter
            for name, parameter in self.parameters.items()
            if isinstance(parameter, RangeParameter)
        }

    @property
    def tunable_parameters(self) -> dict[str, ChoiceParameter | RangeParameter]:
        return {
            name: parameter
            for name, parameter in self.parameters.items()
            if isinstance(parameter, (ChoiceParameter, RangeParameter))
        }

    @property
    def nontunable_parameters(self) -> dict[str, DerivedParameter | FixedParameter]:
        return {
            name: parameter
            for name, parameter in self.parameters.items()
            if isinstance(parameter, (DerivedParameter, FixedParameter))
        }

    @property
    def top_level_parameters(self) -> dict[str, Parameter]:
        """
        Top level parameters are parameters that are not dependent on any other
        parameters.
        """
        dependent_names_by_parameter_by_value = [
            parameter.dependents.values()
            for parameter in self.parameters.values()
            if parameter.is_hierarchical
        ]
        dependent_names_by_parameter = [
            name
            for sublist in dependent_names_by_parameter_by_value
            for name in sublist
        ]
        all_dependent_names = {
            name for sublist in dependent_names_by_parameter for name in sublist
        }

        return {
            name: parameter
            for name, parameter in self.parameters.items()
            if name not in all_dependent_names
        }

    def __getitem__(self, parameter_name: str) -> Parameter:
        """Retrieves the parameter"""
        if parameter_name in self.parameters:
            return self.parameters[parameter_name]
        raise ValueError(
            f"Parameter '{parameter_name}' is not part of the search space."
        )

    @property
    def is_hierarchical(self) -> bool:
        return any(parameter.is_hierarchical for parameter in self.parameters.values())

    @property
    def height(self) -> int:
        """
        Height of the underlying tree structure of this hierarchical search space.
        """

        def _height_from_parameter(parameter: Parameter) -> int:
            if not parameter.is_hierarchical:
                return 1

            return (
                max(
                    _height_from_parameter(parameter=self[param_name])
                    for deps in parameter.dependents.values()
                    for param_name in deps
                )
                + 1
            )

        return max(
            _height_from_parameter(parameter=p)
            for p in self.top_level_parameters.values()
        )

    def add_parameter_constraints(
        self, parameter_constraints: list[ParameterConstraint]
    ) -> None:
        self._validate_parameter_constraints(parameter_constraints)
        self._parameter_constraints.extend(parameter_constraints)

    def set_parameter_constraints(
        self, parameter_constraints: list[ParameterConstraint]
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

        self._parameter_constraints: list[ParameterConstraint] = parameter_constraints

    def add_parameters(
        self,
        parameters: Sequence[Parameter],
    ) -> None:
        """
        Add new parameters to the experiment's search space. This allows extending
        the search space after the experiment has run some trials.

        Backfill values must be provided for all new parameters if the experiment has
        already run some trials. The backfill values represent the parameter values
        that were used in the existing trials.
        """
        # Disabled parameters should be updated
        parameters_to_add = []
        parameters_to_update = []

        # Check which parameters to add to the search space and which to update
        for parameter in parameters:
            # Parameters already exist in search space
            if parameter.name in self.parameters.keys():
                existing_parameter = self.parameters[parameter.name]
                # Only disabled parameters can be re-added
                if not existing_parameter.is_disabled:
                    raise UserInputError(
                        f"Parameter `{parameter.name}` already exists in search space."
                    )
                if type(parameter) is not type(existing_parameter):
                    raise UserInputError(
                        f"Parameter `{parameter.name}` already exists in search "
                        "space. Cannot change parameter type from "
                        f"{type(existing_parameter)} to {type(parameter)}."
                    )
                parameters_to_update.append(parameter)

            # Parameter does not exist in search space
            else:
                parameters_to_add.append(parameter)

        # Add new parameters to search space and status quo
        for parameter in parameters_to_add:
            self.add_parameter(parameter)

        # Update disabled parameters in search space
        for parameter in parameters_to_update:
            self.update_parameter(parameter)

    def disable_parameters(self, default_parameter_values: TParameterization) -> None:
        """
        Disable parameters in the experiment. This allows narrowing the search space
        after the experiment has run some trials.

        When parameters are disabled, they are effectively removed from the search
        space for future trial generation. Existing trials remain valid, and the
        disabled parameters are replaced with fixed default values for all subsequent
        trials.

        Args:
            default_parameter_values: Fixed values to use for the disabled parameters
                in all future trials. These values will be used for the parameter in
                all subsequent trials.
        """
        parameters_to_disable = set(default_parameter_values.keys())
        search_space_parameters = set(self.parameters.keys())
        parameters_not_in_search_space = parameters_to_disable - search_space_parameters

        # Validate that all parameters to disable are in the search space
        if len(parameters_not_in_search_space) > 0:
            raise UserInputError(
                f"Cannot disable parameters {parameters_not_in_search_space} "
                "because they are not in the search space."
            )

        # Validate that all parameters to disable have a valid default
        for parameter_to_disable, default_value in default_parameter_values.items():
            parameter = self.parameters[parameter_to_disable]
            parameter.validate(default_value, raises=True)

        # Disable parameters
        for parameter_to_disable, default_value in default_parameter_values.items():
            self.parameters[parameter_to_disable].disable(default_value)

    def add_parameter(self, parameter: Parameter) -> None:
        if parameter.name in self.parameters.keys():
            raise ValueError(
                f"Parameter `{parameter.name}` already exists in search space. "
                "Use `update_parameter` to update an existing parameter."
            )
        elif isinstance(parameter, DerivedParameter):
            self._validate_derived_parameter(parameter=parameter)
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
        elif isinstance(parameter, DerivedParameter):
            self._validate_derived_parameter(parameter=parameter)

        self._parameters[parameter.name] = parameter

    def check_all_parameters_present(
        self, parameterization: Mapping[str, TParamValue], raise_error: bool = False
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
        parameterization: Mapping[str, TParamValue],
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
        if check_all_parameters_present and not self.is_hierarchical:
            if not self.check_all_parameters_present(
                parameterization=parameterization, raise_error=raise_error
            ):
                return False

        for name, value in parameterization.items():
            p = self.parameters[name]
            kwargs = (
                {"parameters": parameterization}
                if isinstance(p, DerivedParameter)
                else {}
            )
            if not p.validate(value=value, raises=False, **kwargs):
                if raise_error:
                    raise ValueError(
                        f"{value} is not a valid value for "
                        f"parameter {self.parameters[name]}"
                    )
                return False

        # parameter constraints only accept numeric parameters
        numerical_param_dict = {
            name: float(none_throws(value))
            for name, value in parameterization.items()
            if self.parameters[name].is_numeric
        }

        for constraint in self._parameter_constraints:
            if not constraint.check(numerical_param_dict):
                if raise_error:
                    raise ValueError(f"Parameter constraint {constraint} is violated.")
                return False

        if self.is_hierarchical:
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
            except RuntimeError as e:
                if raise_error:
                    raise e
                return False
            parameterization_params = set(parameterization.keys())
            if cast_to_hss_params != parameterization_params:
                if raise_error:
                    raise ValueError(
                        "Parameterization violates the hierarchical structure of the "
                        "search space; cast version would have parameters: "
                        f"{cast_to_hss_params}, but full version contains "
                        f"parameters: {parameterization_params}."
                    )

                return False

        return True

    def check_types(
        self,
        parameterization: TParameterization,
        allow_none: bool = True,
        allow_extra_params: bool = True,
        raise_error: bool = False,
    ) -> bool:
        """Checks that the given parameterization's types match the search space.

        Args:
            parameterization: Dict from parameter name to value to validate.
            allow_none: Whether None is a valid parameter value.
            allow_extra_params: If parameterization can have params not in search space.
            raise_error: If true and parameterization does not belong, raises an error
                with detailed explanation of why.

        Returns:
            Whether the parameterization has valid types.
        """
        for name, value in parameterization.items():
            if name not in self.parameters:
                if allow_extra_params:
                    continue
                elif raise_error:
                    raise ValueError(f"Parameter {name} not defined in search space")
                else:
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
        self, parameters: TParameterization | None = None, name: str | None = None
    ) -> Arm:
        """Construct new arm using given parameters and name. Any missing parameters
        fallback to the experiment defaults, represented as None.
        """
        final_parameters = dict.fromkeys(self.parameters.keys(), None)
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
            final_parameters.update(none_throws(parameters))
        return Arm(parameters=final_parameters, name=name)

    def clone(self) -> SearchSpace:
        return self.__class__(
            parameters=[p.clone() for p in self._parameters.values()],
            parameter_constraints=[pc.clone() for pc in self._parameter_constraints],
        )

    def _validate_parameter_constraints(
        self, parameter_constraints: list[ParameterConstraint]
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
                    p = self._parameters.get(parameter_name)
                    if p is None:
                        raise ValueError(
                            f"`{parameter_name}` does not exist in search space."
                        )
                    elif isinstance(p, DerivedParameter):
                        raise ValueError(
                            "Parameter constraints cannot be used with derived "
                            "parameters."
                        )

    def _validate_hierarchical_structure(self) -> None:
        """Validate the structure of this hierarchical search space, ensuring that all
        subtrees are independent (not sharing any parameters) and that all parameters
        are reachable and part of the tree.
        """

        def visit(parameter_name: str) -> list[str]:
            """
            DFS of a hierarchical search space, returning a list of all parameters
            visited in the traversal.
            """
            parameter = self[parameter_name]

            if parameter.is_hierarchical:
                visited = [
                    visit(parameter_name=child)
                    for sublist in parameter.dependents.values()
                    for child in sublist
                ]

                return [
                    parameter_name,
                    *[child for sublist in visited for child in sublist],
                ]

            return [parameter_name]

        traversal = [
            item
            for sublist in [
                visit(parameter_name=parameter_name)
                for parameter_name in self.top_level_parameters.keys()
            ]
            for item in sublist
        ]
        traversal_set = {*traversal}

        # Ensure that no parameters were visited more than once (i.e. no cycles).
        if len(traversal) != len(traversal_set):
            duplicates = [
                parameter_name
                for parameter_name in traversal_set
                if traversal.count(parameter_name) > 1
            ]
            raise UserInputError(
                "Hierarchical search space contains a cycle. Please check that the "
                "hierachical search space provided is represented as a valid tree. "
                f"{duplicates} could be reached from multiple paths."
            )

    def validate_membership(self, parameters: TParameterization) -> None:
        self.check_membership(parameterization=parameters, raise_error=True)
        # `check_membership` uses int and float interchangeably, which we don't
        # want here.
        for p_name, parameter in self.parameters.items():
            if self.is_hierarchical and p_name not in parameters:
                # Parameterizations in HSS-s can be missing some of the dependent
                # parameters based on the hierarchical structure and values of
                # the parameters those depend on.
                continue
            param_val = parameters.get(p_name)
            if not isinstance(param_val, parameter.python_type):
                typ = type(param_val)
                raise UnsupportedError(
                    f"Value for parameter {p_name}: {param_val} is of type {typ}, "
                    f"expected  {parameter.python_type}. If the intention was to have"
                    f" the parameter on experiment be of type {typ}, set `value_type`"
                    f" on experiment creation for {p_name}."
                )

    def _validate_derived_parameter(self, parameter: DerivedParameter) -> None:
        is_int = parameter.parameter_type == ParameterType.INT
        for p_name in parameter.parameter_names_to_weights.keys():
            p = self._parameters.get(p_name)
            if p is None:
                raise ValueError(
                    f"Parameter {p_name} is not in the search space, but is used in a "
                    "derived parameter."
                )
            if not p.is_numeric:
                raise ValueError(
                    f"Parameter {p_name} is not a float or int, but is used in a "
                    "derived parameter."
                )
            elif is_int and p.parameter_type == ParameterType.FLOAT:
                raise ValueError(
                    f"Parameter {p_name} is a float, but is used in a derived "
                    "parameter with int type."
                )
            elif isinstance(p, DerivedParameter):
                raise ValueError(
                    "Parameter cannot be derived from another derived parameter."
                )
            elif isinstance(p, FixedParameter):
                # Note: relaxing this would require updating RemoveFixed to ensure
                # FixedParameters are added back in untransform_observation_features,
                # before derived parameters are added back.
                raise ValueError(
                    "Parameter cannot be derived from a fixed parameter. The "
                    "`intercept` argument in a derived parameter can be used "
                    "to add an fixed value to a derived parameter."
                )

    def backfill_values(self) -> TParameterization:
        """Backfill values for parameters that have a backfill value."""
        return {
            name: parameter.backfill_value
            for name, parameter in self.parameters.items()
            if parameter.backfill_value is not None
        }

    def hierarchical_structure_str(self, parameter_names_only: bool = False) -> str:
        """String representation of the hierarchical structure.

        Args:
            parameter_names_only: Whether parameter should show up just as names
                (instead of full parameter strings), useful for a more concise
                representation.
        """

        def _hrepr(param: Parameter | None, value: str | None, level: int) -> str:
            is_level_param = param and not value
            if is_level_param:
                param = none_throws(param)
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
                value = none_throws(value)
                node_name = f"({value})"
                ret = "\t" * level + node_name + "\n"

            return ret

        return "".join(
            _hrepr(param=param, value=None, level=0)
            for param in self.top_level_parameters.values()
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

    @property
    def summary_df(self) -> pd.DataFrame:
        """Creates a dataframe with information about each parameter in the given
        search space. The resulting dataframe has one row per parameter, and the
        following columns:
            - Name: the name of the parameter.
            - Type: the parameter subclass (Fixed, Range, Choice).
            - Domain: the parameter's domain (e.g., "range=[0, 1]" or
              "values=['a', 'b']").
            - Datatype: the datatype of the parameter (int, float, str, bool).
            - Flags: flags associated with the parameter, if any.
            - Target Value: the target value of the parameter, if applicable.
            - Dependent Parameters: for parameters in hierarchical search spaces,
            mapping from parameter value -> list of dependent parameter names.
        """
        records = [p.summary_dict for p in self.parameters.values()]
        df = pd.DataFrame(records).fillna(value="None")
        df.rename(columns=PARAMETER_DF_COLNAMES, inplace=True)
        # Reorder columns.
        df = df[
            [
                colname
                for colname in PARAMETER_DF_COLNAMES.values()
                if colname in df.columns
            ]
        ]
        return df

    def flatten(self) -> SearchSpace:
        """
        Returns a new SearchSpace with all hierarchical structure removed.
        """

        def flatten_parameter(parameter: Parameter) -> Parameter:
            if (
                isinstance(parameter, ChoiceParameter)
                or isinstance(parameter, FixedParameter)
            ) and parameter.is_hierarchical:
                cloned_parameter = parameter.clone()
                cloned_parameter._dependents = None

                return cloned_parameter

            return parameter

        return SearchSpace(
            parameters=[flatten_parameter(p) for p in self.parameters.values()],
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
                values for parameters that are not in the parameterization.
                This will be used to complete the parameterization after re-injecting
                the parameters that are recorded in the metadata (for parameters
                that were generated by Ax).
        """
        obs_feats = observation_features
        has_full_parameterization = Keys.FULL_PARAMETERIZATION in (
            obs_feats.metadata or {}
        )

        if obs_feats.parameters == {} and not has_full_parameterization:
            # Return as is if the observation feature does not have any parameters.
            return obs_feats

        if has_full_parameterization:
            # If full parameterization is recorded, use it to fill in missing values.
            full_parameterization = none_throws(obs_feats.metadata)[
                Keys.FULL_PARAMETERIZATION
            ]
            obs_feats.parameters = {**full_parameterization, **obs_feats.parameters}

        if len(obs_feats.parameters) < len(self.parameters):
            if inject_dummy_values_to_complete_flat_parameterization:
                # Inject dummy values for parameters missing from the parameterization.
                for p_name, p in self.parameters.items():
                    if p_name not in obs_feats.parameters:
                        obs_feats.parameters[p_name] = get_dummy_value_for_parameter(p)
            else:
                # The parameterization is still incomplete.
                warnings.warn(
                    f"Cannot flatten observation features {obs_feats} as full "
                    "parameterization is not recorded in metadata and "
                    "`inject_dummy_values_to_complete_flat_parameterization` is "
                    "set to False.",
                    AxWarning,
                    stacklevel=2,
                )
        return obs_feats

    def _cast_parameterization(
        self,
        parameters: Mapping[str, TParamValue],
        check_all_parameters_present: bool = True,
    ) -> TParameterization:
        """Cast parameterization (of an arm, observation features, etc.) to the
        hierarchical structure of this search space.

        Args:
            parameters: Parameterization to cast to hierarchical structure.
            check_all_parameters_present: Whether to raise an error if a parameter
                that is expected to be present (according to values of other
                parameters and the hierarchical structure of the search space)
                is not specified. When this is False, if a parameter is missing,
                its dependents will not be included in the returned parameterization.
        """
        error_msg_prefix: str = (
            f"Parameterization {parameters} violates the hierarchical structure "
            f"of the search space: {self.hierarchical_structure_str}."
        )

        def find_applicable_parameters(parameter: Parameter) -> set[str]:
            if check_all_parameters_present and parameter.name not in parameters:
                raise RuntimeError(
                    error_msg_prefix
                    + f"Parameter {parameter.name} not in parameterization to cast."
                )

            if (
                parameter.is_hierarchical
                and parameter.name in parameters
                and parameters[parameter.name] in parameter.dependents
            ):
                dependents_applicable_parameters = {
                    item
                    for sublist in [
                        find_applicable_parameters(self.parameters[child])
                        for child in parameter.dependents[parameters[parameter.name]]
                    ]
                    for item in sublist
                }

                return {
                    parameter.name,
                    *dependents_applicable_parameters,
                }

            return {parameter.name}

        all_applicable_parameters = {
            item
            for sublist in [
                find_applicable_parameters(parameter=parameter)
                for parameter in self.top_level_parameters.values()
            ]
            for item in sublist
        }

        if check_all_parameters_present and not all(
            k in parameters for k in all_applicable_parameters
        ):
            raise RuntimeError(
                error_msg_prefix
                + f"Parameters {all_applicable_parameters - set(parameters.keys())} are"
                " missing."
            )

        return {k: v for k, v in parameters.items() if k in all_applicable_parameters}


class HierarchicalSearchSpace(SearchSpace):
    """
    HierarchicalSearchSpace is deprecated. Its functionality has been upstreamed into
    SearchSpace. Please use SearchSpace instead. The HierarchicalSearchSpace class
    remains for backwards compatibility with previously stored experiments but will be
    removed in the future.
    """

    pass


@dataclass
class SearchSpaceDigest:
    """Container for lightweight representation of search space properties.

    This is used for communicating between adapter and models. This is
    an ephemeral object and not meant to be stored / serialized. It is typically
    constructed from the transformed search space using `extract_search_space_digest`,
    whose docstring explains how various fields are populated.

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
        target_values: A dictionary mapping parameter indices of fidelity or
            task parameters to their respective target value.
        hierarchical_dependencies: A dictionary that specifies the dependencies between
            parameters if using `HierarchicalSearchSpace`. It looks like as follows
            ```
            # P2 is active if P1 == 0
            # P3 and P4 are active if P1 == 1
            {
                1: {0: [2], 1: [3, 4]},
            }
            ```
    """

    feature_names: list[str]
    bounds: list[tuple[int | float, int | float]]
    ordinal_features: list[int] = field(default_factory=list)
    categorical_features: list[int] = field(default_factory=list)
    discrete_choices: Mapping[int, Sequence[int | float]] = field(default_factory=dict)
    task_features: list[int] = field(default_factory=list)
    fidelity_features: list[int] = field(default_factory=list)
    target_values: dict[int, int | float] = field(default_factory=dict)
    # NOTE: We restrict that hierarchical parameters have to be either categorical or
    # discrete.
    hierarchical_dependencies: dict[int, dict[int, list[int]]] | None = None


def _disjoint_union(set1: set[str], set2: set[str]) -> set[str]:
    if not set1.isdisjoint(set2):
        raise UserInputError(
            "Two subtrees in the search space contain the same parameters: "
            f"{set1.intersection(set2)}."
        )
    logger.debug(f"Subtrees {set1} and {set2} are disjoint.")
    return set1.union(set2)

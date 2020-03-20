#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, List, Optional

from ax.core.arm import Arm
from ax.core.base import Base
from ax.core.parameter import FixedParameter, Parameter
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.core.types import TParameterization
from ax.utils.common.typeutils import not_none


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
    def parameters(self) -> Dict[str, Parameter]:
        return self._parameters

    @property
    def parameter_constraints(self) -> List[ParameterConstraint]:
        return self._parameter_constraints

    @property
    def tunable_parameters(self) -> Dict[str, Parameter]:
        return {
            name: parameter
            for name, parameter in self._parameters.items()
            if not isinstance(parameter, FixedParameter)
        }

    def add_parameter_constraints(
        self, parameter_constraints: List[ParameterConstraint]
    ) -> None:
        self._validate_parameter_constraints(parameter_constraints)
        self._parameter_constraints.extend(parameter_constraints)

    def set_parameter_constraints(
        self, parameter_constraints: List[ParameterConstraint]
    ) -> None:
        self._validate_parameter_constraints(parameter_constraints)
        self._parameter_constraints: List[ParameterConstraint] = parameter_constraints

    def add_parameter(self, parameter: Parameter) -> None:
        if parameter.name in self._parameters.keys():
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

    def check_membership(
        self, parameterization: TParameterization, raise_error: bool = False
    ) -> bool:
        """Whether the given parameterization belongs in the search space.

        Checks that the given parameter values have the same name/type as
        search space parameters, are contained in the search space domain,
        and satisfy the parameter constraints.

        Args:
            parameterization: Dict from parameter name to value to validate.
            raise_error: If true parameterization does not belong, raises an error
                with detailed explanation of why.

        Returns:
            Whether the parameterization is contained in the search space.
        """
        if len(parameterization) != len(self._parameters):
            if raise_error:
                raise ValueError(
                    f"Parameterization has {len(parameterization)} parameters "
                    f"but search space has {len(self._parameters)}."
                )
            return False

        for name, value in parameterization.items():
            if name not in self._parameters:
                if raise_error:
                    raise ValueError(
                        f"Parameter {name} not defined in search space"
                        f"with parameters {self._parameters}"
                    )
                return False

            if not self._parameters[name].validate(value):
                if raise_error:
                    raise ValueError(
                        f"{value} is not a valid value for "
                        f"parameter {self._parameters[name]}"
                    )
                return False

        # parameter constraints only accept numeric parameters
        numerical_param_dict = {
            # pyre-fixme[6]: Expected `typing.Union[...oat]` but got `unknown`.
            name: float(value)
            for name, value in parameterization.items()
            if self._parameters[name].is_numeric
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

        Checks that the names of the parameterization match those specified in
        the search space, and the given values are of the correct type.

        Args:
            parameterization: Dict from parameter name to value to validate.
            allow_none: Whether None is a valid parameter value.
            raise_error: If true and parameterization does not belong, raises an error
                with detailed explanation of why.

        Returns:
            Whether the parameterization has valid types.
        """
        if len(parameterization) != len(self._parameters):
            if raise_error:
                raise ValueError(
                    f"Parameterization has {len(parameterization)} parameters "
                    f"but search space has {len(self._parameters)}.\n"
                    f"Parameterization: {parameterization}.\n"
                    f"Search Space: {self._parameters}."
                )
            return False

        for name, value in parameterization.items():
            if name not in self._parameters:
                if raise_error:
                    raise ValueError(f"Parameter {name} not defined in search space")
                return False

            if value is None and allow_none:
                continue

            if not self._parameters[name].is_valid_type(value):
                if raise_error:
                    raise ValueError(
                        f"{value} is not a valid value for "
                        f"parameter {self._parameters[name]}"
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
            if name not in self._parameters:
                new_parameters[name] = value
            else:
                new_parameters[name] = self._parameters[name].cast(value)
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
        """Construct new arm using given parameters and name. Any
        missing parameters fallback to the experiment defaults,
        represented as None
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

    def clone(self) -> "SearchSpace":
        return SearchSpace(
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
            "SearchSpace("
            "parameters=" + repr(list(self._parameters.values())) + ", "
            "parameter_constraints=" + repr(self._parameter_constraints) + ")"
        )

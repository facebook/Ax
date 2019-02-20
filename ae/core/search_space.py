#!/usr/bin/env python3
# pyre-strict

from typing import Dict, List, Optional

from ae.lazarus.ae.core.base import Base
from ae.lazarus.ae.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    Parameter,
    RangeParameter,
)
from ae.lazarus.ae.core.parameter_constraint import ParameterConstraint
from ae.lazarus.ae.core.types.types import TParamValue


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

    def validate(self, parameter_dict: Dict[str, TParamValue]) -> bool:
        if len(parameter_dict) != len(self._parameters):
            return False

        for name, value in parameter_dict.items():
            if not self._parameters[name].validate(value):
                return False

        # parameter constraints only accept numeric parameters
        numerical_param_dict = {
            # pyre-fixme[6]: Expected `typing.Union[...oat]` but got `unknown`.
            name: float(value)
            for name, value in parameter_dict.items()
            if self._parameters[name].is_numeric
        }

        for constraint in self._parameter_constraints:
            if not constraint.check(numerical_param_dict):
                return False

        return True

    def clone(self) -> "SearchSpace":
        return SearchSpace(
            parameters=[p.clone() for p in self._parameters.values()],
            parameter_constraints=[pc.clone() for pc in self._parameter_constraints],
        )

    def _validate_parameter_constraints(
        self, parameter_constraints: List[ParameterConstraint]
    ) -> None:
        for constraint in parameter_constraints:
            for parameter_name in constraint.constraint_dict.keys():
                if parameter_name not in self._parameters.keys():
                    raise ValueError(
                        f"`{parameter_name}` does not exist in search space."
                    )

                parameter = self._parameters[parameter_name]
                if not parameter.is_numeric:
                    raise ValueError(
                        f"Parameter constraints only supported for types int and float."
                    )

                # ChoiceParameters are transformed either using OneHotEncoding
                # or the OrderedChoice transform. Both are non-linear, and
                # AE models only support linear constraints.
                if isinstance(parameter, ChoiceParameter):
                    raise ValueError(
                        f"Parameter constraints not supported for ChoiceParameter."
                    )

                # Log parameters require a non-linear transformation, and AE
                # models only support linear constraints.
                if (
                    isinstance(parameter, RangeParameter)
                    and parameter.log_scale is True
                ):
                    raise ValueError(
                        f"Parameter constraints not allowed on log scale parameters."
                    )

    def __repr__(self) -> str:
        return (
            "SearchSpace("
            "parameters=" + repr(list(self._parameters.values())) + ", "
            "parameter_constraints=" + repr(self._parameter_constraints) + ")"
        )

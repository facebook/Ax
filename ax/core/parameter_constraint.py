#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Sequence

from ax.core.parameter import Parameter, RangeParameter
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.utils.common.base import SortableBase
from ax.utils.common.string_utils import unsanitize_name
from ax.utils.common.sympy import extract_coefficient_dict_from_inequality


class ParameterConstraint(SortableBase):
    """Base class for linear parameter constraints.

    Constraints are expressed as a SymPy parsable inequality string.
    """

    def __init__(self, inequality: str) -> None:
        """Initialize ParameterConstraint

        Args:
            inequality: String representation of the constraint. At this point in time
            Ax only accepts linear inequality constraints.
        """
        self._inequality_str = inequality

        coefficient_dict = extract_coefficient_dict_from_inequality(
            inequality_str=inequality
        )

        # Iterate through the coefficients to extract the parameter names and weights
        # and the bound
        self._constraint_dict: dict[str, float] = {}
        self._bound: float = 0.0
        for term, coefficient in coefficient_dict.items():
            if term.is_symbol:
                self._constraint_dict[unsanitize_name(term.name)] = coefficient
            elif term.is_number:
                # Invert because we are "moving" the bound to the right hand side
                self._bound = -1 * coefficient
            else:
                raise UserInputError(
                    "Only linear inequality parameter constraints are supported, found "
                    f"{inequality}"
                )

    @property
    def constraint_dict(self) -> dict[str, float]:
        """Get mapping from parameter names to weights."""
        return self._constraint_dict

    @property
    def bound(self) -> float:
        """Get bound of the inequality of the constraint."""
        return self._bound

    @bound.setter
    def bound(self, bound: float) -> None:
        """Set bound."""
        self._bound = bound

    def check(self, parameter_dict: dict[str, int | float]) -> bool:
        """Whether or not the set of parameter values satisfies the constraint.

        Does a weighted sum of the parameter values based on the constraint_dict
        and checks that the sum is less than the bound.

        Args:
            parameter_dict: Map from parameter name to parameter value.

        Returns:
            Whether the constraint is satisfied.
        """
        for parameter_name in self.constraint_dict.keys():
            if parameter_name not in parameter_dict.keys():
                raise ValueError(f"`{parameter_name}` not present in param_dict.")

        weighted_sum = sum(
            float(parameter_dict[param]) * weight
            for param, weight in self.constraint_dict.items()
        )
        # Expected `int` for 2nd anonymous parameter to call `int.__le__` but got
        # `float`.
        return weighted_sum <= self._bound + 1e-8  # allow for numerical imprecision

    def clone(self) -> ParameterConstraint:
        """Clone."""
        return ParameterConstraint(inequality=self._inequality_str)

    def clone_with_transformed_parameters(
        self, transformed_parameters: dict[str, Parameter]
    ) -> ParameterConstraint:
        """Clone, but replaced parameters with transformed versions."""
        return self.clone()

    def __repr__(self) -> str:
        return (
            "ParameterConstraint("
            + " + ".join(f"{v}*{k}" for k, v in sorted(self.constraint_dict.items()))
            + f" <= {self._bound})"
        )

    @property
    def _unique_id(self) -> str:
        return str(self)


class SumConstraint(ParameterConstraint):
    """Constraint on the sum of parameters being greater or less than a bound."""

    def __init__(
        self, parameters: list[Parameter], is_upper_bound: bool, bound: float
    ) -> None:
        """Initialize SumConstraint

        Args:
            parameters: List of parameters whose sum to constrain on.
            is_upper_bound: Whether the bound is an upper or lower bound on the sum.
            bound: The bound on the sum.
        """
        validate_constraint_parameters(parameters)

        self._parameters = parameters
        self._is_upper_bound: bool = is_upper_bound
        self._parameter_names: list[str] = [parameter.name for parameter in parameters]
        self._bound: float = self._inequality_weight * bound
        self._constraint_dict: dict[str, float] = {
            name: self._inequality_weight for name in self._parameter_names
        }

    @property
    def parameters(self) -> list[Parameter]:
        """Parameters."""
        return self._parameters

    @property
    def constraint_dict(self) -> dict[str, float]:
        """Weights on parameters for linear constraint representation."""
        return self._constraint_dict

    @property
    def op(self) -> ComparisonOp:
        """Whether the sum is constrained by a <= or >= inequality."""
        return ComparisonOp.LEQ if self._is_upper_bound else ComparisonOp.GEQ

    @property
    def is_upper_bound(self) -> bool:
        """Whether the bound is an upper or lower bound on the sum."""
        return self._is_upper_bound

    def clone(self) -> SumConstraint:
        """Clone.

        To use the same constraint, we need to reconstruct the original bound.
        We do this by re-applying the original bound weighting.
        """
        return SumConstraint(
            parameters=[p.clone() for p in self._parameters],
            is_upper_bound=self._is_upper_bound,
            bound=self._inequality_weight * self._bound,
        )

    def clone_with_transformed_parameters(
        self, transformed_parameters: dict[str, Parameter]
    ) -> SumConstraint:
        """Clone, but replace parameters with transformed versions."""
        return SumConstraint(
            parameters=[transformed_parameters[p.name] for p in self._parameters],
            is_upper_bound=self._is_upper_bound,
            bound=self._inequality_weight * self._bound,
        )

    @property
    def _inequality_weight(self) -> float:
        """Multiplier of all terms in the inequality.

        If the constraint is an upper bound, it is v1 + v2 ... v_n <= b
        If the constraint is an lower bound, it is -v1 + -v2 ... -v_n <= -b
        This property returns 1 or -1 depending on the scenario
        """
        return 1.0 if self._is_upper_bound else -1.0

    def __repr__(self) -> str:
        symbol = ">=" if self.op == ComparisonOp.GEQ else "<="
        return (
            "SumConstraint("
            + " + ".join(self._parameter_names)
            + " {} {})".format(
                symbol, self._bound if self.op == ComparisonOp.LEQ else -self._bound
            )
        )


def validate_constraint_parameters(parameters: Sequence[Parameter]) -> None:
    """Basic validation of parameters used in a constraint.

    Args:
        parameters: Parameters used in constraint.

    Raises:
        ValueError if the parameters are not valid for use.
    """
    unique_parameter_names = {p.name for p in parameters}
    if len(unique_parameter_names) != len(parameters):
        raise ValueError("Duplicate parameter in constraint.")

    for parameter in parameters:
        if not isinstance(parameter, RangeParameter):
            raise ValueError(
                "All parameters in a parameter constraint must be RangeParameters."
                f"Found {parameter}"
            )

        # Log parameters require a non-linear transformation, and Ax
        # models only support linear constraints.
        if isinstance(parameter, RangeParameter) and parameter.log_scale is True:
            raise ValueError(
                "Parameter constraints not allowed on log scale parameters."
            )

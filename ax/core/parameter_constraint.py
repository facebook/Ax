#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Sequence

from ax.core.parameter import ChoiceParameter, Parameter, RangeParameter
from ax.exceptions.core import UserInputError
from ax.utils.common.base import SortableBase
from ax.utils.common.string_utils import unsanitize_name
from ax.utils.common.sympy import (
    extract_coefficient_dict_from_equality,
    extract_coefficient_dict_from_inequality,
)
from pyre_extensions import none_throws


class ParameterConstraint(SortableBase):
    """Base class for linear parameter constraints.

    Supports both inequality constraints (``w^T x <= b``) and equality
    constraints (``w^T x == b``).  Exactly one of ``inequality`` or
    ``equality`` must be provided.
    """

    def __init__(
        self,
        inequality: str | None = None,
        *,
        equality: str | None = None,
    ) -> None:
        """Initialize ParameterConstraint.

        Args:
            inequality: String representation of a linear inequality
                constraint, e.g. ``"x1 + x2 <= 3"``.
            equality: String representation of a linear equality
                constraint, e.g. ``"x1 + x2 == 3"``.

        Exactly one of ``inequality`` or ``equality`` must be provided.
        """
        if (inequality is None) == (equality is None):
            raise UserInputError(
                "Exactly one of `inequality` or `equality` must be provided."
            )

        self.is_equality: bool = equality is not None
        constraint_str = none_throws(equality if self.is_equality else inequality)
        self._constraint_str: str = constraint_str

        if self.is_equality:
            coefficient_dict = extract_coefficient_dict_from_equality(
                equality_str=constraint_str
            )
        else:
            coefficient_dict = extract_coefficient_dict_from_inequality(
                inequality_str=constraint_str
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
                constraint_type = "equality" if self.is_equality else "inequality"
                other_type = "inequality" if self.is_equality else "equality"
                raise UserInputError(
                    f"Only linear {constraint_type} parameter constraints are "
                    f"supported, found {constraint_str}. "
                    f"Did you mean to use the `{other_type}` argument?"
                )

    @property
    def constraint_dict(self) -> dict[str, float]:
        """Get mapping from parameter names to weights."""
        return self._constraint_dict

    @property
    def bound(self) -> float:
        """Get bound of the constraint."""
        return self._bound

    @bound.setter
    def bound(self, bound: float) -> None:
        """Set bound."""
        self._bound = bound

    def check(self, parameter_dict: dict[str, int | float]) -> bool:
        """Whether or not the set of parameter values satisfies the constraint.

        Does a weighted sum of the parameter values based on the constraint_dict
        and checks against the bound. For inequality constraints checks
        ``weighted_sum <= bound``; for equality constraints checks
        ``|weighted_sum - bound| <= tolerance``.

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
        if self.is_equality:
            return abs(weighted_sum - self._bound) <= 1e-8
        return weighted_sum <= self._bound + 1e-8  # allow for numerical imprecision

    def clone(self) -> ParameterConstraint:
        """Clone."""
        if self.is_equality:
            return ParameterConstraint(equality=self._constraint_str)
        return ParameterConstraint(inequality=self._constraint_str)

    def clone_with_transformed_parameters(
        self, transformed_parameters: dict[str, Parameter]
    ) -> ParameterConstraint:
        """Clone, but replaced parameters with transformed versions."""
        return self.clone()

    def __repr__(self) -> str:
        op = "==" if self.is_equality else "<="
        return (
            "ParameterConstraint("
            + " + ".join(f"{v}*{k}" for k, v in sorted(self.constraint_dict.items()))
            + f" {op} {self._bound})"
        )

    @property
    def _unique_id(self) -> str:
        return str(self)


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
        if isinstance(parameter, RangeParameter):
            # Log parameters require a non-linear transformation, and Ax
            # models only support linear constraints.
            if parameter.log_scale:
                raise ValueError(
                    "Parameter constraints are not allowed on log scale parameters."
                )
        elif isinstance(parameter, ChoiceParameter):
            if not parameter.is_numeric:
                raise ValueError(
                    "Only numerical ChoiceParameters can be used in parameter "
                    f"constraints. Found {parameter.name} with type "
                    f"{parameter.parameter_type.name}. "
                )
            if not parameter.is_ordered:
                raise ValueError(
                    "Only ordered ChoiceParameters can be used in parameter "
                    f"constraints. Found {parameter.name} with is_ordered=False."
                )
            # Log parameters require a non-linear transformation, and Ax
            # models only support linear constraints.
            if parameter.log_scale:
                raise ValueError(
                    "Parameter constraints are not allowed on log scale parameters."
                )
        else:
            raise ValueError(
                "Parameters in constraints must be RangeParameters or numerical "
                f"ordered ChoiceParameters. Found {parameter}"
            )

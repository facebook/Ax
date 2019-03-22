#!/usr/bin/env python3
# pyre-strict

from typing import Dict, List, Union

from ax.core.base import Base
from ax.core.types import ComparisonOp


class ParameterConstraint(Base):
    """Base class for linear parameter constraints.

    Constraints are expressed using a map from parameter name to weight
    followed by a bound.

    The constraint is satisfied if w * v <= b where:
        w is the vector of parameter weights.
        v is a vector of parameter values.
        b is the specified bound.
        * is the dot product operator.
    """

    def __init__(self, constraint_dict: Dict[str, float], bound: float) -> None:
        """Initialize ParameterConstraint

        Args:
            constraint_dict: Map from parameter name to weight.
            bound: Bound of the inequality of the constraint.
        """
        self._constraint_dict = constraint_dict
        self._bound = bound

    @property
    def constraint_dict(self) -> Dict[str, float]:
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

    def check(self, parameter_dict: Dict[str, Union[int, float]]) -> bool:
        """Whether or not the set of parameter values satisfies the constraint.

        Does a weighted sum of the parameter values based on the constraint_dict
        and checks that the sum is less than the bound.

        Args:
            parameter_dict: Map from parameter name to parameter value.

        Returns:
            Whether the constraint is satisfied.
        """
        for parameter in self.constraint_dict.keys():
            if parameter not in parameter_dict.keys():
                raise ValueError(f"`{parameter}` not present in param_dict.")

        weighted_sum = sum(
            float(parameter_dict[param]) * weight
            for param, weight in self.constraint_dict.items()
        )
        # Expected `int` for 2nd anonymous parameter to call `int.__le__` but got
        # `float`.
        # pyre-fixme[6]: Expected `int` for 1st param but got `float`.
        return weighted_sum <= self._bound

    def clone(self) -> "ParameterConstraint":
        return ParameterConstraint(
            constraint_dict=self._constraint_dict, bound=self._bound
        )

    def __repr__(self) -> str:
        return (
            "ParameterConstraint("
            + " + ".join("{}*{}".format(v, k) for k, v in self.constraint_dict.items())
            + " <= {})".format(self._bound)
        )


class OrderConstraint(ParameterConstraint):
    """Constraint object for specifying one parameter to be smaller than another."""

    _bound: float

    def __init__(self, lower_name: str, upper_name: str) -> None:
        """Initialize OrderConstraint

        Args:
            lower_name: Name of parameter that should have the lower value.
            upper_name: Name of parameter that should have the higher value.

        Note:
            The constraint p1 <= p2 can be expressed in matrix notation as
            [1, -1] * [p1, p2]^T <= 0.
        """

        self._lower_name = lower_name
        self._upper_name = upper_name
        self._bound = 0.0

    @property
    def lower_name(self) -> str:
        return self._lower_name

    @property
    def upper_name(self) -> str:
        return self._upper_name

    @property
    def constraint_dict(self) -> Dict[str, float]:
        return {self._lower_name: 1.0, self._upper_name: -1.0}

    def clone(self) -> "OrderConstraint":
        return OrderConstraint(lower_name=self._lower_name, upper_name=self._upper_name)

    def __repr__(self) -> str:
        return "OrderConstraint({} <= {})".format(self._lower_name, self._upper_name)


class SumConstraint(ParameterConstraint):
    """Constraint on the sum of parameters being greater or less than a bound.
    """

    def __init__(
        self, parameter_names: List[str], is_upper_bound: bool, bound: float
    ) -> None:
        """Initialize SumConstraint

        Args:
            parameter_names: List of parameters whose sum to constrain on.
            is_upper_bound: Whether the bound is an upper or lower bound on the sum.
            bound: The bound on the sum.
        """
        if len(set(parameter_names)) < len(parameter_names):
            raise ValueError("Parameter names are not unique.")

        self._is_upper_bound: bool = is_upper_bound
        self._parameter_names: List[str] = parameter_names
        self._bound: float = self._inequality_weight * bound
        self._constraint_dict: Dict[str, float] = {
            name: self._inequality_weight for name in self._parameter_names
        }

    @property
    def constraint_dict(self) -> Dict[str, float]:
        return self._constraint_dict

    @property
    def op(self) -> ComparisonOp:
        """Whether the sum is constrained by a <= or >= inequality."""
        return ComparisonOp.LEQ if self._is_upper_bound else ComparisonOp.GEQ

    def clone(self) -> "SumConstraint":
        return SumConstraint(
            parameter_names=self._parameter_names,
            is_upper_bound=self._is_upper_bound,
            bound=self._bound,
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
            + " + ".join(self.constraint_dict.keys())
            + " {} {})".format(
                symbol, self._bound if self.op == ComparisonOp.LEQ else -self._bound
            )
        )

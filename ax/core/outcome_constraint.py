#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import re
import warnings
from functools import cached_property

from ax.core.metric import Metric
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.utils.common.base import SortableBase
from ax.utils.common.string_utils import unsanitize_name
from ax.utils.common.sympy import (
    build_constraint_expression_str,
    extract_coefficient_dict_from_inequality,
)


class OutcomeConstraint(SortableBase):
    """Class for representing outcome constraints via an expression string.

    An outcome constraint is an inequality of the form
    ``metric >= bound`` or ``metric <= bound``, where the bound can be
    absolute or relative.

    To indicate a relative constraint (i.e. performance relative to some baseline)
    multiply your bound by "baseline". For example "qps >= 0.95 * baseline" will
    constrain such that the QPS is at least 95% of the baseline arm's QPS.
    The "baseline" in these expressions refers to the experiment's
    ``status_quo`` arm.

    Examples::

        OutcomeConstraint(expression="qps >= 700")
        OutcomeConstraint(expression="loss <= 0.5")
        OutcomeConstraint(expression="2*m1 + 3*m2 <= 10")
        OutcomeConstraint(expression="latency <= 0.95 * baseline", )

    Attributes:
        expression: The expression string defining this constraint.
    """

    def __init__(
        self,
        expression: str | None = None,
        *,
        relative: bool | None = None,
        # Deprecated backward-compat kwargs
        metric: Metric | None = None,
        op: ComparisonOp | None = None,
        bound: float | None = None,
    ) -> None:
        """Create a new outcome constraint.

        Args:
            expression: A string expression defining the constraint, e.g.
                ``"qps >= 700"`` or ``"loss <= 0.5 * baseline"``.
            relative: If provided, overrides the relativity inferred from the
                expression string. If ``True``, the bound is interpreted as
                a percent-difference from the status-quo arm's metric value.
            metric: *Deprecated.* A single metric to constrain. Provide
                ``expression`` instead.
            op: *Deprecated.* Only used with ``metric``.
            bound: *Deprecated.* Only used with ``metric``.
        """
        if metric is not None:
            # Backward-compat path
            warnings.warn(
                "Passing `metric` to OutcomeConstraint is deprecated. "
                "Use `expression` instead, e.g. "
                f'OutcomeConstraint(expression="{metric.name} >= 0").',
                DeprecationWarning,
                stacklevel=2,
            )
            if expression is not None:
                raise UserInputError("Cannot specify both `expression` and `metric`.")
            if op is None:
                raise UserInputError(
                    "Must specify `op` when using the deprecated `metric` kwarg."
                )
            if bound is None:
                raise UserInputError(
                    "Must specify `bound` when using the deprecated `metric` kwarg."
                )
            inferred_relative = relative if relative is not None else False
            expression = build_constraint_expression_str(
                metric_weights=[(metric.name, 1.0)],
                op=_op_to_str(op),
                bound=float(bound),
                relative=inferred_relative,
            )
            relative = inferred_relative

        if expression is None:
            raise UserInputError(
                "An expression string is required. "
                "Example: OutcomeConstraint(expression='qps >= 700') or "
                "OutcomeConstraint(expression='loss <= 0.5')."
            )

        self._expression_str: str = expression
        # Eagerly validate the expression so errors surface at construction time.
        _ = self._parsed

    @cached_property
    def _parsed(
        self,
    ) -> tuple[list[tuple[str, float]], ComparisonOp, float, bool]:
        """Parse the expression string into its constituent parts.

        Returns a tuple of
        ``(metric_weights, op, bound, relative)`` where ``metric_weights``
        is a list of ``(metric_name, weight)`` pairs. All public properties
        -- ``metric_names``, ``metric_weights``, ``op``, ``bound``, and
        ``relative`` -- delegate to this parsed representation.
        """
        return _parse_constraint_expression(self._expression_str)

    @property
    def expression(self) -> str:
        """Get the expression string defining this constraint."""
        return self._expression_str

    @property
    def metric_names(self) -> list[str]:
        """Get a list of all metric names referenced in the constraint."""
        return [name for name, _ in self._parsed[0]]

    @property
    def metric_weights(self) -> list[tuple[str, float]]:
        """Get (metric_name, weight) pairs for this constraint."""
        return list(self._parsed[0])

    @property
    def op(self) -> ComparisonOp:
        """Get the comparison operator."""
        return self._parsed[1]

    @property
    def bound(self) -> float:
        """Get the constraint bound."""
        return self._parsed[2]

    @property
    def relative(self) -> bool:
        """Whether the bound is relative to the status-quo arm."""
        return self._parsed[3]

    def clone(self) -> OutcomeConstraint:
        """Create a copy of this OutcomeConstraint."""
        return OutcomeConstraint(expression=self._expression_str)

    def __repr__(self) -> str:
        return f"OutcomeConstraint({self._expression_str})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OutcomeConstraint):
            return False
        return self._expression_str == other._expression_str

    def __hash__(self) -> int:
        return hash(self._expression_str)

    @property
    def _unique_id(self) -> str:
        return str(self)


class ObjectiveThreshold(OutcomeConstraint):
    """*Deprecated.* Use ``OutcomeConstraint`` directly.

    An objective threshold represents the threshold for an objective metric
    to contribute to hypervolume calculations.

    Example::

        # Old
        ObjectiveThreshold(metric=m, bound=0.5, relative=False)
        # New
        OutcomeConstraint(expression="m >= 0.5", relative=False)
    """

    def __init__(
        self,
        metric: Metric,
        bound: float,
        relative: bool = True,
        op: ComparisonOp | None = None,
    ) -> None:
        warnings.warn(
            "ObjectiveThreshold is deprecated. Use OutcomeConstraint "
            "with an expression string instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if metric.lower_is_better is None and op is None:
            raise ValueError(
                f"Metric {metric} must have attribute `lower_is_better` set or "
                f"op {op} must be manually specified."
            )
        elif op is None:
            op = ComparisonOp.LEQ if metric.lower_is_better else ComparisonOp.GEQ

        expression = build_constraint_expression_str(
            metric_weights=[(metric.name, 1.0)],
            op=_op_to_str(op),
            bound=float(bound),
            relative=relative,
        )
        super().__init__(expression=expression, relative=relative)

    def clone(self) -> ObjectiveThreshold:
        """Create a copy of this ObjectiveThreshold."""
        ot = ObjectiveThreshold.__new__(ObjectiveThreshold)
        ot._expression_str = self._expression_str
        return ot

    def __repr__(self) -> str:
        return f"ObjectiveThreshold({self._expression_str})"


class ScalarizedOutcomeConstraint(OutcomeConstraint):
    """*Deprecated.* Use ``OutcomeConstraint`` with a scalarized expression.

    Example::

        # Old
        ScalarizedOutcomeConstraint(
            metrics=[m1, m2], op=ComparisonOp.LEQ, bound=10,
            weights=[2.0, 3.0],
        )
        # New
        OutcomeConstraint(expression="2.0*m1 + 3.0*m2 <= 10", relative=False)
    """

    def __init__(
        self,
        metrics: list[Metric],
        op: ComparisonOp,
        bound: float,
        relative: bool = True,
        weights: list[float] | None = None,
    ) -> None:
        warnings.warn(
            "ScalarizedOutcomeConstraint is deprecated. Use OutcomeConstraint "
            "with a scalarized expression instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if weights is None:
            weights = [1.0 / len(metrics)] * len(metrics)
        elif len(weights) != len(metrics):
            raise ValueError("Length of weights must equal length of metrics")

        metric_weights = [(m.name, w) for m, w in zip(metrics, weights)]
        expression = build_constraint_expression_str(
            metric_weights=metric_weights,
            op=_op_to_str(op),
            bound=float(bound),
            relative=relative,
        )
        super().__init__(expression=expression, relative=relative)

    @property
    def weights(self) -> list[float]:
        """Get the scalarization weights, derived from the expression."""
        return [w for _, w in self.metric_weights]

    def clone(self) -> ScalarizedOutcomeConstraint:
        """Create a copy of this ScalarizedOutcomeConstraint."""
        cloned = ScalarizedOutcomeConstraint.__new__(ScalarizedOutcomeConstraint)
        cloned._expression_str = self._expression_str
        return cloned

    def __repr__(self) -> str:
        return f"ScalarizedOutcomeConstraint({self._expression_str})"


def _op_to_str(op: ComparisonOp) -> str:
    """Convert a ComparisonOp enum to its string representation."""
    return ">=" if op == ComparisonOp.GEQ else "<="


def _parse_constraint_expression(
    expression_str: str,
) -> tuple[list[tuple[str, float]], ComparisonOp, float, bool]:
    """
    Parse an outcome constraint string into an OutcomeConstraint object using SymPy.
    Currently only supports linear constraints of the form "a * x + b * y >= k" or
    "a * x + b * y <= k".

    To indicate a relative constraint (i.e. performance relative to some baseline)
    multiply your bound by "baseline". For example "qps >= 0.95 * baseline" will
    constrain such that the QPS is at least 95% of the baseline arm's QPS.

    Returns:
        * A list of (metric_name, weight) pairs.
        * The comparison operator.
        * The bound. If the constraint is relative, the bound specifies percent-
            difference from the observed metric value on the status-quo arm. That is,
            the bound's value will be ``(1 + bound / 100.0) * baseline``.
        * Whether the bound is relative to the status-quo arm.
    """
    # Step 1: Detect and strip "baseline" before sympy parsing.
    # This avoids sympy's direction-dependent normalization affecting the
    # coefficient sign of the baseline term.
    is_relative = False
    if "baseline" in expression_str:
        is_relative = True
        # Try to strip "* baseline" (with optional whitespace)
        cleaned = re.sub(r"\s*\*\s*baseline\b", "", expression_str)
        if cleaned != expression_str:
            expression_str = cleaned
        else:
            # Standalone "baseline" (no multiplier) means multiplier = 1
            expression_str = re.sub(r"\bbaseline\b", "1", expression_str)

    # Step 2: Parse with sympy (no baseline symbol)
    coefficient_dict = extract_coefficient_dict_from_inequality(
        inequality_str=expression_str
    )

    # Step 3: Extract metrics and bound
    constraint_dict: dict[str, float] = {}
    bound = 0.0
    for term, coefficient in coefficient_dict.items():
        if term.is_symbol:
            constraint_dict[term.name] = coefficient
        elif term.is_number:
            # Invert because we are "moving" the bound to the right hand side
            bound = -1 * coefficient
        else:
            raise UserInputError(
                f"Only linear outcome constraints are supported, found {expression_str}"
            )

    # Step 4: Normalize and convert multiplier -> percentage for relative
    if len(constraint_dict) == 1:
        term, coefficient = next(iter(constraint_dict.items()))
        bound = bound / coefficient + 0.0  # +0.0 normalizes -0.0 to 0.0
        if is_relative:
            bound = round((bound - 1) * 100, 10)

        return (
            [(unsanitize_name(term), 1)],
            ComparisonOp.LEQ if coefficient > 0 else ComparisonOp.GEQ,
            bound,
            is_relative,
        )

    names, coefficients = zip(*constraint_dict.items())

    # Detect GEQ: sympy normalizes all inequalities to LEQ by negating both
    # sides for GEQ. If all coefficients are negative, flip back.
    if all(c < 0 for c in coefficients):
        op = ComparisonOp.GEQ
        coefficients = tuple(-c for c in coefficients)
        bound = -bound
    else:
        op = ComparisonOp.LEQ

    if is_relative:
        bound = round((bound - 1) * 100, 10)

    return (
        [
            (unsanitize_name(name), coefficient)
            for name, coefficient in zip(names, coefficients)
        ],
        op,
        bound,
        is_relative,
    )

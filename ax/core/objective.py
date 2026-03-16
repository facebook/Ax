#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import warnings
from typing import Self

from ax.core.metric import Metric
from ax.exceptions.core import UserInputError
from ax.utils.common.base import SortableBase
from ax.utils.common.sympy import (
    extract_metric_names_from_objective_expr,
    extract_metric_weights_from_objective_expr,
    parse_objective_expression,
)


class Objective(SortableBase):
    """Class for representing an optimization objective via an expression string.

    An objective is defined by an expression string that describes the quantity
    to be **maximized**. To minimize a metric, negate it in the expression
    (e.g. ``"-loss"``).

    Examples::

        Objective(expression="accuracy")           # maximize accuracy
        Objective(expression="-loss")              # minimize loss
        Objective(expression="2*acc + recall")     # scalarized objective
        Objective(expression="acc, -loss")         # multi-objective

    Attributes:
        expression: The expression string defining this objective.
    """

    def __init__(
        self,
        expression: str | None = None,
        *,
        # Deprecated backward-compat kwargs
        metric: Metric | None = None,
        minimize: bool | None = None,
    ) -> None:
        """Create a new objective.

        Args:
            expression: A string expression defining the objective. Metrics are
                referenced by name. The expression is always interpreted as
                something to be **maximized**. Use negation to minimize (e.g.
                ``"-loss"``). Comma-separated expressions define a
                multi-objective (e.g. ``"accuracy, -loss"``).
            metric: *Deprecated.* A single metric to optimize. Provide
                ``expression`` instead.
            minimize: *Deprecated.* Only used together with ``metric``. If
                ``True`` the expression is negated so the metric is minimized.
        """
        if metric is not None:
            # Backward-compat path
            warnings.warn(
                "Passing `metric` to Objective is deprecated. "
                "Use `expression` instead, e.g. "
                f'Objective(expression="{metric.name}") or '
                f'Objective(expression="-{metric.name}").',
                DeprecationWarning,
                stacklevel=2,
            )
            if expression:
                raise UserInputError("Cannot specify both `expression` and `metric`.")
            # Resolve minimize
            lower_is_better = metric.lower_is_better
            if minimize is None:
                if lower_is_better is None:
                    raise UserInputError(
                        f"Metric {metric.name} does not specify `lower_is_better` "
                        "and `minimize` is not specified. At least one of these "
                        "must be specified."
                    )
                else:
                    minimize = lower_is_better
            elif lower_is_better is not None and lower_is_better != minimize:
                raise UserInputError(
                    f"Metric {metric.name} specifies {lower_is_better=}, "
                    "which doesn't match the specified optimization direction "
                    f"{minimize=}."
                )
            expression = f"-{metric.name}" if minimize else metric.name

        if expression is None:
            raise UserInputError(
                "An expression string is required. "
                "Example: Objective(expression='accuracy') or "
                "Objective(expression='-loss')."
            )

        self._expression_str: str = expression

        # Eagerly validate: error on duplicate metric names
        parsed = parse_objective_expression(expression)
        sub_exprs = parsed if isinstance(parsed, tuple) else (parsed,)
        seen: list[str] = []
        for sub_expr in sub_exprs:
            for name in extract_metric_names_from_objective_expr(sub_expr):
                if name in seen:
                    raise UserInputError(
                        f"Metric '{name}' appears more than once in the objective "
                        f"expression '{expression}'."
                    )
                seen.append(name)

    @property
    def expression(self) -> str:
        """Get the expression string defining this objective."""
        return self._expression_str

    @property
    def metric_names(self) -> list[str]:
        """Get a list of all metric names referenced in the expression."""
        parsed = parse_objective_expression(self._expression_str)
        sub_exprs = parsed if isinstance(parsed, tuple) else (parsed,)
        names: list[str] = []
        for sub_expr in sub_exprs:
            for name in extract_metric_names_from_objective_expr(sub_expr):
                if name not in names:
                    names.append(name)
        return names

    @property
    def metric_weights(self) -> list[tuple[str, float]]:
        """Get a list of (metric_name, weight) tuples in the expression."""
        parsed = parse_objective_expression(self._expression_str)
        sub_exprs = parsed if isinstance(parsed, tuple) else (parsed,)

        result: list[tuple[str, float]] = []
        for sub_expr in sub_exprs:
            result.extend(extract_metric_weights_from_objective_expr(sub_expr))

        return result

    @property
    def is_multi_objective(self) -> bool:
        """True if the objective has multiple comma-separated
        sub-objectives."""
        return isinstance(parse_objective_expression(self._expression_str), tuple)

    @property
    def is_scalarized_objective(self) -> bool:
        """True if the objective is a single linear combination of
        multiple metrics."""
        return not self.is_multi_objective and len(self.metric_names) > 1

    @property
    def is_single_objective(self) -> bool:
        """True if the objective has exactly one metric and is not multi."""
        return not self.is_multi_objective and not self.is_scalarized_objective

    @property
    def minimize(self) -> bool:
        """Whether this objective is minimizing.

        Only meaningful for single-metric objectives. A negative coefficient
        in the expression indicates minimization.

        Raises:
            UserInputError: If the objective is multi or scalarized.
        """
        if self.is_multi_objective or self.is_scalarized_objective:
            raise UserInputError(
                "`minimize` is only defined for single-metric objectives. "
                "For scalarized or multi-objectives, inspect the expression "
                "or metric_weights directly."
            )
        weights = self.metric_weights
        if len(weights) == 1:
            return weights[0][1] < 0
        return False

    def get_unconstrainable_metric_names(self) -> list[str]:
        """Return metric names that are incompatible with
        OutcomeConstraints."""
        return self.metric_names

    def clone(self) -> Self:
        """Create a copy of the objective."""
        return self.__class__(expression=self._expression_str)

    def __repr__(self) -> str:
        return f'Objective(expression="{self._expression_str}")'

    @property
    def _unique_id(self) -> str:
        return str(self)


class MultiObjective(Objective):
    """*Deprecated.* Use ``Objective`` with a comma-separated expression
    instead.

    Example::

        # Old
        MultiObjective(objectives=[
            Objective(expression="acc"),
            Objective(expression="-loss"),
        ])
        # New
        Objective(expression="acc, -loss")
    """

    def __init__(self, objectives: list[Objective]) -> None:
        warnings.warn(
            "MultiObjective is deprecated. Use Objective with a "
            "comma-separated expression instead, e.g. "
            'Objective(expression="acc, -loss").',
            DeprecationWarning,
            stacklevel=2,
        )
        if any(isinstance(o, ScalarizedObjective) for o in objectives):
            raise NotImplementedError(
                "Scalarized objectives are not supported for a `MultiObjective`."
            )
        expression = ", ".join(obj.expression for obj in objectives)
        super().__init__(expression=expression)

    def clone(self) -> Objective:  # pyre-ignore[15]: Inconsistent override
        """Clone as a base Objective (MultiObjective is deprecated)."""
        return Objective(expression=self._expression_str)


class ScalarizedObjective(Objective):
    """*Deprecated.* Use ``Objective`` with a weighted expression instead.

    Example::

        # Old
        ScalarizedObjective(
            metrics=[m1, m2], weights=[2.0, 1.0], minimize=True
        )
        # New
        Objective(expression="-2*m1 - m2")
    """

    def __init__(
        self,
        metrics: list[Metric],
        weights: list[float] | None = None,
        minimize: bool = False,
    ) -> None:
        warnings.warn(
            "ScalarizedObjective is deprecated. Use Objective with a "
            "weighted expression instead, e.g. "
            'Objective(expression="2*m1 + m2").',
            DeprecationWarning,
            stacklevel=2,
        )
        if weights is None:
            weights = [1.0 for _ in metrics]
        else:
            if len(weights) != len(metrics):
                raise ValueError("Length of weights must equal length of metrics")

        # Validate lower_is_better consistency
        for m, w in zip(metrics, weights):
            is_minimized = minimize if w > 0 else not minimize
            if m.lower_is_better is not None and is_minimized != m.lower_is_better:
                raise ValueError(
                    f"Metric with name {m.name} specifies `lower_is_better` = "
                    f"{m.lower_is_better}, which doesn't match the specified "
                    "optimization direction. You most likely want to flip the "
                    "sign of the corresponding metric weight."
                )

        # Build expression string
        # When minimize=True, flip sign so expression represents maximization
        sign = -1.0 if minimize else 1.0
        parts: list[str] = []
        for m, w in zip(metrics, weights):
            effective_w = sign * w
            if effective_w == 1.0:
                parts.append(m.name)
            elif effective_w == -1.0:
                parts.append(f"-{m.name}")
            else:
                parts.append(f"{effective_w}*{m.name}")

        expression_str = " + ".join(parts).replace(" + -", " - ")
        super().__init__(expression=expression_str)

    def clone(self) -> Objective:  # pyre-ignore[15]: Inconsistent override
        """Clone as a base Objective (ScalarizedObjective is deprecated)."""
        return Objective(expression=self._expression_str)

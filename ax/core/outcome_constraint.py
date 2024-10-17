#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
from collections.abc import Iterable

from ax.core.metric import Metric
from ax.core.types import ComparisonOp
from ax.utils.common.base import SortableBase
from ax.utils.common.logger import get_logger


logger: logging.Logger = get_logger(__name__)

CONSTRAINT_WARNING_MESSAGE: str = (
    "Constraint on {name} appears invalid: {bound} bound on metric "
    + "for which {is_better} values are better."
)
LOWER_BOUND_MISMATCH: dict[str, str] = {"bound": "Lower", "is_better": "lower"}
UPPER_BOUND_MISMATCH: dict[str, str] = {"bound": "Upper", "is_better": "higher"}

CONSTRAINT_THRESHOLD_WARNING_MESSAGE: str = (
    "Constraint threshold on {name} appears invalid: {bound} bound on metric "
    + "for which {is_better} values is better."
)
UPPER_BOUND_THRESHOLD: dict[str, str] = {"bound": "Positive", "is_better": "lower"}
LOWER_BOUND_THRESHOLD: dict[str, str] = {"bound": "Negative", "is_better": "higher"}


class OutcomeConstraint(SortableBase):
    """Base class for representing outcome constraints.

    Outcome constraints may of the form metric >= bound or metric <= bound,
    where the bound can be expressed as an absolute measurement or relative
    to the status quo (if applicable).

    Attributes:
        metric: Metric to constrain.
        op: Specifies whether metric should be greater or equal
            to, or less than or equal to, some bound.
        bound: The bound in the constraint.
        relative: [default ``True``] Whether the provided bound value is relative to
            some status-quo arm's metric value. If False, ``bound`` is interpreted as an
            absolute number, else ``bound`` specifies percent-difference from the
            observed metric value on the status-quo arm. That is, the bound's value will
            be ``(1 + sign * bound/100.0) * status_quo_metric_value``, where `sign` is
            the sign of status_quo_metric_value. This ensures that a positive relative
            bound gives rise to an absolute upper bound, even if the status-quo arm has
            a negative metric value. This requires specification of a status-quo arm in
            ``Experiment``.
    """

    def __init__(
        self, metric: Metric, op: ComparisonOp, bound: float, relative: bool = True
    ) -> None:
        self._validate_metric_constraint_op(metric=metric, op=op)
        self._metric = metric
        self._op = op
        self.bound = bound
        self.relative = relative

    @property
    def metric(self) -> Metric:
        return self._metric

    @metric.setter
    def metric(self, metric: Metric) -> None:
        self._validate_metric_constraint_op(metric=metric, op=self.op)
        self._metric = metric

    @property
    def op(self) -> ComparisonOp:
        return self._op

    @op.setter
    def op(self, op: ComparisonOp) -> None:
        self._validate_metric_constraint_op(metric=self.metric, op=op)
        self._op = op

    def clone(self) -> OutcomeConstraint:
        """Create a copy of this OutcomeConstraint."""
        return OutcomeConstraint(
            metric=self.metric.clone(),
            op=self.op,
            bound=self.bound,
            relative=self.relative,
        )

    @staticmethod
    def _validate_metric_constraint_op(
        metric: Metric, op: ComparisonOp
    ) -> tuple[bool, str]:
        """Ensure constraint is compatible with metric definition.

        Args:
            metric: Metric to constrain.
            op: Specifies whether metric should be greater or equal
                to, or less than or equal to, some bound. In case metric has:
                - lower_is_better=True, op is interpreted as an upper bound and
                    should be LEQ.
                - lower_is_better=False, op is interpreted as a lower bound and
                    should be GEQ.

        Returns: A tuple consisting of
            - A boolean indicating whether the constraint is valid,
            - A string containing a warning message if the constraint is invalid.

        """
        fmt_data = None
        if metric.lower_is_better is not None:
            if op == ComparisonOp.GEQ and metric.lower_is_better:
                fmt_data = LOWER_BOUND_MISMATCH
            if op == ComparisonOp.LEQ and not metric.lower_is_better:
                fmt_data = UPPER_BOUND_MISMATCH
        if fmt_data is not None:
            fmt_data["name"] = metric.name
            msg = CONSTRAINT_WARNING_MESSAGE.format(**fmt_data)
            logger.debug(msg)
            return False, msg
        return True, ""

    def _validate_constraint(self) -> tuple[bool, str]:
        """Ensure constraint is compatible with metric definition.
        In case metric has:
            - lower_is_better=True: op is interpreted as an upper bound
                and should be LEQ; bound should be positive in case
                of relative constraint.
            - lower_is_better=False, op is interpreted as a lower bound
                and should be GEQ; bound should be negative in case
                of relative constraint.

        Returns: A tuple consisting of
            - A boolean indicating whether the constraint is valid,
            - A string containing a warning message if the constraint is invalid.
        """
        valid_op, msg = self._validate_metric_constraint_op(
            metric=self.metric, op=self.op
        )
        if not valid_op:
            return False, msg

        if not self.relative:
            return True, ""

        fmt_data = None
        if self.metric.lower_is_better is not None:
            if self.bound < 0 and self.metric.lower_is_better:
                fmt_data = UPPER_BOUND_THRESHOLD
            if self.bound > 0 and not self.metric.lower_is_better:
                fmt_data = LOWER_BOUND_THRESHOLD
        if fmt_data is not None:
            fmt_data["name"] = self.metric.name
            msg += CONSTRAINT_THRESHOLD_WARNING_MESSAGE.format(**fmt_data)
            logger.debug(msg)
            return False, msg

        return True, ""

    def __repr__(self) -> str:
        op = ">=" if self.op == ComparisonOp.GEQ else "<="
        relative = "%" if self.relative else ""
        return f"OutcomeConstraint({self.metric.name} {op} {self.bound}{relative})"

    @property
    def _unique_id(self) -> str:
        return str(self)


class ObjectiveThreshold(OutcomeConstraint):
    """Class for representing Objective Thresholds.

    An objective threshold represents the threshold for an objective metric
    to contribute to hypervolume calculations. A list containing the objective
    threshold for each metric collectively form a reference point.

    Objective thresholds may bound the metric from above or from below.
    The bound can be expressed as an absolute measurement or relative
    to the status quo (if applicable).

    The direction of the bound is inferred from the Metric's lower_is_better attribute.

    Attributes:
        metric: Metric to constrain.
        bound: The bound in the constraint.
        relative: Whether you want to bound on an absolute or relative scale. If
            relative, bound is the acceptable percent change. That is, the bound's value
            will be ``(1 + sign * bound/100.0) * status_quo_metric_value``, where `sign`
            is the sign of status_quo_metric_value, ensuring that a positive relative
            bound gives rise to an absolute upper bound, even if the status-quo arm has
            a negative metric value. This requires specification of a status-quo arm in
            ``Experiment``.

        op: automatically inferred, but manually overwritable.
            specifies whether metric should be greater or equal to, or less
            than or equal to, some bound.
    """

    def __init__(
        self,
        metric: Metric,
        bound: float,
        relative: bool = True,
        op: ComparisonOp | None = None,
    ) -> None:
        if metric.lower_is_better is None and op is None:
            raise ValueError(
                f"Metric {metric} must have attribute `lower_is_better` set or "
                f"op {op} must be manually specified."
            )
        elif op is None:
            op = ComparisonOp.LEQ if metric.lower_is_better else ComparisonOp.GEQ

        # It's likely that the metric passed into the ObjectiveThreshold constructor
        # is the same instance as the metric in the Objective. Thus, we have to clone
        # the metric passed in here to ensure a 1:1 relationship between user-facing
        # objects and DB objects.
        super().__init__(metric=metric.clone(), op=op, bound=bound, relative=relative)

    def clone(self) -> ObjectiveThreshold:
        """Create a copy of this ObjectiveThreshold."""
        return ObjectiveThreshold(
            metric=self.metric.clone(),
            bound=self.bound,
            relative=self.relative,
            op=self.op,
        )

    def __repr__(self) -> str:
        op = ">=" if self.op == ComparisonOp.GEQ else "<="
        relative = "%" if self.relative else ""
        return f"ObjectiveThreshold({self.metric.name} {op} {self.bound}{relative})"


class ScalarizedOutcomeConstraint(OutcomeConstraint):
    """Class for presenting outcome constraints composed of a linear
    scalarization of metrics.

    Attributes:
        metrics: List of metrics.
        weights: Weights for scalarization; default to 1.0 / len(metrics).
        op: Specifies whether metric should be greater or equal
            to, or less than or equal to, some bound.
        bound: The bound in the constraint.
        relative: [default ``True``] Whether the provided bound value is relative to
            some status-quo arm's metric value. If False, ``bound`` is interpreted as an
            absolute number, else ``bound`` specifies percent-difference from the
            observed metric value on the status-quo arm. That is, the bound's value will
            be ``(1 + sign * bound/100.0) * status_quo_metric_value``, where sign is the
            sign of status_quo_metric_value. This ensures that a positive relative bound
            always gives rise to an absolute upper bound, even if the status-quo arm has
            a negative metric value. This requires specification of a status-quo arm in
            ``Experiment``.
    """

    weights: list[float]

    def __init__(
        self,
        metrics: list[Metric],
        op: ComparisonOp,
        bound: float,
        relative: bool = True,
        weights: list[float] | None = None,
    ) -> None:
        for metric in metrics:
            self._validate_metric_constraint_op(metric=metric, op=op)

        if weights is None:
            weights = [1.0 / len(metrics)] * len(metrics)
        elif len(weights) != len(metrics):
            raise ValueError("Length of weights must equal length of metrics")
        self._metrics = metrics
        self.weights = weights
        self._op = op
        self.bound = bound
        self.relative = relative

    @property
    def metric_weights(self) -> Iterable[tuple[Metric, float]]:
        """Get the objective metrics and weights."""
        return zip(self.metrics, self.weights)

    @property
    def metrics(self) -> list[Metric]:
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: list[Metric]) -> None:
        for metric in metrics:
            self._validate_metric_constraint_op(metric=metric, op=self.op)
        self._metrics = metrics

    @property
    def metric(self) -> Metric:
        """Override base method to error."""
        raise NotImplementedError(
            f"{type(self).__name__} is composed of multiple metrics"
        )

    @metric.setter
    def metric(self, metric: Metric) -> None:
        """Override base method to error."""
        raise NotImplementedError(
            f"{type(self).__name__} is composed of multiple metrics"
        )

    @property
    def op(self) -> ComparisonOp:
        return self._op

    @op.setter
    def op(self, op: ComparisonOp) -> None:
        for metric in self.metrics:
            self._validate_metric_constraint_op(metric=metric, op=op)
        self._op = op

    def clone(self) -> ScalarizedOutcomeConstraint:
        """Create a copy of this ScalarizedOutcomeConstraint."""
        return ScalarizedOutcomeConstraint(
            metrics=[m.clone() for m in self.metrics],
            op=self.op,
            bound=self.bound,
            relative=self.relative,
            weights=self.weights.copy(),
        )

    def __repr__(self) -> str:
        op = ">=" if self.op == ComparisonOp.GEQ else "<="
        relative = "%" if self.relative else ""
        return (
            "ScalarizedOutcomeConstraint(metric_names={}, weights={}, {} {}{})".format(
                [metric.name for metric in self.metrics],
                self.weights,
                op,
                self.bound,
                relative,
            )
        )

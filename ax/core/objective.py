#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import warnings
from collections.abc import Iterable
from logging import Logger
from typing import Any

from ax.core.metric import Metric
from ax.exceptions.core import UserInputError
from ax.utils.common.base import SortableBase
from ax.utils.common.logger import get_logger
from pyre_extensions import none_throws

logger: Logger = get_logger(__name__)


class Objective(SortableBase):
    """Base class for representing an objective.

    Attributes:
        minimize: If True, minimize metric.
    """

    def __init__(self, metric: Metric, minimize: bool | None = None) -> None:
        """Create a new objective.

        Args:
            metric: The metric to be optimized.
            minimize: If True, minimize metric. If None, will be set based on the
                `lower_is_better` property of the metric (if that is not specified,
                will raise a `UserInputError`).

        """
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
        self._metric: Metric = metric
        self.minimize: bool = none_throws(minimize)

    @property
    def metric(self) -> Metric:
        """Get the objective metric."""
        return self._metric

    @property
    def metrics(self) -> list[Metric]:
        """Get a list of objective metrics."""
        return [self._metric]

    @property
    def metric_names(self) -> list[str]:
        """Get a list of objective metric names."""
        return [m.name for m in self.metrics]

    def clone(self) -> Objective:
        """Create a copy of the objective."""
        return Objective(self.metric.clone(), self.minimize)

    def __repr__(self) -> str:
        return 'Objective(metric_name="{}", minimize={})'.format(
            self.metric.name, self.minimize
        )

    def get_unconstrainable_metrics(self) -> list[Metric]:
        """Return a list of metrics that are incompatible with OutcomeConstraints."""
        return self.metrics

    @property
    def _unique_id(self) -> str:
        return str(self)


class MultiObjective(Objective):
    """Class for an objective composed of a multiple component objectives.

    The Acquisition function determines how the objectives are weighted.

    Attributes:
        objectives: List of objectives.
    """

    weights: list[float]

    def __init__(
        self,
        objectives: list[Objective] | None = None,
        **extra_kwargs: Any,  # Here to satisfy serialization.
    ) -> None:
        """Create a new objective.

        Args:
            objectives: The list of objectives to be jointly optimized.

        """
        # Support backwards compatibility for old API in which
        # MultiObjective constructor accepted `metrics` and `minimize`
        # rather than `objectives`
        if objectives is None:
            if "metrics" not in extra_kwargs:
                raise ValueError(
                    "Must either specify `objectives` or `metrics` "
                    "as input to `MultiObjective` constructor."
                )
            metrics = extra_kwargs["metrics"]
            minimize = extra_kwargs.get("minimize", None)
            warnings.warn(
                "Passing `metrics` and `minimize` as input to the `MultiObjective` "
                "constructor will soon be deprecated. Instead, pass a list of "
                "`objectives`. This will become an error in the future.",
                DeprecationWarning,
                stacklevel=2,
            )
            objectives = []
            for metric in metrics:
                objectives.append(Objective(metric=metric, minimize=minimize))

        # pyre-fixme[4]: Attribute must be annotated.
        self._objectives = none_throws(objectives)

        # For now, assume all objectives are weighted equally.
        # This might be used in the future to change emphasis on the
        # relative focus of the exploration during the optimization.
        self.weights = [1.0 for _ in range(len(objectives))]

    @property
    def metric(self) -> Metric:
        """Override base method to error."""
        raise NotImplementedError(
            f"{type(self).__name__} is composed of multiple metrics"
        )

    @property
    def metrics(self) -> list[Metric]:
        """Get the objective metrics."""
        return [o.metric for o in self._objectives]

    @property
    def objectives(self) -> list[Objective]:
        """Get the objectives."""
        return self._objectives

    @property
    def objective_weights(self) -> Iterable[tuple[Objective, float]]:
        """Get the objectives and weights."""
        return zip(self.objectives, self.weights)

    def clone(self) -> MultiObjective:
        """Create a copy of the objective."""
        return MultiObjective(objectives=[o.clone() for o in self.objectives])

    def __repr__(self) -> str:
        return f"MultiObjective(objectives={self.objectives})"


class ScalarizedObjective(Objective):
    """Class for an objective composed of a linear scalarization of metrics.

    Attributes:
        metrics: List of metrics.
        weights: Weights for scalarization; default to 1.
    """

    weights: list[float]

    def __init__(
        self,
        metrics: list[Metric],
        weights: list[float] | None = None,
        minimize: bool = False,
    ) -> None:
        """Create a new objective.

        Args:
            metric: The metric to be optimized.
            weights: The weights for the linear combination of metrics.
            minimize: If true, minimize the linear combination.

        """
        if weights is None:
            weights = [1.0 for i in range(len(metrics))]
        else:
            if len(weights) != len(metrics):
                raise ValueError("Length of weights must equal length of metrics")

        # Check if the optimization direction is consistent with
        # `lower_is_better` (if specified).
        for m, w in zip(metrics, weights):
            is_minimized = minimize if w > 0 else not minimize
            if m.lower_is_better is not None and is_minimized != m.lower_is_better:
                raise ValueError(
                    f"Metric with name {m.name} specifies `lower_is_better` = "
                    f"{m.lower_is_better}, which doesn't match the specified "
                    "optimization direction. You most likely want to flip the sign of "
                    "the corresponding metric weight."
                )

        self._metrics = metrics
        self.weights = weights
        self.minimize = minimize

    @property
    def metric(self) -> Metric:
        """Override base method to error."""
        raise NotImplementedError(
            f"{type(self).__name__} is composed of multiple metrics"
        )

    @property
    def metrics(self) -> list[Metric]:
        """Get the metrics."""
        return self._metrics

    @property
    def metric_weights(self) -> Iterable[tuple[Metric, float]]:
        """Get the metrics and weights."""
        return zip(self.metrics, self.weights)

    def clone(self) -> ScalarizedObjective:
        """Create a copy of the objective."""
        return ScalarizedObjective(
            metrics=[m.clone() for m in self.metrics],
            weights=self.weights.copy(),
            minimize=self.minimize,
        )

    def __repr__(self) -> str:
        return "ScalarizedObjective(metric_names={}, weights={}, minimize={})".format(
            [metric.name for metric in self.metrics], self.weights, self.minimize
        )

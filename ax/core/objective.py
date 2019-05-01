#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Iterable, List, Optional, Tuple

from ax.core.base import Base
from ax.core.metric import Metric


class Objective(Base):
    """Base class for representing an objective.

    Attributes:
        minimize: If True, minimize metric.
    """

    def __init__(self, metric: Metric, minimize: bool = False) -> None:
        """Create a new objective.

        Args:
            metric: The metric to be optimized.
            minimize: If True, minimize metric.

        """
        self._metric = metric
        self.minimize = minimize

    @property
    def metric(self) -> Metric:
        """Get the objective metric."""
        return self._metric

    @property
    def metrics(self) -> List[Metric]:
        """Get a list of objective metrics."""
        return [self._metric]

    def clone(self) -> "Objective":
        """Create a copy of the objective."""
        return Objective(self.metric.clone(), self.minimize)

    def __repr__(self) -> str:
        return 'Objective(metric_name="{}", minimize={})'.format(
            self.metric.name, self.minimize
        )


class ScalarizedObjective(Objective):
    """Class for an objective composed of a linear scalarization of metrics.

    Attributes:
        metrics: List of metrics.
        weights: Weights for scalarization; default to 1.
    """

    weights: List[float]

    def __init__(
        self,
        metrics: List[Metric],
        weights: Optional[List[float]] = None,
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
        self._metrics = metrics
        self.minimize = minimize
        self.weights = weights

    @property
    def metric(self) -> Metric:
        """Override base method to error."""
        raise NotImplementedError("ScalarizedObjective is composed of multiple metrics")

    @property
    def metrics(self) -> List[Metric]:
        """Get the objective metrics."""
        return self._metrics

    @property
    def metric_weights(self) -> Iterable[Tuple[Metric, float]]:
        """Get the objective metrics and weights."""
        return zip(self.metrics, self.weights)

    def clone(self) -> "Objective":
        """Create a copy of the objective."""
        return ScalarizedObjective(
            metrics=[m.clone() for m in self.metrics],
            weights=self.weights.copy(),
            minimize=self.minimize,
        )

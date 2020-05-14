#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any, Iterable, List, Optional, Tuple

from ax.core.base import Base
from ax.core.metric import Metric
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none


logger = get_logger(__name__)


class Objective(Base):
    """Base class for representing an objective.

    Attributes:
        minimize: If True, minimize metric.
    """

    def __init__(self, metric: Metric, minimize: Optional[bool] = None) -> None:
        """Create a new objective.

        Args:
            metric: The metric to be optimized.
            minimize: If True, minimize metric. If None, will be set based on the
                `lower_is_better` property of the metric (if that is not specified,
                will raise a DeprecationWarning).

        """
        lower_is_better = metric.lower_is_better
        if minimize is None:
            if lower_is_better is None:
                warnings.warn(
                    f"Defaulting to `minimize=False` for metric {metric.name} not "
                    + "specifying `lower_is_better` property. This is a wild guess. "
                    + "Specify either `lower_is_better` on the metric, or specify "
                    + "`minimize` explicitly. This will become an error in the future.",
                    DeprecationWarning,
                )
                minimize = False
            else:
                minimize = lower_is_better
        if lower_is_better is not None:
            if lower_is_better and not minimize:
                warnings.warn(
                    f"Attempting to maximize metric {metric.name} with property "
                    "`lower_is_better=True`."
                )
            elif not lower_is_better and minimize:
                warnings.warn(
                    f"Attempting to minimize metric {metric.name} with property "
                    "`lower_is_better=False`."
                )
        self._metric = metric
        self.minimize = not_none(minimize)

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


# TODO (jej): Support sqa_store encoding. Currenlty only single metric obj supported.
class MultiObjective(Objective):
    """Class for an objective composed of a multiple component objectives.

    The Acquisition function determines how the objectives are weighted.

    Attributes:
        metrics: List of metrics.
    """

    weights: List[float]

    def __init__(
        self,
        metrics: List[Metric],
        minimize: bool = False,
        **extra_kwargs: Any,  # Here to satisfy serialization.
    ) -> None:
        """Create a new objective.

        Args:
            metrics: The list of metrics to be jointly optimized.
            minimize: If true, minimize the aggregate of these metrics.

        """
        self._metrics = metrics
        self.weights = []
        for metric in metrics:
            # Set weights from "lower_is_better"
            if metric.lower_is_better is None:
                logger.warning(
                    f"metric {metric.name} has not set `lower_is_better`. "
                    "Treating as `False` (Metric should be maximized)."
                )
            self.weights.append(-1.0 if metric.lower_is_better is True else 1.0)
        self.minimize = minimize

    @property
    def metric_weights(self) -> Iterable[Tuple[Metric, float]]:
        """Get the objective metrics and weights."""
        return zip(self.metrics, self.weights)

    @property
    def metric(self) -> Metric:
        """Override base method to error."""
        raise NotImplementedError(
            f"{type(self).__name__} is composed of multiple metrics"
        )

    @property
    def metrics(self) -> List[Metric]:
        """Get the objective metrics."""
        return self._metrics

    def clone(self) -> "Objective":
        """Create a copy of the objective."""
        return MultiObjective(
            metrics=[m.clone() for m in self.metrics], minimize=self.minimize
        )

    def __repr__(self) -> str:
        return "MultiObjective(metric_names={}, minimize={})".format(
            [metric.name for metric in self.metrics], self.minimize
        )


class ScalarizedObjective(MultiObjective):
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
        super().__init__(metrics, minimize)
        self.weights = weights

    def clone(self) -> "Objective":
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

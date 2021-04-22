#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Type

from ax.core.abstract_data import AbstractDataFrameData
from ax.core.map_data import MapData
from ax.core.metric import Metric


if TYPE_CHECKING:  # pragma: no cover
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


class MapMetric(Metric):
    """Base class for representing metrics that return `MapData`.

    The `fetch_trial_data` method is the essential method to override when
    subclassing, which specifies how to retrieve a Metric, for a given trial.

    A MapMetric must return a MapData object, which requires (at minimum) the following:
        https://ax.dev/api/_modules/ax/core/abstract_data.html#AbstractDataFrameData.required_columns

    Attributes:
        lower_is_better: Flag for metrics which should be minimized.
        properties: Properties specific to a particular metric.
    """

    # pyre-fixme[15]: Inconsistent override of `Type[Data]` with `Type[MapData]`
    data_constructor: Type[MapData] = MapData

    def __init__(
        self,
        name: str,
        lower_is_better: Optional[bool] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Inits Metric.

        Args:
            name: Name of metric.
            lower_is_better: Flag for metrics which should be minimized.
            properties: Dictionary of this metric's properties
        """
        self._name = name
        self.lower_is_better = lower_is_better
        self.properties = properties or {}

    def fetch_trial_data(
        self, trial: core.base_trial.BaseTrial, **kwargs: Any
    ) -> AbstractDataFrameData:
        """Fetch data for one trial."""
        return super().fetch_trial_data(trial=trial, **kwargs)

    def fetch_experiment_data(
        self, experiment: core.experiment.Experiment, **kwargs: Any
    ) -> AbstractDataFrameData:
        """Fetch this metric's data for an experiment.

        Default behavior is to fetch data from all trials expecting data
        and concatenate the results.
        """
        return super().fetch_experiment_data(experiment=experiment, **kwargs)

    @classmethod
    # pyre-fixme [14]: overrides superclass method inconsistently.
    #   Parameter of type `Iterable[MapMetric]` is not a supertype of
    #   overridden `Iterable[Metric]
    def fetch_trial_data_multi(
        cls,
        trial: core.base_trial.BaseTrial,
        metrics: Iterable[MapMetric],
        **kwargs: Any,
    ) -> AbstractDataFrameData:
        """Fetch multiple metrics data for one trial.

        Default behavior calls `fetch_trial_data` for each metric.
        Subclasses should override this to trial data computation for multiple metrics.
        """
        return super().fetch_trial_data_multi(trial=trial, metrics=metrics, **kwargs)

    @classmethod
    # pyre-fixme [14]: overrides superclass method inconsistently.
    #   Parameter of type `Iterable[MapMetric]` is not a supertype of
    #   overridden `Iterable[Metric]`
    def fetch_experiment_data_multi(
        cls,
        experiment: core.experiment.Experiment,
        metrics: Iterable[MapMetric],
        trials: Optional[Iterable[core.base_trial.BaseTrial]] = None,
        **kwargs: Any,
    ) -> AbstractDataFrameData:
        """Fetch multiple metrics data for an experiment.

        Default behavior calls `fetch_trial_data_multi` for each trial.
        Subclasses should override to batch data computation across trials + metrics.
        """
        return super().fetch_experiment_data_multi(
            experiment=experiment, metrics=metrics, trials=trials, **kwargs
        )

    @classmethod
    # pyre-fixme [14]: overrides superclass method inconsistently.
    #   Parameter of type `Iterable[MapMetric]` is not a supertype of
    #   overridden `Iterable[Metric]`
    def lookup_or_fetch_experiment_data_multi(
        cls,
        experiment: core.experiment.Experiment,
        metrics: Iterable[MapMetric],
        trials: Optional[Iterable[core.base_trial.BaseTrial]] = None,
        **kwargs: Any,
    ) -> AbstractDataFrameData:
        """Fetch or lookup (with fallback to fetching) data for given metrics,
        depending on whether they are available while running.

        If metric is available while running, its data can change (and therefore
        we should always re-fetch it). If metric is available only upon trial
        completion, its data does not change, so we can look up that data on
        the experiment and only fetch the data that is not already attached to
        the experiment.

        NOTE: If fetching data for a metrics class that is only available upon
        trial completion, data fetched in this function (data that was not yet
        available on experiment) will be attached to experiment.
        """
        kwargs["merge_trial_data"] = kwargs.get("merge_trial_data", True)
        return super().lookup_or_fetch_experiment_data_multi(
            experiment=experiment,
            metrics=metrics,
            trials=trials,
            **kwargs,
        )

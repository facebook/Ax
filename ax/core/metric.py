#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, Type, TYPE_CHECKING

from ax.core.data import Data
from ax.utils.common.base import SortableBase
from ax.utils.common.serialization import SerializationMixin
from ax.utils.common.typeutils import checked_cast

if TYPE_CHECKING:  # pragma: no cover
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


class Metric(SortableBase, SerializationMixin):
    """Base class for representing metrics.

    The `fetch_trial_data` method is the essential method to override when
    subclassing, which specifies how to retrieve a Metric, for a given trial.

    A Metric must return a Data object, which requires (at minimum) the following:
        https://ax.dev/api/_modules/ax/core/data.html#Data.required_columns

    Attributes:
        lower_is_better: Flag for metrics which should be minimized.
        properties: Properties specific to a particular metric.
    """

    data_constructor: Type[Data] = Data

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

    @classmethod
    def is_available_while_running(cls) -> bool:
        """Whether metrics of this class are available while the trial is running.
        Metrics that are not available while the trial is running are assumed to be
        available only upon trial completion. For such metrics, data is assumed to
        never change once the trial is completed.

        NOTE: If this method returns `False`, data-fetching via `experiment.fetch_data`
        will return the data cached on the experiment (for the metrics of the given
        class) whenever its available. Data is cached on experiment when attached
        via `experiment.attach_data`.
        """
        return False

    @property
    def name(self) -> str:
        """Get name of metric."""
        return self._name

    @property
    def fetch_multi_group_by_metric(self) -> Type[Metric]:
        """Metric class, with which to group this metric in
        `Experiment._metrics_by_class`, which is used to combine metrics on experiment
        into groups and then fetch their data via `Metric.fetch_trial_data_multi` for
        each group.

        NOTE: By default, this property will just return the class on which it is
        defined; however, in some cases it is useful to group metrics by their
        superclass, in which case this property should return that superclass.
        """
        return self.__class__

    def fetch_trial_data(self, trial: core.base_trial.BaseTrial, **kwargs: Any) -> Data:
        """Fetch data for one trial."""
        raise NotImplementedError(
            f"Metric {self.name} does not implement data-fetching logic."
        )  # pragma: no cover

    def fetch_experiment_data(
        self, experiment: core.experiment.Experiment, **kwargs: Any
    ) -> Data:
        """Fetch this metric's data for an experiment.

        Default behavior is to fetch data from all trials expecting data
        and concatenate the results.
        """
        return self.data_constructor.from_multiple_data(
            [
                checked_cast(
                    self.data_constructor, self.fetch_trial_data(trial, **kwargs)
                )
                if trial.status.expecting_data
                else self.data_constructor()
                for trial in experiment.trials.values()
            ],
        )

    @classmethod
    def fetch_trial_data_multi(
        cls, trial: core.base_trial.BaseTrial, metrics: Iterable[Metric], **kwargs: Any
    ) -> Data:
        """Fetch multiple metrics data for one trial.

        Default behavior calls `fetch_trial_data` for each metric.
        Subclasses should override this to trial data computation for multiple metrics.
        """
        dat = cls.data_constructor.from_multiple_data(
            [metric.fetch_trial_data(trial, **kwargs) for metric in metrics]
        )
        return dat

    @classmethod
    def fetch_experiment_data_multi(
        cls,
        experiment: core.experiment.Experiment,
        metrics: Iterable[Metric],
        trials: Optional[Iterable[core.base_trial.BaseTrial]] = None,
        **kwargs: Any,
    ) -> Data:
        """Fetch multiple metrics data for an experiment.

        Default behavior calls `fetch_trial_data_multi` for each trial.
        Subclasses should override to batch data computation across trials + metrics.
        """
        return cls.data_constructor.from_multiple_data(
            [
                cls.fetch_trial_data_multi(trial, metrics, **kwargs)
                if trial.status.expecting_data
                else cls.data_constructor()
                for trial in (experiment.trials.values() if trials is None else trials)
            ]
        )

    @classmethod
    def lookup_or_fetch_experiment_data_multi(
        cls,
        experiment: core.experiment.Experiment,
        metrics: Iterable[Metric],
        trials: Optional[Iterable[core.base_trial.BaseTrial]] = None,
        **kwargs: Any,
    ) -> Tuple[Data, bool]:
        """Fetch or lookup (with fallback to fetching) data for given metrics,
        depending on whether they are available while running. Return a tuple
        containing the data, along with a boolean that will be True if new
        data was fetched, and False if all data was looked up from cache.

        If metric is available while running, its data can change (and therefore
        we should always re-fetch it). If metric is available only upon trial
        completion, its data does not change, so we can look up that data on
        the experiment and only fetch the data that is not already attached to
        the experiment.

        NOTE: If fetching data for a metrics class that is only available upon
        trial completion, data fetched in this function (data that was not yet
        available on experiment) will be attached to experiment.
        """
        # If this metric is available while trial is running, just default to
        # `fetch_experiment_data_multi`.
        if cls.is_available_while_running():
            fetched_data = cls.fetch_experiment_data_multi(
                experiment=experiment, metrics=metrics, trials=trials, **kwargs
            )
            return fetched_data, True

        # If this metric is available only upon trial completion, look up data
        # on experiment and only fetch data that is not already cached.
        if trials is None:
            completed_trials = experiment.trials_by_status[
                core.base_trial.TrialStatus.COMPLETED
            ]
        else:
            completed_trials = [t for t in trials if t.status.is_completed]

        if not completed_trials:
            return cls.data_constructor(), False

        trials_data = []
        contains_new_data = False
        for trial in completed_trials:
            cached_trial_data = experiment.lookup_data_for_trial(
                trial_index=trial.index,
            )[0]

            cached_metric_names = cached_trial_data.metric_names
            metrics_to_fetch = [m for m in metrics if m.name not in cached_metric_names]
            if not metrics_to_fetch:
                # If all needed data fetched from cache, no need to fetch any other data
                # for trial.
                trials_data.append(cached_trial_data)
                continue

            try:
                fetched_trial_data = cls.fetch_experiment_data_multi(
                    experiment=experiment,
                    metrics=metrics_to_fetch,
                    trials=[trial],
                    **kwargs,
                )
                contains_new_data = True
            except NotImplementedError:
                # Metric does not implement fetching logic and only uses lookup.
                fetched_trial_data = cls.data_constructor()

            final_data = cls.data_constructor.from_multiple_data(
                [cached_trial_data, fetched_trial_data]
            )

            trials_data.append(final_data)
        return (
            cls.data_constructor.from_multiple_data(
                trials_data, subset_metrics=[m.name for m in metrics]
            ),
            contains_new_data,
        )

    def clone(self) -> Metric:
        """Create a copy of this Metric."""
        cls = type(self)
        return cls(
            **cls.deserialize_init_args(args=cls.serialize_init_args(obj=self)),
        )

    def __repr__(self) -> str:
        return "{class_name}('{metric_name}')".format(
            class_name=self.__class__.__name__, metric_name=self.name
        )

    @property
    def _unique_id(self) -> str:
        return str(self)

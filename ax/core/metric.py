#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import traceback

from dataclasses import dataclass
from functools import reduce
from logging import Logger

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
)

from ax.core.data import Data
from ax.utils.common.base import SortableBase
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Err, Ok, Result, UnwrapError
from ax.utils.common.serialization import SerializationMixin

if TYPE_CHECKING:  # pragma: no cover
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


logger: Logger = get_logger(__name__)


@dataclass(frozen=True)
class MetricFetchE:
    # NOTE/TODO[mpolson64]: This could probably be generalized to a
    # `PythonExceptionE` class in the future. Let's do our best to avoid
    # reinventing the wheel in the next `Result` use case in Ax.

    message: str
    exception: Optional[Exception]

    def __post_init__(self) -> None:
        logger.info(msg=f"MetricFetchE INFO: Initialized {self}")

    def __repr__(self) -> str:
        if self.exception is None:
            return f'MetricFetchE(message="{self.message}")'

        return (
            f'MetricFetchE(message="{self.message}", exception={self.exception})\n'
            f"with Traceback:\n {self.tb_str()}"
        )

    def tb_str(self) -> Optional[str]:
        if self.exception is None:
            return None

        return reduce(
            lambda left, right: left + right,
            traceback.format_exception(
                None, self.exception, self.exception.__traceback__
            ),
        )


MetricFetchResult = Result[Data, MetricFetchE]


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
        # pyre-fixme[4]: Attribute must be annotated.
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

    def fetch_trial_data(
        self, trial: core.base_trial.BaseTrial, **kwargs: Any
    ) -> MetricFetchResult:
        """Fetch data for one trial."""
        raise NotImplementedError(
            f"Metric {self.name} does not implement data-fetching logic."
        )  # pragma: no cover

    def fetch_experiment_data(
        self, experiment: core.experiment.Experiment, **kwargs: Any
    ) -> Dict[int, MetricFetchResult]:
        """Fetch this metric's data for an experiment.

        Returns Dict of trial_index => Result
        """

        return {
            trial.index: self.fetch_trial_data(trial=trial, **kwargs)
            for trial in experiment.trials.values()
            if trial.status.expecting_data
        }

    @classmethod
    def fetch_trial_data_multi(
        cls, trial: core.base_trial.BaseTrial, metrics: Iterable[Metric], **kwargs: Any
    ) -> Dict[str, MetricFetchResult]:
        """Fetch multiple metrics data for one trial.

        Returns Dict of metric_name => Result
        Default behavior calls `fetch_trial_data` for each metric.
        Subclasses should override this to trial data computation for multiple metrics.
        """

        return {
            metric.name: metric.fetch_trial_data(trial=trial, **kwargs)
            for metric in metrics
        }

    @classmethod
    def fetch_experiment_data_multi(
        cls,
        experiment: core.experiment.Experiment,
        metrics: Iterable[Metric],
        trials: Optional[Iterable[core.base_trial.BaseTrial]] = None,
        **kwargs: Any,
    ) -> Dict[int, Dict[str, MetricFetchResult]]:
        """Fetch multiple metrics data for an experiment.

        Returns Dict of trial_index => (metric_name => Result)
        Default behavior calls `fetch_trial_data_multi` for each trial.
        Subclasses should override to batch data computation across trials + metrics.
        """

        return {
            trial.index: cls.fetch_trial_data_multi(
                trial=trial, metrics=metrics, **kwargs
            )
            for trial in (trials if trials is not None else experiment.trials.values())
            if trial.status.expecting_data
        }

    @classmethod
    def lookup_or_fetch_experiment_data_multi(
        cls,
        experiment: core.experiment.Experiment,
        metrics: Iterable[Metric],
        trials: Optional[Iterable[core.base_trial.BaseTrial]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[int, Dict[str, MetricFetchResult]], bool]:
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
            return {}, False

        trials_results = {}
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
                trials_results[trial.index] = cls._wrap_trial_data_multi(
                    data=cached_trial_data
                )
                continue

            try:
                fetched_trial_data = cls.fetch_experiment_data_multi(
                    experiment=experiment,
                    metrics=metrics_to_fetch,
                    trials=[trial],
                    **kwargs,
                )[trial.index]
                contains_new_data = True
            except NotImplementedError:
                # Metric does not implement fetching logic and only uses lookup.
                fetched_trial_data = {}

            trials_results[trial.index] = {
                **cls._wrap_trial_data_multi(data=cached_trial_data),
                **fetched_trial_data,
            }

        return (
            {
                trial_index: {
                    metric_name: results
                    for metric_name, results in results_by_metric_name.items()
                    if metric_name in [metric.name for metric in metrics]
                }
                for trial_index, results_by_metric_name in trials_results.items()
            },
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

    @classmethod
    def _unwrap_experiment_data(cls, results: Mapping[int, MetricFetchResult]) -> Data:
        # NOTE: This can be lossy (ex. a MapData could get implicitly cast to a Data and
        # lose rows)if some MetricFetchResults contain Data not of type
        # `cls.data_constructor`

        oks: List[Ok[Data, MetricFetchE]] = [
            result for result in results.values() if isinstance(result, Ok)
        ]
        if len(oks) < len(results):
            errs: List[Err[Data, MetricFetchE]] = [
                result for result in results.values() if isinstance(result, Err)
            ]

            # TODO[mpolson64] Raise all errors in a group via PEP 654
            exceptions = [
                err.err.exception
                if err.err.exception is not None
                else Exception(err.err.message)
                for err in errs
            ]

            raise UnwrapError(errs) from (
                exceptions[0] if len(exceptions) == 1 else Exception(exceptions)
            )

        data = [ok.ok for ok in oks]
        return (
            cls.data_constructor.from_multiple_data(data=data)
            if len(data) > 0
            else cls.data_constructor()
        )

    @classmethod
    def _unwrap_trial_data_multi(
        cls,
        results: Mapping[str, MetricFetchResult],
        # TODO[mpolson64] Add critical_metric_names to other unwrap methods
        critical_metric_names: Optional[Iterable[str]] = None,
    ) -> Data:
        # NOTE: This can be lossy (ex. a MapData could get implicitly cast to a Data and
        # lose rows)if some MetricFetchResults contain Data not of type
        # `cls.data_constructor`

        oks: List[Ok[Data, MetricFetchE]] = [
            result for result in results.values() if isinstance(result, Ok)
        ]
        if len(oks) < len(results):
            # If no critical_metric_names supplied all metrics to be treated as
            # critical
            critical_metric_names = critical_metric_names or results.keys()

            # Noncritical Errs should be brought to the user's attention via warnings
            # but not raise an Exception
            noncritical_errs: List[Err[Data, MetricFetchE]] = [
                result
                for metric_name, result in results.items()
                if isinstance(result, Err) and metric_name in critical_metric_names
            ]

            for err in noncritical_errs:
                logger.warning(
                    f"Err encountered while unwrapping MetricFetchResults: {err.err}. "
                    "Metric is not marked critical, ignoring for now."
                )

            critical_errs: List[Err[Data, MetricFetchE]] = [
                result
                for metric_name, result in results.items()
                if isinstance(result, Err) and metric_name in critical_metric_names
            ]

            if len(critical_errs) > 0:
                # TODO[mpolson64] Raise all errors in a group via PEP 654
                exceptions = [
                    err.err.exception
                    if err.err.exception is not None
                    else Exception(err.err.message)
                    for err in critical_errs
                ]
                raise UnwrapError(critical_errs) from (
                    exceptions[0] if len(exceptions) == 1 else Exception(exceptions)
                )

        data = [ok.ok for ok in oks]

        return (
            cls.data_constructor.from_multiple_data(data=data)
            if len(data) > 0
            else cls.data_constructor()
        )

    @classmethod
    def _unwrap_experiment_data_multi(
        cls,
        results: Mapping[int, Mapping[str, MetricFetchResult]],
    ) -> Data:
        # NOTE: This can be lossy (ex. a MapData could get implicitly cast to a Data and
        # lose rows)if some MetricFetchResults contain Data not of type
        # `cls.data_constructor`

        flattened = [
            result for sublist in results.values() for result in sublist.values()
        ]
        oks: List[Ok[Data, MetricFetchE]] = [
            result for result in flattened if isinstance(result, Ok)
        ]
        if len(oks) < len(flattened):
            errs: List[Err[Data, MetricFetchE]] = [
                result for result in flattened if isinstance(result, Err)
            ]

            # TODO[mpolson64] Raise all errors in a group via PEP 654
            exceptions = [
                err.err.exception
                if err.err.exception is not None
                else Exception(err.err.message)
                for err in errs
            ]
            raise UnwrapError(errs) from (
                exceptions[0] if len(exceptions) == 1 else Exception(exceptions)
            )

        data = [ok.ok for ok in oks]
        return (
            cls.data_constructor.from_multiple_data(data=data)
            if len(data) > 0
            else cls.data_constructor()
        )

    @classmethod
    def _wrap_experiment_data(cls, data: Data) -> Dict[int, MetricFetchResult]:
        return {
            trial_index: Ok(value=data.filter(trial_indices=[trial_index]))
            for trial_index in data.true_df["trial_index"]
        }

    @classmethod
    def _wrap_trial_data_multi(cls, data: Data) -> Dict[str, MetricFetchResult]:
        return {
            metric_name: Ok(value=data.filter(metric_names=[metric_name]))
            for metric_name in data.true_df["metric_name"]
        }

    @classmethod
    def _wrap_experiment_data_multi(
        cls, data: Data
    ) -> Dict[int, Dict[str, MetricFetchResult]]:
        # pyre-fixme[7]
        return {
            trial_index: {
                metric_name: Ok(
                    value=data.filter(
                        trial_indices=[trial_index], metric_names=[metric_name]
                    )
                )
                for metric_name in data.true_df["metric_name"]
            }
            for trial_index in data.true_df["trial_index"]
        }

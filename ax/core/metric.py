#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import traceback
import warnings
from collections.abc import Iterable, Mapping

from dataclasses import dataclass
from datetime import timedelta
from functools import reduce
from logging import Logger

from typing import Any, TYPE_CHECKING

from ax.core.data import Data
from ax.utils.common.base import SortableBase
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Err, Ok, Result, UnwrapError
from ax.utils.common.serialization import SerializationMixin

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


logger: Logger = get_logger(__name__)


@dataclass(frozen=True)
class MetricFetchE:
    # TODO[mpolson64] Replace this with ExceptionE

    message: str
    exception: Exception | None

    def __post_init__(self) -> None:
        logger.info(msg=f"MetricFetchE INFO: Initialized {self}")

    def __repr__(self) -> str:
        if self.exception is None:
            return f'MetricFetchE(message="{self.message}")'

        return (
            f'MetricFetchE(message="{self.message}", exception={self.exception})\n'
            f"with Traceback:\n {self.tb_str()}"
        )

    def tb_str(self) -> str | None:
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

    data_constructor: type[Data] = Data
    # The set of exception types stored in a ``MetchFetchE.exception`` that are
    # recoverable ``Scheduler._fetch_and_process_trials_data_results()``.
    # Exception may be a subclass of any of these types.  If you want your metric
    # to never fail the trial, set this to ``{Exception}`` in your metric subclass.
    recoverable_exceptions: set[type[Exception]] = set()

    def __init__(
        self,
        name: str,
        lower_is_better: bool | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Inits Metric.

        Args:
            name: Name of metric.
            lower_is_better: Flag for metrics which should be minimized.
            properties: Dictionary of this metric's properties
        """
        self._name = name
        self.lower_is_better = lower_is_better
        self.properties: dict[str, Any] = properties or {}

    # ---------- Properties and methods that subclasses often override. ----------

    # NOTE: Override this if your metric can be fetched before the trial is complete,
    # especially if new data is available over time while the trial continues running.
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

    # NOTE: Override this if your metric can fetch new data even after the trial is
    # completed.
    @classmethod
    def period_of_new_data_after_trial_completion(cls) -> timedelta:
        """Period of time metrics of this class are still expecting new data to arrive
        after trial completion.  This is useful for metrics whose results are processed
        by some sort of data pipeline, where the pipeline will continue to land
        additional data even after the trial is completed.

        If the metric is not available after trial completion, this method will
        return `timedelta(0)`. Otherwise, it should return the maximum amount of time
        that the metric may have new data arrive after the trial is completed.

        NOTE: This property will not prevent new data from attempting to be refetched
        for completed trials when calling `experiment.fetch_data()`.  Its purpose is to
        prevent `experiment.fetch_data()` from being called in `Scheduler` and anywhere
        else it is checked.
        """
        return timedelta(0)

    @classmethod
    def is_reconverable_fetch_e(cls, metric_fetch_e: MetricFetchE) -> bool:
        """Checks whether the given MetricFetchE is recoverable for this metric class
        in ``Scheduler._fetch_and_process_trials_data_results``.
        """
        if metric_fetch_e.exception is None:
            return False
        return any(
            isinstance(metric_fetch_e.exception, e) for e in cls.recoverable_exceptions
        )

    # NOTE: This is rarely overridden –– oonly if you want to fetch data in groups
    # consisting of multiple different metric classes, for data to be fetched together.
    # This makes sense only if `fetch_trial data_multi` or `fetch_experiment_data_multi`
    # leverages fetching multiple metrics at once instead of fetching each serially,
    # and that fetching logic is shared across the metric group.
    @property
    def fetch_multi_group_by_metric(self) -> type[Metric]:
        """Metric class, with which to group this metric in
        `Experiment._metrics_by_class`, which is used to combine metrics on experiment
        into groups and then fetch their data via `Metric.fetch_trial_data_multi` for
        each group.

        NOTE: By default, this property will just return the class on which it is
        defined; however, in some cases it is useful to group metrics by their
        superclass, in which case this property should return that superclass.
        """
        return self.__class__

    # NOTE: This is always overridden by subclasses, sometimes along with `bulk_fetch
    # trial_data` and/or `bulk_fetch_experiment_data`. The entrypoint to metric
    # fetching is tupically `fetch_data_prefer_lookup`, which calls `bulk_fetch_
    # experiment_data`, so it can be sufficient to override just that method, then
    # use it to implement this one.
    def fetch_trial_data(
        self, trial: core.base_trial.BaseTrial, **kwargs: Any
    ) -> MetricFetchResult:
        """Fetch data for one trial."""
        raise NotImplementedError(
            f"Metric {self.name} does not implement data-fetching logic."
        )

    # NOTE: Override this if your metric requires custom string representation with
    # more attributes included than just the name.
    def __repr__(self) -> str:
        return "{class_name}('{metric_name}')".format(
            class_name=self.__class__.__name__, metric_name=self.name
        )

    @property
    def summary_dict(self) -> dict[str, Any]:
        """Returns a dictionary containing the metric's name and properties."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "lower_is_better": self.lower_is_better,
        }

    # NOTE: This should be overridden if there is a benefit to fetching multiple
    # metrics that all share the `fetch_multi_group_by_metric` setting, at once.
    # This gives an opportunity to perform a given operation (e.g. retrieve results
    # of some remote job) only once, and then use the result to fetch each metric's
    # value (via `fetch_trial_data` if that is how this method is implemented on the
    # subclass, or in other ways).
    # NOTE: this replaces a now-deprecated classmethod `fetch_trial_data_multi`;
    # in the base implementation, it currently still calls `cls.fetch_experiment_data_
    # multi` to avoid backward incompatibility. A `DeprecationWarning` is raised if
    # this is not overridden but `fetch_trial_data_multi` is.
    def bulk_fetch_trial_data(
        self, trial: core.base_trial.BaseTrial, metrics: list[Metric], **kwargs: Any
    ) -> dict[str, MetricFetchResult]:
        """Fetch multiple metrics data for one trial, using instance attributes
        of the metrics.

        Returns Dict of metric_name => Result
        Default behavior calls `fetch_trial_data` for each metric. Subclasses should
        override this to perform trial data computation for multiple metrics.
        """
        # By default, use the legacy classmethods that served the same function.
        # Overriding the instance methods instead will allow to leverage
        # instance-level configurations of the metric.
        self.maybe_raise_deprecation_warning_on_class_methods()

        return self.__class__.fetch_experiment_data_multi(
            experiment=trial.experiment, trials=[trial], metrics=metrics, **kwargs
        )[trial.index]

    # NOTE: This should be overridden if there is a benefit to fetching multiple
    # metrics that all share the `fetch_multi_group_by_metric` setting, at once,
    # AND ALSO FOR MULTIPLE TRIALS AT ONCE.
    # This gives an opportunity to perform a given operation (e.g. retrieve results
    # of some set of remote jobs, each representing a trial) only once, and then
    # use the result to fetch each metric's value for a set of trials.
    # NOTE: this replaces a now-deprecated classmethod `fetch_experiment_data_multi`;
    # in the base implementation, it calls `bulk_fetch_trial_data`.
    def bulk_fetch_experiment_data(
        self,
        experiment: core.experiment.Experiment,
        metrics: list[Metric],
        trials: list[core.base_trial.BaseTrial] | None = None,
        **kwargs: Any,
    ) -> dict[int, dict[str, MetricFetchResult]]:
        """Fetch multiple metrics data for multiple trials on an experiment, using
        instance attributes of the metrics.

        Returns Dict of metric_name => Result
        Default behavior calls `fetch_trial_data` for each metric.
        Subclasses should override this to trial data computation for multiple metrics.
        """
        trials = list(experiment.trials.values()) if trials is None else trials
        experiment.validate_trials(trials=trials)
        return {
            trial.index: self.bulk_fetch_trial_data(
                trial=trial, metrics=metrics, **kwargs
            )
            for trial in trials
            if trial.status.expecting_data
        }

    # NOTE: Also overridable are `serialize_init_args` and `deserialize_init_args`,
    # which are inherited from the `SerializationMixin` base class.
    # Override those if and only if your metric requires custom serialization; e.g. if
    # some of its attributes are not readily serializable and require pre-processing.
    # Note that all these serialized attributes will be deserialized by the
    # `deserialize_init_args` method on the same class.

    # ---------- Properties and metrods that should not be overridden. ----------

    @property
    def name(self) -> str:
        """Get name of metric."""
        return self._name

    def clone(self) -> Metric:
        """Create a copy of this Metric."""
        cls = type(self)
        return cls(
            **cls.deserialize_init_args(args=cls.serialize_init_args(obj=self)),
        )

    def fetch_data_prefer_lookup(
        self,
        experiment: core.experiment.Experiment,
        metrics: list[Metric],
        trials: list[core.base_trial.BaseTrial] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[int, dict[str, MetricFetchResult]], bool]:
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
        if self.is_available_while_running():
            fetched_data = self.bulk_fetch_experiment_data(
                experiment=experiment, metrics=metrics, trials=trials, **kwargs
            )
            return fetched_data, True

        # If this metric is available only upon trial completion, look up data
        # on experiment and only fetch data that is not already cached.
        completed_trials = (
            experiment.completed_trials
            if trials is None
            else [t for t in trials if t.status.is_completed]
        )

        if not completed_trials:
            return {}, False

        trials_results = {}
        contains_new_data = False

        # TODO: Refactor this to not fetch data for each trial individually.
        # Since we use `bulk_fetch_experiment_data`, we should be able to
        # first identify trial + metric combos to fetch, then fetch them all
        # at once.
        for trial in completed_trials:
            cached_trial_data = experiment.lookup_data_for_trial(
                trial_index=trial.index,
            )[0]

            cached_metric_names = cached_trial_data.metric_names
            metrics_to_fetch = [m for m in metrics if m.name not in cached_metric_names]
            if not metrics_to_fetch:
                # If all needed data fetched from cache, no need to fetch any other data
                # for trial.
                trials_results[trial.index] = self._wrap_trial_data_multi(
                    data=cached_trial_data
                )
                continue

            try:
                fetched_trial_data = self.bulk_fetch_experiment_data(
                    experiment=experiment,
                    metrics=metrics_to_fetch,
                    trials=[trial],
                    **kwargs,
                )[trial.index]

                contains_new_data = any(
                    result.is_ok() for result in fetched_trial_data.values()
                )
            except NotImplementedError:
                # Metric does not implement fetching logic and only uses lookup.
                # TODO: This is only useful for base `Metric` –– all other metrics
                # do implement fetching logic. Should this exist then or is it only
                # adding complexity?
                fetched_trial_data = {}

            trials_results[trial.index] = {
                **self._wrap_trial_data_multi(data=cached_trial_data),
                **fetched_trial_data,
            }

        results = {
            trial_index: {
                metric_name: results
                for metric_name, results in results_by_metric_name.items()
                # We subset the metrics because cached results might have more
                # metrics than requested in arguments passed to this method.
                if metric_name in [metric.name for metric in metrics]
            }
            for trial_index, results_by_metric_name in trials_results.items()
        }
        return results, contains_new_data

    @property
    def _unique_id(self) -> str:
        return str(self)

    # ---------------- Legacy class methods for backward compatibility ----------------
    # ------------------- with previously implemented subclasses  ---------------------

    @classmethod
    def fetch_trial_data_multi(
        cls, trial: core.base_trial.BaseTrial, metrics: Iterable[Metric], **kwargs: Any
    ) -> dict[str, MetricFetchResult]:
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
        trials: Iterable[core.base_trial.BaseTrial] | None = None,
        **kwargs: Any,
    ) -> dict[int, dict[str, MetricFetchResult]]:
        """Fetch multiple metrics data for an experiment.

        Returns Dict of trial_index => (metric_name => Result)
        Default behavior calls `fetch_trial_data_multi` for each trial.
        Subclasses should override to batch data computation across trials + metrics.
        """
        trials = experiment.trials.values() if trials is None else trials
        experiment.validate_trials(trials=trials)
        return {
            trial.index: cls.fetch_trial_data_multi(
                trial=trial, metrics=metrics, **kwargs
            )
            for trial in trials
            if trial.status.expecting_data
        }

    def maybe_raise_deprecation_warning_on_class_methods(self) -> None:
        # This is a temporary hack to allow us to deprecate old metric class method
        # implementations. There does not seem to be another way of checking whether
        # base class' classmethods are overridden in subclasses.
        is_fetch_trial_data_multi_overriden = (
            getattr(self.__class__.fetch_trial_data_multi, "__code__", "DEFAULT")
            != Metric.fetch_trial_data_multi.__code__  # pyre-ignore[16]
        )
        is_fetch_experiment_data_multi_overriden = (
            getattr(
                self.__class__.fetch_experiment_data_multi,
                "__code__",
                "DEFAULT",
            )
            != Metric.fetch_experiment_data_multi.__code__  # pyre-ignore[16]
        )
        # Raise deprecation warning if this method from the base class is used (meaning
        # that it is not overridden and the classmethod is overridden instead), unless
        # the only overridden method is `fetch_trial_data` (in which case the setup is
        # not changing for that metric with the deprecation of the classmethods).
        if (
            is_fetch_trial_data_multi_overriden
            or is_fetch_experiment_data_multi_overriden
        ):
            warnings.warn(  # noqa B028 (level 1 stack trace is ok in this case)
                DeprecationWarning(
                    "Data-fetching class-methods: `fetch_trial_data_multi` and "
                    "`fetch_experiment_data_multi`, will soon be deprecated in Ax. "
                    "please leverage instance-methods like `bulk_fetch_trial_data` "
                    "or `bulk_fetch_experiment_data` instead going forward. "
                    f"Metric {self.name} (class: {self.__class__} in {self.__module__})"
                    " implementation overrides the class methods."
                )
            )

    # -------- Wrapping and unwrapping of data into insulated `Result` objects, --------
    # ----------- which enforce proper handling of these errors downstream. ------------

    @classmethod
    def _unwrap_experiment_data(cls, results: Mapping[int, MetricFetchResult]) -> Data:
        # NOTE: This can be lossy (ex. a MapData could get implicitly cast to a Data and
        # lose rows)if some MetricFetchResults contain Data not of type
        # `cls.data_constructor`

        oks: list[Ok[Data, MetricFetchE]] = [
            result for result in results.values() if isinstance(result, Ok)
        ]
        if len(oks) < len(results):
            errs: list[Err[Data, MetricFetchE]] = [
                result for result in results.values() if isinstance(result, Err)
            ]

            # TODO[mpolson64] Raise all errors in a group via PEP 654
            exceptions = [
                (
                    err.err.exception
                    if err.err.exception is not None
                    else Exception(err.err.message)
                )
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
        critical_metric_names: list[str] | None = None,
    ) -> Data:
        # NOTE: This can be lossy (ex. a MapData could get implicitly cast to a Data and
        # lose rows)if some MetricFetchResults contain Data not of type
        # `cls.data_constructor`

        oks: list[Ok[Data, MetricFetchE]] = [
            result for result in results.values() if isinstance(result, Ok)
        ]
        if len(oks) < len(results):
            # If no critical_metric_names supplied all metrics to be treated as
            # critical
            critical_metric_names = critical_metric_names or list(results.keys())

            # Noncritical Errs should be brought to the user's attention via warnings
            # but not raise an Exception
            noncritical_errs: list[Err[Data, MetricFetchE]] = [
                result
                for metric_name, result in results.items()
                if isinstance(result, Err) and metric_name in critical_metric_names
            ]

            for err in noncritical_errs:
                logger.warning(
                    f"Err encountered while unwrapping MetricFetchResults: {err.err}. "
                    "Metric is not marked critical, ignoring for now."
                )

            critical_errs: list[Err[Data, MetricFetchE]] = [
                result
                for metric_name, result in results.items()
                if isinstance(result, Err) and metric_name in critical_metric_names
            ]

            if len(critical_errs) > 0:
                # TODO[mpolson64] Raise all errors in a group via PEP 654
                exceptions = [
                    (
                        err.err.exception
                        if err.err.exception is not None
                        else Exception(err.err.message)
                    )
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
        oks: list[Ok[Data, MetricFetchE]] = [
            result for result in flattened if isinstance(result, Ok)
        ]
        if len(oks) < len(flattened):
            errs: list[Err[Data, MetricFetchE]] = [
                result for result in flattened if isinstance(result, Err)
            ]

            # TODO[mpolson64] Raise all errors in a group via PEP 654
            exceptions = [
                (
                    err.err.exception
                    if err.err.exception is not None
                    else Exception(err.err.message)
                )
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
    def _wrap_experiment_data(cls, data: Data) -> dict[int, MetricFetchResult]:
        return {
            trial_index: Ok(value=data.filter(trial_indices=[trial_index]))
            for trial_index in data.true_df["trial_index"]
        }

    @classmethod
    def _wrap_trial_data_multi(cls, data: Data) -> dict[str, MetricFetchResult]:
        return {
            metric_name: Ok(value=data.filter(metric_names=[metric_name]))
            for metric_name in data.true_df["metric_name"]
        }

    @classmethod
    def _wrap_experiment_data_multi(
        cls, data: Data
    ) -> dict[int, dict[str, MetricFetchResult]]:
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

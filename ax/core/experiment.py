#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime
from enum import Enum
from functools import reduce
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Type

import pandas as pd
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, DEFAULT_STATUSES_TO_WARM_START, TrialStatus
from ax.core.batch_trial import BatchTrial, LifecycleStage
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import Parameter
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.exceptions.core import UnsupportedError
from ax.utils.common.base import Base
from ax.utils.common.constants import EXPERIMENT_IS_TEST_WARNING, Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Err, Ok
from ax.utils.common.timeutils import current_timestamp_in_millis
from ax.utils.common.typeutils import checked_cast, not_none

logger: logging.Logger = get_logger(__name__)


class DataType(Enum):
    DATA = 1
    MAP_DATA = 3


# pyre-fixme[24]: Generic type `type` expects 1 type parameter, use `typing.Type` to
#  avoid runtime subscripting errors.
DATA_TYPE_LOOKUP: Dict[DataType, Type] = {
    DataType.DATA: Data,
    DataType.MAP_DATA: MapData,
}

DEFAULT_OBJECTIVE_NAME = "objective"


# pyre-fixme[13]: Attribute `_search_space` is never initialized.
class Experiment(Base):
    """Base class for defining an experiment."""

    def __init__(
        self,
        search_space: SearchSpace,
        name: Optional[str] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        tracking_metrics: Optional[List[Metric]] = None,
        runner: Optional[Runner] = None,
        status_quo: Optional[Arm] = None,
        description: Optional[str] = None,
        is_test: bool = False,
        experiment_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        default_data_type: Optional[DataType] = None,
    ) -> None:
        """Inits Experiment.

        Args:
            search_space: Search space of the experiment.
            name: Name of the experiment.
            optimization_config: Optimization config of the experiment.
            tracking_metrics: Additional tracking metrics not used for optimization.
            runner: Default runner used for trials on this experiment.
            status_quo: Arm representing existing "control" arm.
            description: Description of the experiment.
            is_test: Convenience metadata tracker for the user to mark test experiments.
            experiment_type: The class of experiments this one belongs to.
            properties: Dictionary of this experiment's properties.
            default_data_type: Enum representing the data type this experiment uses.
        """
        # appease pyre
        self._search_space: SearchSpace
        self._status_quo: Optional[Arm] = None
        self._is_test: bool

        self._name = name
        self.description = description
        self.runner = runner
        self.is_test = is_test

        self._data_by_trial: Dict[int, OrderedDict[int, Data]] = {}
        self._experiment_type: Optional[str] = experiment_type
        # pyre-fixme[4]: Attribute must be annotated.
        self._optimization_config = None
        self._tracking_metrics: Dict[str, Metric] = {}
        self._time_created: datetime = datetime.now()
        self._trials: Dict[int, BaseTrial] = {}
        self._properties: Dict[str, Any] = properties or {}
        # pyre-fixme[4]: Attribute must be annotated.
        self._default_data_type = default_data_type or DataType.DATA
        # Used to keep track of whether any trials on the experiment
        # specify a TTL. Since trials need to be checked for their TTL's
        # expiration often, having this attribute helps avoid unnecessary
        # TTL checks for experiments that do not use TTL.
        self._trials_have_ttl = False
        # Make sure all statuses appear in this dict, to avoid key errors.
        self._trial_indices_by_status: Dict[TrialStatus, Set[int]] = {
            status: set() for status in TrialStatus
        }
        self._arms_by_signature: Dict[str, Arm] = {}
        self._arms_by_name: Dict[str, Arm] = {}

        self.add_tracking_metrics(tracking_metrics or [])

        # call setters defined below
        self.search_space = search_space
        self.status_quo = status_quo
        if optimization_config is not None:
            self.optimization_config = optimization_config

    @property
    def has_name(self) -> bool:
        """Return true if experiment's name is not None."""
        return self._name is not None

    @property
    def name(self) -> str:
        """Get experiment name. Throws if name is None."""
        if self._name is None:
            raise ValueError("Experiment's name is None.")
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Set experiment name."""
        self._name = name

    @property
    def is_test(self) -> bool:
        """Get whether the experiment is a test."""
        return self._is_test

    @is_test.setter
    def is_test(self, is_test: bool) -> None:
        """Set whether the experiment is a test."""
        if is_test:
            logger.info(EXPERIMENT_IS_TEST_WARNING)
        self._is_test = is_test

    @property
    def time_created(self) -> datetime:
        """Creation time of the experiment."""
        return self._time_created

    @property
    def experiment_type(self) -> Optional[str]:
        """The type of the experiment."""
        return self._experiment_type

    @experiment_type.setter
    def experiment_type(self, experiment_type: Optional[str]) -> None:
        """Set the type of the experiment."""
        self._experiment_type = experiment_type

    @property
    def search_space(self) -> SearchSpace:
        """The search space for this experiment.

        When setting a new search space, all parameter names and types
        must be preserved. However, if no trials have been created, all
        modifications are allowed.
        """
        # TODO: maybe return a copy here to guard against implicit changes
        return self._search_space

    @search_space.setter
    def search_space(self, search_space: SearchSpace) -> None:
        # Allow all modifications when no trials present.
        if not hasattr(self, "_search_space") or len(self.trials) < 1:
            self._search_space = search_space
            return

        # At least 1 trial is present.
        if self.immutable_search_space_and_opt_config:
            raise UnsupportedError(
                "Modifications of search space are disabled by the "
                f"`{Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF.value}` "
                "property that is set to `True` on this experiment."
            )
        if len(search_space.parameters) < len(self._search_space.parameters):
            raise ValueError(
                "New search_space must contain all parameters in the existing."
            )
        for param_name, parameter in search_space.parameters.items():
            if param_name not in self._search_space.parameters:
                raise ValueError(
                    f"Cannot add new parameter `{param_name}` because "
                    "it is not defined in the existing search space."
                )
            elif (
                parameter.parameter_type
                != self._search_space.parameters[param_name].parameter_type
            ):
                raise ValueError(
                    f"Expected parameter `{param_name}` to be of type "
                    f"{self._search_space.parameters[param_name].parameter_type}, "
                    f"got {parameter.parameter_type}."
                )
        self._search_space = search_space

    @property
    def status_quo(self) -> Optional[Arm]:
        """The existing arm that new arms will be compared against."""
        return self._status_quo

    @status_quo.setter
    def status_quo(self, status_quo: Optional[Arm]) -> None:
        if status_quo is not None:
            self.search_space.check_types(
                parameterization=status_quo.parameters, raise_error=True
            )
            self.search_space.check_all_parameters_present(
                parameterization=status_quo.parameters, raise_error=True
            )

            # Compute a unique name if "status_quo" is taken
            name = "status_quo"
            sq_idx = 0
            arms_by_name = self.arms_by_name
            while name in arms_by_name:
                name = f"status_quo_e{sq_idx}"
                sq_idx += 1

            self._name_and_store_arm_if_not_exists(arm=status_quo, proposed_name=name)

        # If old status_quo not present in any trials,
        # remove from _arms_by_signature
        if self._status_quo is not None:
            persist_old_sq = False
            for trial in self._trials.values():
                # pyre-fixme[16]: `Optional` has no attribute `name`.
                if self._status_quo.name in trial.arms_by_name:
                    persist_old_sq = True
                    break
            if not persist_old_sq:
                # pyre-fixme[16]: `Optional` has no attribute `signature`.
                self._arms_by_signature.pop(self._status_quo.signature)
                self._arms_by_name.pop(self._status_quo.name)

        self._status_quo = status_quo

    @property
    def parameters(self) -> Dict[str, Parameter]:
        """The parameters in the experiment's search space."""
        return self.search_space.parameters

    @property
    def arms_by_name(self) -> Dict[str, Arm]:
        """The arms belonging to this experiment, by their name."""
        return self._arms_by_name

    @property
    def arms_by_signature(self) -> Dict[str, Arm]:
        """The arms belonging to this experiment, by their signature."""
        return self._arms_by_signature

    @property
    def sum_trial_sizes(self) -> int:
        """Sum of numbers of arms attached to each trial in this experiment."""
        return reduce(lambda a, b: a + len(b.arms_by_name), self._trials.values(), 0)

    @property
    def num_abandoned_arms(self) -> int:
        """How many arms attached to this experiment are abandoned."""
        abandoned = set()
        for trial in self.trials.values():
            for x in trial.abandoned_arms:
                abandoned.add(x)
        return len(abandoned)

    @property
    def optimization_config(self) -> Optional[OptimizationConfig]:
        """The experiment's optimization config."""
        return self._optimization_config

    @property
    def is_moo_problem(self) -> bool:
        """Whether the experiment's optimization config contains multiple objectives."""
        if self.optimization_config is None:
            return False
        return not_none(self.optimization_config).is_moo_problem

    @optimization_config.setter
    def optimization_config(self, optimization_config: OptimizationConfig) -> None:
        if (
            len(self.trials) > 0
            and getattr(self, "_optimization_config", None) is not None
            and self.immutable_search_space_and_opt_config
        ):
            raise UnsupportedError(
                "Modifications of optimization config are disabled by the "
                f"`{Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF.value}` "
                "property that is set to `True` on this experiment."
            )
        for metric_name in optimization_config.metrics.keys():
            if metric_name in self._tracking_metrics:
                self.remove_tracking_metric(metric_name)
        self._optimization_config = optimization_config

        if any(
            isinstance(metric, MapMetric)
            for metric in optimization_config.metrics.values()
        ):
            self._default_data_type = DataType.MAP_DATA

    @property
    def data_by_trial(self) -> Dict[int, OrderedDict[int, Data]]:
        """Data stored on the experiment, indexed by trial index and storage time.

        First key is trial index and second key is storage time in milliseconds.
        For a given trial, data is ordered by storage time, so first added data
        will appear first in the list.
        """
        return self._data_by_trial

    @property
    def immutable_search_space_and_opt_config(self) -> bool:
        """Boolean representing whether search space and metrics on this experiment
        are mutable (by default they are).

        NOTE: For experiments with immutable search spaces and metrics, generator
        runs will not store copies of search space and metrics, which improves
        storage layer performance. Not keeping copies of those on generator runs
        also disables keeping track of changes to search space and metrics,
        thereby necessitating that those attributes be immutable on experiment.
        """
        return self._properties.get(Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF, False)

    @property
    def tracking_metrics(self) -> List[Metric]:
        return list(self._tracking_metrics.values())

    def add_tracking_metric(self, metric: Metric) -> Experiment:
        """Add a new metric to the experiment.

        Args:
            metric: Metric to be added.
        """
        if metric.name in self._tracking_metrics:
            raise ValueError(
                f"Metric `{metric.name}` already defined on experiment. "
                "Use `update_tracking_metric` to update an existing metric definition."
            )

        optimization_config = self.optimization_config
        if optimization_config and metric.name in optimization_config.metrics:
            raise ValueError(
                f"Metric `{metric.name}` already present in experiment's "
                "OptimizationConfig. Set a new OptimizationConfig without this metric "
                "before adding it to tracking metrics."
            )

        if isinstance(metric, MapMetric):
            self._default_data_type = DataType.MAP_DATA

        self._tracking_metrics[metric.name] = metric
        return self

    def add_tracking_metrics(self, metrics: List[Metric]) -> Experiment:
        """Add a list of new metrics to the experiment.

        If any of the metrics are already defined on the experiment,
        we raise an error and don't add any of them to the experiment

        Args:
            metrics: Metrics to be added.
        """
        # Before setting any metrics, we validate none are already on
        # the experiment
        for metric in metrics:
            self.add_tracking_metric(metric)
        return self

    def update_tracking_metric(self, metric: Metric) -> Experiment:
        """Redefine a metric that already exists on the experiment.

        Args:
            metric: New metric definition.
        """
        if metric.name not in self._tracking_metrics:
            raise ValueError(f"Metric `{metric.name}` doesn't exist on experiment.")

        self._tracking_metrics[metric.name] = metric
        return self

    def remove_tracking_metric(self, metric_name: str) -> Experiment:
        """Remove a metric that already exists on the experiment.

        Args:
            metric_name: Unique name of metric to remove.
        """
        if metric_name not in self._tracking_metrics:
            raise ValueError(f"Metric `{metric_name}` doesn't exist on experiment.")

        del self._tracking_metrics[metric_name]
        return self

    @property
    def metrics(self) -> Dict[str, Metric]:
        """The metrics attached to the experiment."""
        optimization_config_metrics: Dict[str, Metric] = {}
        if self.optimization_config is not None:
            optimization_config_metrics = self.optimization_config.metrics
        return {**self._tracking_metrics, **optimization_config_metrics}

    def _metrics_by_class(
        self, metrics: Optional[List[Metric]] = None
    ) -> Dict[Type[Metric], List[Metric]]:
        metrics_by_class: Dict[Type[Metric], List[Metric]] = defaultdict(list)
        for metric in metrics or list(self.metrics.values()):
            # By default, all metrics are grouped by their class for fetch;
            # however, for some metrics, `fetch_trial_data_multi` of a
            # superclass is used for fetch the subclassing metrics' data. In
            # those cases, "fetch_multi_group_by_metric" property on metric
            # will be set to a class other than its own (likely a superclass).
            metrics_by_class[metric.fetch_multi_group_by_metric].append(metric)
        return metrics_by_class

    def fetch_data_results(
        self,
        metrics: Optional[List[Metric]] = None,
        combine_with_last_data: bool = False,
        overwrite_existing_data: bool = False,
        **kwargs: Any,
    ) -> Dict[int, Dict[str, MetricFetchResult]]:
        """Fetches data for all trials on this experiment and for either the
        specified metrics or all metrics currently on the experiment, if `metrics`
        argument is not specified.

        If a metric fetch fails, the Exception will be captured in the
        MetricFetchResult along with a message.

        NOTE: For metrics that are not available while trial is running, the data
        may be retrieved from cache on the experiment. Data is cached on the experiment
        via calls to `experiment.attach_data` and whether a given metric class is
        available while trial is running is determined by the boolean returned from its
        `is_available_while_running` class method.

        Args:
            metrics: If provided, fetch data for these metrics instead of the ones
                defined on the experiment.
            kwargs: keyword args to pass to underlying metrics' fetch data functions.

        Returns:
            A nested Dictionary from trial_index => metric_name => result
        """

        return self._lookup_or_fetch_trials_results(
            trials=list(self.trials.values()),
            metrics=metrics,
            combine_with_last_data=combine_with_last_data,
            overwrite_existing_data=overwrite_existing_data,
            **kwargs,
        )

    def fetch_trials_data_results(
        self,
        trial_indices: Iterable[int],
        metrics: Optional[List[Metric]] = None,
        combine_with_last_data: bool = False,
        overwrite_existing_data: bool = False,
        **kwargs: Any,
    ) -> Dict[int, Dict[str, MetricFetchResult]]:
        """Fetches data for specific trials on the experiment.

        If a metric fetch fails, the Exception will be captured in the
        MetricFetchResult along with a message.

        NOTE: For metrics that are not available while trial is running, the data
        may be retrieved from cache on the experiment. Data is cached on the experiment
        via calls to `experiment.attach_data` and whether a given metric class is
        available while trial is running is determined by the boolean returned from its
        `is_available_while_running` class method.

        Args:
            trial_indices: Indices of trials, for which to fetch data.
            metrics: If provided, fetch data for these metrics instead of the ones
                defined on the experiment.
            kwargs: keyword args to pass to underlying metrics' fetch data functions.

        Returns:
            A nested Dictionary from trial_index => metric_name => result
        """
        return self._lookup_or_fetch_trials_results(
            trials=self.get_trials_by_indices(trial_indices=trial_indices),
            metrics=metrics,
            combine_with_last_data=combine_with_last_data,
            overwrite_existing_data=overwrite_existing_data,
            **kwargs,
        )

    def fetch_data(
        self,
        metrics: Optional[List[Metric]] = None,
        combine_with_last_data: bool = False,
        overwrite_existing_data: bool = False,
        **kwargs: Any,
    ) -> Data:
        """Fetches data for all trials on this experiment and for either the
        specified metrics or all metrics currently on the experiment, if `metrics`
        argument is not specified.

        NOTE: For metrics that are not available while trial is running, the data
        may be retrieved from cache on the experiment. Data is cached on the experiment
        via calls to `experiment.attach_data` and whether a given metric class is
        available while trial is running is determined by the boolean returned from its
        `is_available_while_running` class method.

        NOTE: This can be lossy (ex. a MapData could get implicitly cast to a Data and
        lose rows) if Experiment.default_data_type is misconfigured!

        Args:
            metrics: If provided, fetch data for these metrics instead of the ones
                defined on the experiment.
            kwargs: keyword args to pass to underlying metrics' fetch data functions.

        Returns:
            Data for the experiment.
        """

        results = self._lookup_or_fetch_trials_results(
            trials=list(self.trials.values()),
            metrics=metrics,
            combine_with_last_data=combine_with_last_data,
            overwrite_existing_data=overwrite_existing_data,
            **kwargs,
        )

        base_metric_cls = (
            MapMetric if self.default_data_constructor == MapData else Metric
        )

        return base_metric_cls._unwrap_experiment_data_multi(
            results=results,
        )

    def fetch_trials_data(
        self,
        trial_indices: Iterable[int],
        metrics: Optional[List[Metric]] = None,
        combine_with_last_data: bool = False,
        overwrite_existing_data: bool = False,
        **kwargs: Any,
    ) -> Data:
        """Fetches data for specific trials on the experiment.

        NOTE: For metrics that are not available while trial is running, the data
        may be retrieved from cache on the experiment. Data is cached on the experiment
        via calls to `experiment.attach_data` and whetner a given metric class is
        available while trial is running is determined by the boolean returned from its
        `is_available_while_running` class method.

        NOTE: This can be lossy (ex. a MapData could get implicitly cast to a Data and
        lose rows) if Experiment.default_data_type is misconfigured!

        Args:
            trial_indices: Indices of trials, for which to fetch data.
            metrics: If provided, fetch data for these metrics instead of the ones
                defined on the experiment.
            kwargs: Keyword args to pass to underlying metrics' fetch data functions.

        Returns:
            Data for the specific trials on the experiment.
        """

        results = self._lookup_or_fetch_trials_results(
            trials=self.get_trials_by_indices(trial_indices=trial_indices),
            metrics=metrics,
            combine_with_last_data=combine_with_last_data,
            overwrite_existing_data=overwrite_existing_data,
            **kwargs,
        )

        base_metric_cls = (
            MapMetric if self.default_data_constructor == MapData else Metric
        )
        return base_metric_cls._unwrap_experiment_data_multi(
            results=results,
        )

    def _lookup_or_fetch_trials_results(
        self,
        trials: List[BaseTrial],
        metrics: Optional[Iterable[Metric]] = None,
        combine_with_last_data: bool = False,
        overwrite_existing_data: bool = False,
        **kwargs: Any,
    ) -> Dict[int, Dict[str, MetricFetchResult]]:
        if not self.metrics and not metrics:
            raise ValueError(
                "No metrics to fetch data for, as no metrics are defined for "
                "this experiment, and none were passed in to `fetch_data`."
            )
        if not any(t.status.expecting_data for t in trials):
            logger.info("No trials are in a state expecting data.")
            return {}
        metrics_to_fetch = list(metrics or self.metrics.values())
        metrics_by_class = self._metrics_by_class(metrics=metrics_to_fetch)

        results: Dict[int, Dict[str, MetricFetchResult]] = {}
        contains_new_data = False

        for metric_cls in metrics_by_class:
            (
                new_fetch_results,
                new_results_contains_new_data,
            ) = metric_cls.lookup_or_fetch_experiment_data_multi(
                experiment=self,
                metrics=metrics_by_class[metric_cls],
                trials=trials,
                **kwargs,
            )
            contains_new_data = contains_new_data or new_results_contains_new_data

            # Merge in results
            results = {
                trial.index: {
                    **(
                        new_fetch_results[trial.index]
                        if trial.index in new_fetch_results
                        else {}
                    ),
                    **(results[trial.index] if trial.index in results else {}),
                }
                for trial in trials
            }

        if contains_new_data:
            self.attach_fetch_results(
                results=results,
                combine_with_last_data=combine_with_last_data,
                overwrite_existing_data=overwrite_existing_data,
            )

        return results

    @copy_doc(BaseTrial.fetch_data)
    def _fetch_trial_data(
        self, trial_index: int, metrics: Optional[List[Metric]] = None, **kwargs: Any
    ) -> Dict[str, MetricFetchResult]:
        trial = self.trials[trial_index]

        trial_data = self._lookup_or_fetch_trials_results(
            trials=[trial], metrics=metrics, **kwargs
        )

        if trial_index in trial_data:
            return trial_data[trial_index]

        return {}

    def attach_data(
        self,
        data: Data,
        combine_with_last_data: bool = False,
        overwrite_existing_data: bool = False,
    ) -> int:
        """Attach data to experiment. Stores data in `experiment._data_by_trial`,
        to be looked up via `experiment.lookup_data_for_trial`.

        Args:
            data: Data object to store.
            combine_with_last_data: By default, when attaching data, it's identified
                by its timestamp, and `experiment.lookup_data_for_trial` returns
                data by most recent timestamp. Sometimes, however, we want to combine
                the data from multiple calls to `attach_data` into one dataframe.
                This might be because:
                    - We attached data for some metrics at one point and data for
                    the rest of the metrics later on.
                    - We attached data for some fidelity at one point and data for
                    another fidelity later one.
                To achieve that goal, set `combine_with_last_data` to `True`.
                In this case, we will take the most recent previously attached
                data, append the newly attached data to it, attach a new
                Data object with the merged result, and delete the old one.
                Afterwards, calls to `lookup_data_for_trial` will return this
                new combined data object. This operation will also validate that the
                newly added data does not contain observations for metrics that
                already have observations at the same fidelity in the most recent data.
            overwrite_existing_data: By default, we keep around all data that has
                ever been attached to the experiment. However, if we know that
                the incoming data contains all the information we need for a given
                trial, we can replace the existing data for that trial, thereby
                reducing the amount we need to store in the database.

        Returns:
            Timestamp of storage in millis.
        """
        if combine_with_last_data and overwrite_existing_data:
            raise UnsupportedError(
                "Cannot set both combine_with_last_data=True and "
                "overwrite_existing_data=True. Data can either be "
                "combined, or overwritten, or neither."
            )
        data_type = type(data)
        data_init_args = data.deserialize_init_args(data.serialize_init_args(data))
        if data.df.empty:
            raise ValueError("Data to attach is empty.")
        metrics_not_on_exp = set(data.true_df["metric_name"].values) - set(
            self.metrics.keys()
        )
        if metrics_not_on_exp:
            logger.info(
                f"Attached data has some metrics ({metrics_not_on_exp}) that are "
                "not among the metrics on this experiment. Note that attaching data "
                "will not automatically add those metrics to the experiment. "
                "For these metrics to be automatically fetched by `experiment."
                "fetch_data`, add them via `experiment.add_tracking_metric` or update "
                "the experiment's optimization config."
            )
        cur_time_millis = current_timestamp_in_millis()
        for trial_index, trial_df in data.true_df.groupby(data.true_df["trial_index"]):
            # Overwrite `df` so that `data` only has current trial data.
            data_init_args["df"] = trial_df
            current_trial_data = (
                self._data_by_trial[trial_index]
                if trial_index in self._data_by_trial
                else OrderedDict()
            )
            if combine_with_last_data and len(current_trial_data) > 0:
                last_ts, last_data = list(current_trial_data.items())[-1]
                last_data_type = type(last_data)
                merge_keys = ["trial_index", "metric_name", "arm_name"] + (
                    last_data.map_keys if issubclass(last_data_type, MapData) else []
                )
                merged = pd.merge(
                    last_data.true_df,
                    trial_df,
                    on=merge_keys,
                    how="inner",
                )
                if not merged.empty:
                    raise ValueError(
                        f"Last data for trial {trial_index} already contained an "
                        f"observation for metric {merged.head()['metric_name']}."
                    )
                del current_trial_data[last_ts]
                current_trial_data[cur_time_millis] = last_data_type.from_multiple_data(
                    [
                        last_data,
                        last_data_type(**data_init_args),
                    ]
                )
            elif overwrite_existing_data:
                if len(current_trial_data) > 0:
                    _, last_data = list(current_trial_data.items())[-1]
                    last_data_metrics = set(last_data.df["metric_name"])
                    new_data_metrics = set(trial_df["metric_name"])
                    if last_data_metrics.difference(new_data_metrics):
                        raise ValueError(
                            "overwrite_trial_data is True, but the new data contains "
                            "only a subset of the metrics that are present in the "
                            "previous data."
                        )
                current_trial_data = OrderedDict(
                    {cur_time_millis: data_type(**data_init_args)}
                )
            else:
                current_trial_data[cur_time_millis] = data_type(**data_init_args)
            self._data_by_trial[trial_index] = current_trial_data

        return cur_time_millis

    def attach_fetch_results(
        self,
        results: Mapping[int, Mapping[str, MetricFetchResult]],
        combine_with_last_data: bool = False,
        overwrite_existing_data: bool = False,
    ) -> Optional[int]:
        """
        UNSAFE: Prefer to use attach_data directly instead.

        Attach fetched data results to the Experiment so they will not have to be
        fetched again. Returns the timestamp from attachment, which is used as a
        dict key for _data_by_trial.

        NOTE: Any Errs in the results passed in will silently be dropped! This will
        cause the Experiment to fail to find them in the _data_by_trial cache and
        attempt to refetch at fetch time. If this is not your intended behavior you
        MUST resolve your results first and use attach_data directly instead.
        """

        flattened = [
            result for sublist in results.values() for result in sublist.values()
        ]

        oks: List[Ok[Data, MetricFetchE]] = [
            result for result in flattened if isinstance(result, Ok)
        ]

        for result in flattened:
            if isinstance(result, Err):
                logger.error(
                    "Discovered Metric fetching Err while attaching data "
                    f"{result.err}. "
                    "Ignoring for now -- will retry query on next call to fetch."
                )

        data = self.default_data_constructor.from_multiple_data(
            data=[ok.ok for ok in oks]
        )

        if data.df.empty:
            return None

        return self.attach_data(
            data=data,
            combine_with_last_data=combine_with_last_data,
            overwrite_existing_data=overwrite_existing_data,
        )

    def lookup_data_for_ts(self, timestamp: int) -> Data:
        """Collect data for all trials stored at this timestamp.

        Useful when many trials' data was fetched and stored simultaneously
        and user wants to retrieve same collection of data later.

        Can also be used to lookup specific data for a single trial
        when storage time is known.

        Args:
            timestamp: Timestamp in millis at which data was stored.

        Returns:
            Data object with all data stored at the timestamp.
        """
        trial_datas = []
        for _trial_index, ts_to_data in self._data_by_trial.items():
            if timestamp in ts_to_data:
                trial_datas.append(ts_to_data[timestamp])

        return self.default_data_constructor.from_multiple_data(trial_datas)

    def lookup_data_for_trial(
        self,
        trial_index: int,
    ) -> Tuple[Data, int]:
        """Lookup stored data for a specific trial.

        Returns latest data object, and its storage timestamp, present for this trial.
        Returns empty data and -1 if no data present.

        Args:
            trial_index: The index of the trial to lookup data for.
        Returns:
            The requested data object, and its storage timestamp in milliseconds.
        """
        try:
            trial_data_dict = self._data_by_trial[trial_index]
        except KeyError:
            return (self.default_data_constructor(), -1)

        if len(trial_data_dict) == 0:
            return (self.default_data_constructor(), -1)

        storage_time = max(trial_data_dict.keys())
        trial_data = trial_data_dict[storage_time]
        return trial_data, storage_time

    def lookup_data(
        self,
        trial_indices: Optional[Iterable[int]] = None,
    ) -> Data:
        """Lookup data for all trials on this experiment and for either the
        specified metrics or all metrics currently on the experiment, if `metrics`
        argument is not specified.

        Args:
            trial_indices: Indices of trials, for which to fetch data.
        Returns:
            Data for the experiment.
        """
        data_by_trial = []
        trial_indices = trial_indices or list(self.trials.keys())
        for trial_index in trial_indices:
            data_by_trial.append(
                self.lookup_data_for_trial(
                    trial_index=trial_index,
                )[0]
            )
        if not data_by_trial:
            return self.default_data_constructor()
        last_data = data_by_trial[-1]
        last_data_type = type(last_data)
        data = last_data_type.from_multiple_data(data_by_trial)
        return data

    @property
    def num_trials(self) -> int:
        """How many trials are associated with this experiment."""
        return len(self._trials)

    @property
    def trials(self) -> Dict[int, BaseTrial]:
        """The trials associated with the experiment.

        NOTE: If some trials on this experiment specify their TTL, `RUNNING` trials
        will be checked for whether their TTL elapsed during this call. Found past-
        TTL trials will be marked as `FAILED`.
        """
        self._check_TTL_on_running_trials()
        return self._trials

    @property
    def trials_by_status(self) -> Dict[TrialStatus, List[BaseTrial]]:
        """Trials associated with the experiment, grouped by trial status."""
        # Make sure all statuses appear in this dict, to avoid key errors.
        return {
            status: self.get_trials_by_indices(trial_indices=idcs)
            for status, idcs in self.trial_indices_by_status.items()
        }

    @property
    def trials_expecting_data(self) -> List[BaseTrial]:
        """List[BaseTrial]: the list of all trials for which data has arrived
        or is expected to arrive.
        """
        return [trial for trial in self.trials.values() if trial.status.expecting_data]

    @property
    def trial_indices_by_status(self) -> Dict[TrialStatus, Set[int]]:
        """Indices of trials associated with the experiment, grouped by trial
        status.
        """
        self._check_TTL_on_running_trials()  # Marks past-TTL trials as failed.
        return self._trial_indices_by_status

    @property
    def running_trial_indices(self) -> Set[int]:
        """Indices of running trials, associated with the experiment."""
        return self._trial_indices_by_status[TrialStatus.RUNNING]

    @property
    def default_data_type(self) -> DataType:
        return self._default_data_type

    @property
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    def default_data_constructor(self) -> Type:
        return DATA_TYPE_LOOKUP[self.default_data_type]

    def new_trial(
        self,
        generator_run: Optional[GeneratorRun] = None,
        trial_type: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Trial:
        """Create a new trial associated with this experiment.

        Args:
            generator_run: GeneratorRun, associated with this trial.
                Trial has only one arm attached to it and this generator_run
                must therefore contain one arm. This arm can also be set later
                through `add_arm` or `add_generator_run`, but a trial's
                associated generator run is immutable once set.
            trial_type: Type of this trial, if used in MultiTypeExperiment.
            ttl_seconds: If specified, trials will be considered failed after
                this many seconds since the time the trial was ran, unless the
                trial is completed before then. Meant to be used to detect
                'dead' trials, for which the evaluation process might have
                crashed etc., and which should be considered failed after
                their 'time to live' has passed.
        """
        if ttl_seconds is not None:
            self._trials_have_ttl = True
        return Trial(
            experiment=self,
            trial_type=trial_type,
            generator_run=generator_run,
            ttl_seconds=ttl_seconds,
        )

    def new_batch_trial(
        self,
        generator_run: Optional[GeneratorRun] = None,
        trial_type: Optional[str] = None,
        optimize_for_power: Optional[bool] = False,
        ttl_seconds: Optional[int] = None,
        lifecycle_stage: Optional[LifecycleStage] = None,
    ) -> BatchTrial:
        """Create a new batch trial associated with this experiment.

        Args:
            generator_run: GeneratorRun, associated with this trial. This can a
                also be set later through `add_arm` or `add_generator_run`, but a
                trial's associated generator run is immutable once set.
            trial_type: Type of this trial, if used in MultiTypeExperiment.
            optimize_for_power: Whether to optimize the weights of arms in this
                trial such that the experiment's power to detect effects of
                certain size is as high as possible. Refer to documentation of
                `BatchTrial.set_status_quo_and_optimize_power` for more detail.
            ttl_seconds: If specified, trials will be considered failed after
                this many seconds since the time the trial was ran, unless the
                trial is completed before then. Meant to be used to detect
                'dead' trials, for which the evaluation process might have
                crashed etc., and which should be considered failed after
                their 'time to live' has passed.
            lifecycle_stage: The stage of the experiment lifecycle that this
                trial represents
        """
        if ttl_seconds is not None:
            self._trials_have_ttl = True
        return BatchTrial(
            experiment=self,
            trial_type=trial_type,
            generator_run=generator_run,
            optimize_for_power=optimize_for_power,
            ttl_seconds=ttl_seconds,
            lifecycle_stage=lifecycle_stage,
        )

    def get_trials_by_indices(self, trial_indices: Iterable[int]) -> List[BaseTrial]:
        """Grabs trials on this experiment by their indices."""
        trial_indices = list(trial_indices)
        try:
            return [self.trials[idx] for idx in trial_indices]
        except KeyError:
            missing = set(trial_indices) - set(self.trials)
            raise ValueError(
                f"Trial indices {missing} are not associated with the experiment."
            )

    def reset_runners(self, runner: Runner) -> None:
        """Replace all candidate trials runners.

        Args:
            runner: New runner to replace with.
        """
        for trial in self._trials.values():
            if trial.status == TrialStatus.CANDIDATE:
                trial.runner = runner
        self.runner = runner

    def _attach_trial(self, trial: BaseTrial, index: Optional[int] = None) -> int:
        """Attach a trial to this experiment.

        Should only be called within the trial constructor.

        Args:
            trial: The trial to be attached.
            index: If specified, the trial's index will be set accordingly.
                This should generally not be specified, as the index
                will be automatically determined based on the number
                of existing trials. This is only used for the purpose
                of loading from storage.

        Returns:
            The index of the trial within the experiment's trial list.
        """

        if trial.experiment is not self:
            raise ValueError("BatchTrial does not belong to this experiment.")

        for existing_trial in self._trials.values():
            if existing_trial is trial:
                raise ValueError("BatchTrial already attached to experiment.")

        if index is not None and index in self._trials:
            logger.debug(  # pragma: no cover
                f"Trial index {index} already exists on the experiment. Overwriting."
            )
        index = (
            index
            if index is not None
            else (0 if len(self._trials) == 0 else max(self._trials.keys()) + 1)
        )
        self._trials[index] = trial
        return index

    def warm_start_from_old_experiment(
        self,
        old_experiment: Experiment,
        copy_run_metadata_keys: Optional[List[str]] = None,
        trial_statuses_to_copy: Optional[List[TrialStatus]] = None,
        search_space_check_membership_raise_error: bool = True,
    ) -> List[Trial]:
        """Copy all completed trials with data from an old Ax expeirment to this one.
        This function checks that the parameters of each trial are members of the
        current experiment's search_space.

        NOTE: Currently only handles experiments with 1-arm ``Trial``-s, not
        ``BatchTrial``-s as there has not yet been need for support of the latter.

        Args:
            old_experiment: The experiment from which to transfer trials and data
            copy_run_metadata_keys: A list of keys denoting which items to copy over
                from each trial's run_metadata.
            trial_statuses_to_copy: All trials with a status in this list will be
                copied. By default, copies all ``RUNNING``, ``COMPLETED``,
                ``ABANDONED``, and ``EARLY_STOPPED`` trials.
            search_space_check_membership_raise_error: Whether to raise an exception
                if the warm started trials being imported fall outside of the
                defined search space.

        Returns:
            List of trials successfully copied from old_experiment to this one
        """
        if len(self.trials) > 0:
            raise ValueError(  # pragma: no cover
                f"Can only warm-start experiments that don't yet have trials. "
                f"Experiment {self._name} has {len(self.trials)} trials."
            )

        old_parameter_names = set(old_experiment.search_space.parameters.keys())
        parameter_names = set(self.search_space.parameters.keys())
        if old_parameter_names.symmetric_difference(parameter_names):
            raise ValueError(  # pragma: no cover
                f"Cannot warm-start experiment '{self._name}' from experiment "
                f"'{old_experiment._name}' due to mismatch in search space parameters."
                f"Parameters in '{self._name}' but not in '{old_experiment._name}': "
                f"{old_parameter_names - parameter_names}. Vice-versa: "
                f"{parameter_names - old_parameter_names}."
            )

        trial_statuses_to_copy = (
            trial_statuses_to_copy
            if trial_statuses_to_copy is not None
            else DEFAULT_STATUSES_TO_WARM_START
        )

        warm_start_trials = [
            trial
            for trial in old_experiment.trials.values()
            if trial.status in trial_statuses_to_copy
        ]
        copied_trials = []
        for trial in warm_start_trials:
            if not isinstance(trial, Trial):
                raise NotImplementedError(  # pragma: no cover
                    "Only experiments with 1-arm trials currently supported."
                )
            self.search_space.check_membership(
                not_none(trial.arm).parameters,
                raise_error=search_space_check_membership_raise_error,
            )
            dat, ts = old_experiment.lookup_data_for_trial(trial_index=trial.index)
            # Set trial index and arm name to their values in new trial.
            new_trial = self.new_trial()
            add_arm_and_prevent_naming_collision(
                new_trial=new_trial,
                old_trial=trial,
                old_experiment_name=old_experiment._name,
            )
            new_trial.mark_running(no_runner_required=True)
            new_trial._properties["source"] = (
                f"Warm start from Experiment: `{old_experiment._name}`, "
                f"trial: `{trial.index}`"
            )
            # Associates a generation_model_key to the new trial.
            generation_model_key = trial._properties.get("generation_model_key")
            if generation_model_key is None and trial.generator_run is not None:
                generation_model_key = trial.generator_run._model_key or "Manual"
            new_trial._properties["generation_model_key"] = generation_model_key

            if copy_run_metadata_keys is not None:
                for run_metadata_field in copy_run_metadata_keys:
                    new_trial.update_run_metadata(
                        {run_metadata_field: trial.run_metadata.get(run_metadata_field)}
                    )
            # Trial has data, so we replicate it on the new experiment.
            has_data = ts != -1 and not dat.df.empty
            if has_data:
                new_df = dat.true_df.copy()
                new_df["trial_index"].replace(
                    {trial.index: new_trial.index}, inplace=True
                )
                new_df["arm_name"].replace(
                    {not_none(trial.arm).name: not_none(new_trial.arm).name},
                    inplace=True,
                )
                # Attach updated data to new trial on experiment.
                old_data = (
                    old_experiment.default_data_constructor(
                        df=new_df,
                        map_key_infos=checked_cast(
                            MapData, old_experiment.lookup_data()
                        ).map_key_infos,
                    )
                    if old_experiment.default_data_type == DataType.MAP_DATA
                    else old_experiment.default_data_constructor(df=new_df)
                )
                self.attach_data(data=old_data)
            if trial.status == TrialStatus.ABANDONED:
                new_trial.mark_abandoned(reason=trial.abandoned_reason)
            elif trial.status is not TrialStatus.RUNNING:
                new_trial.mark_as(trial.status)
            copied_trials.append(new_trial)

        if self._name is not None:
            logger.info(
                f"Copied {len(copied_trials)} completed trials and their data "
                f"from {old_experiment._name} to {self._name}."
            )
        else:
            logger.info(
                f"Copied {len(copied_trials)} completed trials and their data "
                f"from {old_experiment._name}."
            )

        return copied_trials

    def _name_and_store_arm_if_not_exists(self, arm: Arm, proposed_name: str) -> None:
        """Tries to lookup arm with same signature, otherwise names and stores it.

        - Looks up if arm already exists on experiment
            - If so, name the input arm the same as the existing arm
            - else name the arm with given name and store in _arms_by_signature

        Args:
            arm: The arm object to name.
            proposed_name: The name to assign if it doesn't have one already.
        """

        # If arm is identical to an existing arm, return that
        # so that the names match.
        if arm.signature in self.arms_by_signature:
            existing_arm = self.arms_by_signature[arm.signature]
            if arm.has_name:
                if arm.name != existing_arm.name:
                    raise ValueError(
                        f"Arm already exists with name {existing_arm.name}, "
                        f"which doesn't match given arm name of {arm.name}."
                    )
            else:
                arm.name = existing_arm.name
        else:
            if not arm.has_name:
                arm.name = proposed_name
            self._register_arm(arm)

    def _register_arm(self, arm: Arm) -> None:
        """Add a new arm to the experiment, updating the relevant
        lookup dictionaries.

        Args:
            arm: Arm to add
        """
        self._arms_by_signature[arm.signature] = arm
        self._arms_by_name[arm.name] = arm

    def _check_TTL_on_running_trials(self) -> None:
        """Checks whether any past-TTL trials are still marked as `RUNNING`
        and marks them as failed if so.

        NOTE: this function just calls `trial.status` for each trial, as the
        computation of that property checks the TTL for trials.
        """
        if not self._trials_have_ttl:
            return

        running = list(self._trial_indices_by_status[TrialStatus.RUNNING])
        for idx in running:
            self._trials[idx].status  # `status` property checks TTL if applicable.

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({self._name})"

    # --- MultiTypeExperiment convenience functions ---
    #
    # Certain functionalities have special behavior for multi-type experiments.
    # This defines the base behavior for regular experiments that will be
    # overridden in the MultiTypeExperiment class.

    @property
    def default_trial_type(self) -> Optional[str]:
        """Default trial type assigned to trials in this experiment.

        In the base experiment class this is always None. For experiments
        with multiple trial types, use the MultiTypeExperiment class.
        """
        return None

    def runner_for_trial(self, trial: BaseTrial) -> Optional[Runner]:
        """The default runner to use for a given trial.

        In the base experiment class, this is always the default experiment runner.
        For experiments with multiple trial types, use the MultiTypeExperiment class.
        """
        return self.runner

    def supports_trial_type(self, trial_type: Optional[str]) -> bool:
        """Whether this experiment allows trials of the given type.

        The base experiment class only supports None. For experiments
        with multiple trial types, use the MultiTypeExperiment class.
        """
        return trial_type is None


def add_arm_and_prevent_naming_collision(
    new_trial: Trial, old_trial: Trial, old_experiment_name: Optional[str] = None
) -> None:
    # Add all of an old trial's arms to a new trial. Rename any arm with auto-generated
    # naming format to prevent naming collisions during warm-start. If an old
    # experiment name is provided, append that to the original arm name. Else, clear
    # the arm name. Preserves all names not matching the automatic naming format.
    # experiment is not named, clear the arm's name.
    # `arm_index` is 0 since all trials are single-armed.
    old_arm_name = not_none(old_trial.arm).name
    has_default_name = bool(old_arm_name == old_trial._get_default_name(arm_index=0))
    if has_default_name:
        new_arm = not_none(old_trial.arm).clone(clear_name=True)
        if old_experiment_name is not None:
            new_arm.name = f"{old_arm_name}_{old_experiment_name}"
        new_trial.add_arm(new_arm)
    else:
        try:
            new_trial.add_arm(not_none(old_trial.arm).clone(clear_name=False))
        except ValueError as e:
            warnings.warn(
                f"Attaching arm {old_trial.arm} to trial {new_trial} while preserving "
                f"its name failed with error: {e}. Retrying with `clear_name=True`."
            )
            new_trial.add_arm(not_none(old_trial.arm).clone(clear_name=True))

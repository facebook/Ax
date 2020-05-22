#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import OrderedDict, defaultdict
from datetime import datetime
from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type

import pandas as pd
from ax.core.arm import Arm
from ax.core.base import Base
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import Parameter
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.utils.common.constants import UNEXPECTED_METRIC_COMBINATION
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from ax.utils.common.timeutils import current_timestamp_in_millis


logger: logging.Logger = get_logger(__name__)


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
        """
        # appease pyre
        self._search_space: SearchSpace
        self._status_quo: Optional[Arm] = None

        self._name = name
        self.description = description
        self.runner = runner
        self.is_test = is_test

        self._data_by_trial: Dict[int, OrderedDict[int, Data]] = {}
        self._experiment_type: Optional[str] = experiment_type
        self._optimization_config = None
        self._tracking_metrics: Dict[str, Metric] = {}
        self._time_created: datetime = datetime.now()
        self._trials: Dict[int, BaseTrial] = {}
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
        # pyre-fixme[7]: Expected `str` but got `Optional[str]`.
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Set experiment name."""
        self._name = name

    @property
    def is_simple_experiment(self):
        """Whether this experiment is a regular Experiment or the subclassing
        `SimpleExperiment`."""
        return False

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

        # TODO maybe return a copy here to guard against implicit changes
        return self._search_space

    @search_space.setter
    def search_space(self, search_space: SearchSpace) -> None:
        # Allow all modifications when no trials present.
        if len(self.trials) > 0:
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
            self.search_space.check_types(status_quo.parameters, raise_error=True)

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

    @optimization_config.setter
    def optimization_config(self, optimization_config: OptimizationConfig) -> None:
        for metric_name in optimization_config.metrics.keys():
            if metric_name in self._tracking_metrics:
                self.remove_tracking_metric(metric_name)
        self._optimization_config = optimization_config

    @property
    def data_by_trial(self) -> Dict[int, OrderedDict]:
        """Data stored on the experiment, indexed by trial index and storage time.

        First key is trial index and second key is storage time in milliseconds.
        For a given trial, data is ordered by storage time, so first added data
        will appear first in the list.
        """
        return self._data_by_trial

    def add_tracking_metric(self, metric: Metric) -> "Experiment":
        """Add a new metric to the experiment.

        Args:
            metric: Metric to be added.
        """
        if metric.name in self._tracking_metrics:
            raise ValueError(
                f"Metric `{metric.name}` already defined on experiment. "
                "Use `update_tracking_metric` to update an existing metric definition."
            )

        if self.optimization_config and metric.name in self.optimization_config.metrics:
            raise ValueError(
                f"Metric `{metric.name}` already present in experiment's "
                "OptimizationConfig. Set a new OptimizationConfig without this metric "
                "before adding it to tracking metrics."
            )

        self._tracking_metrics[metric.name] = metric
        return self

    def add_tracking_metrics(self, metrics: List[Metric]) -> "Experiment":
        """Add a list of new metrics to the experiment.

        If any of the metrics are already defined on the experiment,
        we raise an error and don't add any of them to the experiment

        Args:
            metrics: Metrics to be added.
        """
        # Before setting any metrics, we validate none are already on
        # the experiment
        for metric in metrics:
            if metric.name in self._tracking_metrics:
                raise ValueError(
                    f"Metric `{metric.name}` already defined on experiment. "
                    "Use `update_tracking_metric` to update an existing metric"
                    " definition."
                )

            if (
                self.optimization_config
                and metric.name in self.optimization_config.metrics
            ):
                raise ValueError(
                    f"Metric `{metric.name}` already present in experiment's "
                    "OptimizationConfig. Set a new OptimizationConfig without"
                    " this metric before adding it to tracking metrics."
                )
        for metric in metrics:
            self._tracking_metrics[metric.name] = metric
        return self

    def update_tracking_metric(self, metric: Metric) -> "Experiment":
        """Redefine a metric that already exists on the experiment.

        Args:
            metric: New metric definition.
        """
        if metric.name not in self._tracking_metrics:
            raise ValueError(f"Metric `{metric.name}` doesn't exist on experiment.")

        self._tracking_metrics[metric.name] = metric
        return self

    def remove_tracking_metric(self, metric_name: str) -> "Experiment":
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
            metrics_by_class[metric.__class__].append(metric)
        return metrics_by_class

    def fetch_data(self, metrics: Optional[List[Metric]] = None, **kwargs: Any) -> Data:
        """Fetches data for all metrics and trials on this experiment.

        Args:
            metrics: If provided, fetch data for these metrics instead of the ones
                defined on the experiment.
            kwargs: keyword args to pass to underlying metrics' fetch data functions.

        Returns:
            Data for the experiment.
        """
        return self._fetch_trials_data(
            trials=list(self.trials.values()), metrics=metrics, **kwargs
        )

    def fetch_trials_data(
        self,
        trial_indices: Iterable[int],
        metrics: Optional[List[Metric]] = None,
        **kwargs: Any,
    ) -> Data:
        """Fetches data for specific trials on the experiment.

        Args:
            trial_indices: Indices of trials, for which to fetch data.
            metrics: If provided, fetch data for these metrics instead of the ones
                defined on the experiment.
            kwargs: Keyword args to pass to underlying metrics' fetch data functions.

        Returns:
            Data for the specific trials on the experiment.
        """
        return self._fetch_trials_data(
            trials=self.get_trials_by_indices(trial_indices=trial_indices),
            metrics=metrics,
            **kwargs,
        )

    def _fetch_trials_data(
        self,
        trials: List[BaseTrial],
        metrics: Optional[Iterable[Metric]] = None,
        **kwargs: Any,
    ) -> Data:
        if not self.metrics and not metrics:
            raise ValueError(
                "No metrics to fetch data for, as no metrics are defined for "
                "this experiment, and none were passed in to `fetch_data`."
            )
        metrics = list(metrics or self.metrics.values())
        if all(type(m) is Metric for m in metrics):
            # All metrics are 'dummy' base `Metric` class metrics, which do not
            # implement actual data-fetching logic, so should look up attached
            # data instead of trying to fetch it via logic in metrics.
            return Data.from_multiple_data(
                [self.lookup_data_for_trial(trial_index=t.index)[0] for t in trials]
            )
        elif all(isinstance(m, Metric) and type(m) is not Metric for m in metrics):
            # All metrics are subclasses of `Metric`, which should implement fetching.
            data_list = [
                metric_cls.fetch_experiment_data_multi(
                    experiment=self, metrics=metric_list, trials=trials, **kwargs
                )
                for metric_cls, metric_list in self._metrics_by_class(
                    metrics=metrics
                ).items()
            ]
            # For trials in candidate phase, append any attached data
            for trial in trials:
                if trial.status == TrialStatus.CANDIDATE:
                    trial_data, _ = self.lookup_data_for_trial(trial_index=trial.index)
                    if not trial_data.df.empty:
                        data_list.append(trial_data)

            return Data.from_multiple_data(data_list)

        raise ValueError(UNEXPECTED_METRIC_COMBINATION)

    @copy_doc(BaseTrial.fetch_data)
    def _fetch_trial_data(
        self, trial_index: int, metrics: Optional[List[Metric]] = None, **kwargs: Any
    ) -> Data:
        if not self.metrics and not metrics:
            raise ValueError(
                "No metrics to fetch data for, as no metrics are defined for "
                "this experiment, and none were passed in to `fetch_trial_data`."
            )
        trial = self.trials[trial_index]
        metrics = list(metrics or self.metrics.values())

        if trial.status == TrialStatus.CANDIDATE or all(
            type(m) is Metric for m in metrics
        ):
            # Either trial is a `CANDIDATE` (so cannot use fetching logic) or
            # all metrics are 'dummy' base `Metric` class metrics, which do not
            # implement actual data-fetching logic. Should look up attached
            # data instead of trying to fetch it via logic in metrics.
            return self.lookup_data_for_trial(trial_index=trial_index)[0]

        elif all(isinstance(m, Metric) and type(m) is not Metric for m in metrics):
            # All metrics are subclasses of `Metric`, which should implement fetching.
            if not trial.status.expecting_data:
                return Data()
            return self._fetch_trial_data_no_lookup(
                trial_index=trial_index, metrics=metrics, **kwargs
            )

        raise ValueError(UNEXPECTED_METRIC_COMBINATION)

    def _fetch_trial_data_no_lookup(
        self, trial_index: int, metrics: Optional[List[Metric]], **kwargs: Any
    ) -> Data:
        """Fetches data explicitly from metric logic, does not look up attached
        data on experiment.
        """
        return Data.from_multiple_data(
            [
                metric_cls.fetch_trial_data_multi(
                    self.trials[trial_index], metric_list, **kwargs
                )
                for metric_cls, metric_list in self._metrics_by_class(
                    metrics=metrics
                ).items()
            ]
        )

    def attach_data(self, data: Data, combine_with_last_data: bool = False) -> int:
        """Attach data to experiment. Stores data in `experiment._data_by_trial`,
        to be looked up via `experiment.lookup_data_by_trial`.

        Args:
            data: Data object to store.
            combine_with_last_data: By default, when attaching data, it's identified
                by its timestamp, and `experiment.lookup_data_by_trial` returns
                data by most recent timestamp. In some cases, however, the goal
                is to combine all data attached for a trial into a single `Data`
                object. To achieve that goal, every call to `attach_data` after
                the initial data is attached to trials, should be set to `True`.
                Then, the newly attached data will be appended to existing data,
                rather than stored as a separate object, and `lookup_data_by_trial`
                will return the combined data object, rather than just the most
                recently added data. This will validate that the newly added data
                does not contain observations for the metrics that already have
                observations in the most recent data stored.

        Returns:
            Timestamp of storage in millis.
        """
        if data.df.empty:
            raise ValueError("Data to attach is empty.")
        cur_time_millis = current_timestamp_in_millis()
        for trial_index, trial_df in data.df.groupby(data.df["trial_index"]):
            current_trial_data = (
                self._data_by_trial[trial_index]
                if trial_index in self._data_by_trial
                else OrderedDict()
            )
            if combine_with_last_data and len(current_trial_data) > 0:
                last_ts, last_data = list(current_trial_data.items())[-1]
                merged = pd.merge(
                    last_data.df,
                    trial_df,
                    on=["trial_index", "metric_name", "arm_name"],
                    how="inner",
                )
                if not merged.empty:
                    raise ValueError(
                        f"Last data for trial {trial_index} already contained an "
                        f"observation for metric {merged.head()['metric_name']}."
                    )
                current_trial_data[cur_time_millis] = Data.from_multiple_data(
                    [last_data, Data(trial_df)]
                )
            else:
                current_trial_data[cur_time_millis] = Data(trial_df)
            self._data_by_trial[trial_index] = current_trial_data

        return cur_time_millis

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

        return Data.from_multiple_data(trial_datas)

    def lookup_data_for_trial(self, trial_index: int) -> Tuple[Data, int]:
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
            return (Data(), -1)

        if len(trial_data_dict) == 0:
            return (Data(), -1)

        storage_time = max(trial_data_dict.keys())
        trial_data = trial_data_dict[storage_time]
        return trial_data, storage_time

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
    def trial_indices_by_status(self) -> Dict[TrialStatus, Set[int]]:
        """Indices of trials associated with the experiment, grouped by trial
        status.
        """
        self._check_TTL_on_running_trials()  # Marks past-TTL trials as failed.
        return self._trial_indices_by_status

    def new_trial(
        self,
        generator_run: Optional[GeneratorRun] = None,
        trial_type: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Trial:
        """Create a new trial associated with this experiment.

        Args:
            generator_run: GeneratorRun, associated with this trial.
                Trial has only one generator run (and thus arm)
                attached to it. This can also be set later through `add_arm`
                or `add_generator_run`, but a trial's associated generator run is
                immutable once set.
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
        """
        if ttl_seconds is not None:
            self._trials_have_ttl = True
        return BatchTrial(
            experiment=self,
            trial_type=trial_type,
            generator_run=generator_run,
            optimize_for_power=optimize_for_power,
            ttl_seconds=ttl_seconds,
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

    def _attach_trial(self, trial: BaseTrial) -> int:
        """Attach a trial to this experiment.

        Should only be called within the trial constructor.

        Args:
            trial: The trial to be attached.

        Returns:
            The index of the trial within the experiment's trial list.
        """

        if trial.experiment is not self:
            raise ValueError("BatchTrial does not belong to this experiment.")

        for existing_trial in self._trials.values():
            if existing_trial is trial:
                raise ValueError("BatchTrial already attached to experiment.")

        index = 0 if len(self._trials) == 0 else max(self._trials.keys()) + 1
        self._trials[index] = trial
        return index

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

    @property
    def trials_expecting_data(self) -> List[BaseTrial]:
        """List[BaseTrial]: the list of all trials for which data has arrived
        or is expected to arrive.

        """
        return [trial for trial in self.trials.values() if trial.status.expecting_data]

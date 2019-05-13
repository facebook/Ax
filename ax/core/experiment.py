#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import time as time
from collections import OrderedDict, defaultdict
from datetime import datetime
from functools import reduce
from typing import Any, Dict, List, Optional, Type

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
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger


logger: logging.Logger = get_logger(__name__)


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
        """
        # appease pyre
        self._search_space: SearchSpace
        self._status_quo: Optional[Arm] = None

        self._name = name
        self.description = description
        self.runner = runner
        self.is_test = is_test

        self._data_by_trial: Dict[int, OrderedDict[int, Data]] = {}
        self._experiment_type: Optional[str] = None
        self._optimization_config = None
        self._tracking_metrics: Dict[str, Metric] = {}
        self._time_created: datetime = datetime.now()
        self._trials: Dict[int, BaseTrial] = {}
        self._arms_by_signature: Dict[str, Arm] = {}

        for metric in tracking_metrics or []:
            self.add_tracking_metric(metric)

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
                if self._status_quo.name in trial.arms_by_name:
                    persist_old_sq = True
                    break
            if not persist_old_sq:
                self._arms_by_signature.pop(self._status_quo.signature)

        self._status_quo = status_quo

    @property
    def parameters(self) -> Dict[str, Parameter]:
        """The parameters in the experiment's search space."""
        return self.search_space.parameters

    @property
    def arms_by_name(self) -> Dict[str, Arm]:
        """The arms belonging to this experiment, by their name."""
        return {arm.name: arm for arm in self._arms_by_signature.values()}

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
        if not self.metrics and not metrics:
            raise ValueError(
                "No metrics to fetch data for, as no metrics are defined for "
                "this experiment, and none were passed in to `fetch_data`."
            )
        try:
            data_list = [
                metric_cls.fetch_experiment_data_multi(self, metric_list, **kwargs)
                for metric_cls, metric_list in self._metrics_by_class(
                    metrics=metrics
                ).items()
            ]

            # For trials in candidate phase, append any attached data
            for trial in self.trials.values():
                if trial.status == TrialStatus.CANDIDATE:
                    trial_data = self.lookup_data_for_trial(trial_index=trial.index)
                    if not trial_data.df.empty:
                        data_list.append(trial_data)

            return Data.from_multiple_data(data_list)
        except NotImplementedError:
            # If some of the metrics do not implement data fetching, we should
            # fall back to data that has been attached.
            return Data.from_multiple_data(
                [self.lookup_data_for_trial(trial_index=idx) for idx in self.trials]
            )

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

        if trial.status == TrialStatus.CANDIDATE:
            return self.lookup_data_for_trial(trial_index=trial_index)
        elif not trial.status.expecting_data:
            return Data()

        try:
            return Data.from_multiple_data(
                [
                    metric_cls.fetch_trial_data_multi(trial, metric_list, **kwargs)
                    for metric_cls, metric_list in self._metrics_by_class(
                        metrics=metrics
                    ).items()
                ]
            )
        except NotImplementedError:
            # If some of the metrics do not implement data fetching, we should
            # fall back to data that has been attached.
            return self.lookup_data_for_trial(trial_index=trial_index)

    def attach_data(self, data: Data) -> int:
        """Attach data to experiment.

        Args:
            data: Data object to store.

        Returns:
            Timestamp of storage in millis.
        """
        cur_time_millis = int(round(time.time() * 1000))
        for trial_index, trial_df in data.df.groupby(data.df["trial_index"]):
            current_trial_data = (
                self._data_by_trial[trial_index]
                if trial_index in self._data_by_trial
                else OrderedDict()
            )
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

    def lookup_data_for_trial(self, trial_index: int) -> Data:
        """Lookup stored data for a specific trial.

        Returns latest data object present for this trial.
        Returns empty data if no data present.

        Args:
            trial_index: The index of the trial to lookup data for.

        Returns:
            Requested data object.
        """
        if trial_index not in self._data_by_trial:
            return Data()

        trial_data_list = list(self._data_by_trial[trial_index].values())
        return trial_data_list[-1] if len(trial_data_list) > 0 else Data()

    @property
    def num_trials(self) -> int:
        """How many trials are associated with this experiment."""
        return len(self._trials)

    @property
    def trials(self) -> Dict[int, BaseTrial]:
        """The trials associated with the experiment."""
        return self._trials

    def new_trial(
        self,
        generator_run: Optional[GeneratorRun] = None,
        trial_type: Optional[str] = None,
    ) -> Trial:
        """Create a new trial associated with this experiment."""
        return Trial(
            experiment=self, trial_type=trial_type, generator_run=generator_run
        )

    def new_batch_trial(
        self,
        generator_run: Optional[GeneratorRun] = None,
        trial_type: Optional[str] = None,
    ) -> BatchTrial:
        """Create a new batch trial associated with this experiment."""
        return BatchTrial(
            experiment=self, trial_type=trial_type, generator_run=generator_run
        )

    def _attach_trial(self, trial: BaseTrial) -> int:
        """Attach a trial to this experiment.

        Should only be called within the trial constructor.

        Args:
            trial: The trial to be attached.

        Returns:
            The index of the trial within the experiment's trial list.
        """

        if trial.experiment != self:
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
            self._arms_by_signature[arm.signature] = arm

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

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Set

from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.optimization_config import OptimizationConfig
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger


logger: logging.Logger = get_logger(__name__)


class MultiTypeExperiment(Experiment):
    """Class for experiment with multiple trial types.

    A canonical use case for this is tuning a large production system
    with limited evaluation budget and a simulator which approximates
    evaluations on the main system. Trial deployment and data fetching
    is separate for the two systems, but the final data is combined and
    fed into multi-task models.

    See the Multi-Task Modeling tutorial for more details.

    Attributes:
        name: Name of the experiment.
        description: Description of the experiment.
    """

    def __init__(
        self,
        name: str,
        search_space: SearchSpace,
        default_trial_type: str,
        default_runner: Runner,
        optimization_config: Optional[OptimizationConfig] = None,
        status_quo: Optional[Arm] = None,
        description: Optional[str] = None,
        is_test: bool = False,
        experiment_type: Optional[str] = None,
    ) -> None:
        """Inits Experiment.

        Args:
            name: Name of the experiment.
            search_space: Search space of the experiment.
            default_trial_type: Default type for trials on this experiment.
            default_runner: Default runner for trials of the default type.
            optimization_config: Optimization config of the experiment.
            tracking_metrics: Additional tracking metrics not used for optimization.
            runner: Default runner used for trials on this experiment.
            status_quo: Arm representing existing "control" arm.
            description: Description of the experiment.
            is_test: Convenience metadata tracker for the user to mark test experiments.
            experiment_type: The class of experiments this one belongs to.
        """

        self._default_trial_type = default_trial_type

        # Map from trial type to default runner of that type
        self._trial_type_to_runner: Dict[str, Runner] = {
            default_trial_type: default_runner
        }

        # Specifies which trial type each metric belongs to
        self._metric_to_trial_type: Dict[str, str] = {}

        # Maps certain metric names to a canonical name. Useful for ancillary trial
        # types' metrics, to specify which primary metrics they correspond to
        # (e.g. 'comment_prediction' => 'comment')
        self._metric_to_canonical_name: Dict[str, str] = {}

        # call super.__init__() after defining fields above, because we need
        # them to be populated before optimization config is set
        super(MultiTypeExperiment, self).__init__(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            status_quo=status_quo,
            description=description,
            is_test=is_test,
            experiment_type=experiment_type,
        )

    def add_trial_type(self, trial_type: str, runner: Runner) -> "MultiTypeExperiment":
        """Add a new trial_type to be supported by this experiment.

        Args:
            trial_type: The new trial_type to be added.
            runner: The default runner for trials of this type.
        """
        if self.supports_trial_type(trial_type):
            raise ValueError(f"Experiment already contains trial_type `{trial_type}`")

        self._trial_type_to_runner[trial_type] = runner
        return self

    def update_runner(self, trial_type: str, runner: Runner) -> "MultiTypeExperiment":
        """Update the default runner for an existing trial_type.

        Args:
            trial_type: The new trial_type to be added.
            runner: The new runner for trials of this type.
        """
        if not self.supports_trial_type(trial_type):
            raise ValueError(f"Experiment does not contain trial_type `{trial_type}`")

        self._trial_type_to_runner[trial_type] = runner
        return self

    def add_tracking_metric(
        self, metric: Metric, trial_type: str, canonical_name: Optional[str] = None
    ) -> "MultiTypeExperiment":
        """Add a new metric to the experiment.

        Args:
            metric: The metric to add.
            trial_type: The trial type for which this metric is used.
            canonical_name: The default metric for which this metric is a proxy.
        """
        if not self.supports_trial_type(trial_type):
            raise ValueError(f"`{trial_type}` is not a supported trial type.")

        super(MultiTypeExperiment, self).add_tracking_metric(metric)
        self._metric_to_trial_type[metric.name] = trial_type
        if canonical_name is not None:
            self._metric_to_canonical_name[metric.name] = canonical_name
        return self

    def update_tracking_metric(
        self, metric: Metric, trial_type: str, canonical_name: Optional[str] = None
    ) -> "MultiTypeExperiment":
        """Update an existing metric on the experiment.

        Args:
            metric: The metric to add.
            trial_type: The trial type for which this metric is used.
            canonical_name: The default metric for which this metric is a proxy.
        """
        oc = self.optimization_config
        oc_metrics = oc.metrics if oc else []
        if metric.name in oc_metrics and trial_type != self._default_trial_type:
            raise ValueError(
                f"Metric `{metric.name}` must remain a `{self._default_trial_type}` "
                "metric because it is part of the optimization_config."
            )
        elif not self.supports_trial_type(trial_type):
            raise ValueError(f"`{trial_type}` is not a supported trial type.")

        super(MultiTypeExperiment, self).update_tracking_metric(metric)
        self._metric_to_trial_type[metric.name] = trial_type
        if canonical_name is not None:
            self._metric_to_canonical_name[metric.name] = canonical_name
        return self

    @copy_doc(Experiment.remove_tracking_metric)
    def remove_tracking_metric(self, metric_name: str) -> "MultiTypeExperiment":
        if metric_name not in self._tracking_metrics:
            raise ValueError(f"Metric `{metric_name}` doesn't exist on experiment.")

        # Required fields
        del self._tracking_metrics[metric_name]
        del self._metric_to_trial_type[metric_name]

        # Optional
        if metric_name in self._metric_to_canonical_name:
            del self._metric_to_canonical_name[metric_name]
        return self

    @copy_doc(Experiment.fetch_data)
    def fetch_data(self, metrics: Optional[List[Metric]] = None, **kwargs: Any) -> Data:
        return Data.from_multiple_data(
            [
                trial.fetch_data(**kwargs, metrics=metrics)
                if trial.status.expecting_data
                else Data()
                for trial in self.trials.values()
            ]
        )

    @copy_doc(Experiment._fetch_trial_data)
    def _fetch_trial_data(
        self, trial_index: int, metrics: Optional[List[Metric]] = None, **kwargs: Any
    ) -> Data:
        trial = self.trials[trial_index]
        metrics = [
            metric
            for metric in (metrics or self.metrics.values())
            if self.metric_to_trial_type[metric.name] == trial.trial_type
        ]
        # Invoke parent's fetch method using only metrics for this trial_type
        return super()._fetch_trial_data(trial.index, metrics=metrics, **kwargs)

    @property
    def default_trials(self) -> Set[int]:
        """Return the indicies for trials of the default type."""
        return {
            idx
            for idx, trial in self.trials.items()
            if trial.trial_type == self.default_trial_type
        }

    @property
    def metric_to_trial_type(self) -> Dict[str, str]:
        """Map metrics to trial types.

        Adds in default trial type for OC metrics to custom defined trial types..
        """
        opt_config_types = {
            metric_name: self.default_trial_type
            for metric_name in self.optimization_config.metrics.keys()
        }
        return {**opt_config_types, **self._metric_to_trial_type}

    # -- Overridden functions from Base Experiment Class --
    @property
    def default_trial_type(self) -> Optional[str]:
        """Default trial type assigned to trials in this experiment."""
        return self._default_trial_type

    def runner_for_trial(self, trial: BaseTrial) -> Optional[Runner]:
        """The default runner to use for a given trial.

        Looks up the appropriate runner for this trial type in the trial_type_to_runner.
        """
        if trial.trial_type is None or not self.supports_trial_type(trial.trial_type):
            raise ValueError(f"Batch type `{trial.trial_type}` is not supported.")
        # pyre-fixme[6]: Expected `str` for 1st param but got `Optional[str]`.
        return self._trial_type_to_runner[trial.trial_type]

    def supports_trial_type(self, trial_type: Optional[str]) -> bool:
        """Whether this experiment allows trials of the given type.

        Only trial types defined in the trial_type_to_runner are allowed.
        """
        return trial_type in self._trial_type_to_runner.keys()

    def reset_runners(self, runner: Runner) -> None:
        raise NotImplementedError(
            "MultiTypeExperiment does not support resetting all runners."
        )

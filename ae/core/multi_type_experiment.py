#!/usr/bin/env python3

import logging
from typing import Any, Dict, Optional

from ae.lazarus.ae.core.arm import Arm
from ae.lazarus.ae.core.base_trial import BaseTrial
from ae.lazarus.ae.core.data import Data
from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.core.metric import Metric
from ae.lazarus.ae.core.optimization_config import OptimizationConfig
from ae.lazarus.ae.core.runner import Runner
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.utils.common.logger import get_logger


logger: logging.Logger = get_logger(__name__)


class MultiTypeExperiment(Experiment):
    """Class for experiment with multiple trial types.

    TODO add details.

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
        )

    @property
    def optimization_config(self) -> Optional[OptimizationConfig]:
        """The experiment's optimization config.

        All contained metrics are assumed to belong to the primary trial type.
        """
        return self._optimization_config

    @optimization_config.setter
    def optimization_config(self, optimization_config: OptimizationConfig) -> None:
        for metric_name, metric in optimization_config.metrics.items():
            self._metrics[metric_name] = metric
            self._metric_to_trial_type[metric_name] = self._default_trial_type
        self._optimization_config = optimization_config

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

    def add_metric(
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

        super(MultiTypeExperiment, self).add_metric(metric)
        self._metric_to_trial_type[metric.name] = trial_type
        if canonical_name is not None:
            self._metric_to_canonical_name[metric.name] = canonical_name
        return self

    def update_metric(
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

        super(MultiTypeExperiment, self).update_metric(metric)
        self._metric_to_trial_type[metric.name] = trial_type
        if canonical_name is not None:
            self._metric_to_canonical_name[metric.name] = canonical_name
        return self

    def fetch_data(self, **kwargs: Any) -> Data:
        """Fetches data for all metrics and trials on this experiment."""
        return Data.from_multiple_data(
            [
                self.fetch_trial_data(trial.index, **kwargs)
                if trial.status.expecting_data
                else Data()
                for trial in self.trials.values()
            ]
        )

    def fetch_trial_data(self, trial_index: int, **kwargs: Any) -> Data:
        """Fetches data for all metrics and a single trial on this experiment."""
        trial = self.trials[trial_index]
        return Data.from_multiple_data(
            [
                metric.fetch_trial_data(trial, **kwargs)
                if trial.trial_type == self._metric_to_trial_type[metric.name]
                else Data()
                for metric in self._metrics.values()
            ]
        )

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
        return self._trial_type_to_runner[trial.trial_type]

    def supports_trial_type(self, trial_type: Optional[str]) -> bool:
        """Whether this experiment allows trials of the given type.

        Only trial types defined in the trial_type_to_runner are allowed.
        """
        return trial_type in self._trial_type_to_runner.keys()

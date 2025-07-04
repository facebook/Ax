#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterable, Sequence
from typing import Any

from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.experiment import DataType, Experiment
from ax.core.metric import Metric, MetricFetchResult
from ax.core.optimization_config import OptimizationConfig
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.utils.common.docutils import copy_doc
from pyre_extensions import none_throws


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
        default_runner: Runner | None,
        optimization_config: OptimizationConfig | None = None,
        tracking_metrics: list[Metric] | None = None,
        status_quo: Arm | None = None,
        description: str | None = None,
        is_test: bool = False,
        experiment_type: str | None = None,
        properties: dict[str, Any] | None = None,
        default_data_type: DataType | None = None,
    ) -> None:
        """Inits Experiment.

        Args:
            name: Name of the experiment.
            search_space: Search space of the experiment.
            default_trial_type: Default type for trials on this experiment.
            default_runner: Default runner for trials of the default type.
            optimization_config: Optimization config of the experiment.
            tracking_metrics: Additional tracking metrics not used for optimization.
                These are associated with the default trial type.
            runner: Default runner used for trials on this experiment.
            status_quo: Arm representing existing "control" arm.
            description: Description of the experiment.
            is_test: Convenience metadata tracker for the user to mark test experiments.
            experiment_type: The class of experiments this one belongs to.
            properties: Dictionary of this experiment's properties.
            default_data_type: Enum representing the data type this experiment uses.
        """

        self._default_trial_type = default_trial_type

        # Map from trial type to default runner of that type
        self._trial_type_to_runner: dict[str, Runner | None] = {
            default_trial_type: default_runner
        }

        # Specifies which trial type each metric belongs to
        self._metric_to_trial_type: dict[str, str] = {}

        # Maps certain metric names to a canonical name. Useful for ancillary trial
        # types' metrics, to specify which primary metrics they correspond to
        # (e.g. 'comment_prediction' => 'comment')
        self._metric_to_canonical_name: dict[str, str] = {}

        # call super.__init__() after defining fields above, because we need
        # them to be populated before optimization config is set
        super().__init__(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            status_quo=status_quo,
            description=description,
            is_test=is_test,
            experiment_type=experiment_type,
            properties=properties,
            default_data_type=default_data_type,
            tracking_metrics=tracking_metrics,
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

    # pyre-fixme [56]: Pyre was not able to infer the type of the decorator
    # `Experiment.optimization_config.setter`.
    @Experiment.optimization_config.setter
    def optimization_config(self, optimization_config: OptimizationConfig) -> None:
        # pyre-fixme [16]: `Optional` has no attribute `fset`.
        Experiment.optimization_config.fset(self, optimization_config)
        for metric_name in optimization_config.metrics.keys():
            # Optimization config metrics are required to be the default trial type
            # currently. TODO: remove that restriction (T202797235)
            self._metric_to_trial_type[metric_name] = none_throws(
                self.default_trial_type
            )
        # prune metrics that are no longer attached to the experiment
        for metric_name in list(self._metric_to_trial_type.keys()):
            if metric_name not in self.metrics:
                del self._metric_to_trial_type[metric_name]

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

    # pyre-fixme[14]: `add_tracking_metric` overrides method defined in `Experiment`
    #  inconsistently.
    def add_tracking_metric(
        self, metric: Metric, trial_type: str, canonical_name: str | None = None
    ) -> "MultiTypeExperiment":
        """Add a new metric to the experiment.

        Args:
            metric: The metric to add.
            trial_type: The trial type for which this metric is used.
            canonical_name: The default metric for which this metric is a proxy.
        """
        if not self.supports_trial_type(trial_type):
            raise ValueError(f"`{trial_type}` is not a supported trial type.")

        super().add_tracking_metric(metric)
        self._metric_to_trial_type[metric.name] = trial_type
        if canonical_name is not None:
            self._metric_to_canonical_name[metric.name] = canonical_name
        return self

    def add_tracking_metrics(
        self,
        metrics: list[Metric],
        metrics_to_trial_types: dict[str, str] | None = None,
        canonical_names: dict[str, str] | None = None,
    ) -> Experiment:
        """Add a list of new metrics to the experiment.

        If any of the metrics are already defined on the experiment,
        we raise an error and don't add any of them to the experiment

        Args:
            metrics: Metrics to be added.
            metrics_to_trial_types: The mapping from metric names to corresponding
                trial types for each metric. If provided, the metrics will be
                added to their trial types. If not provided, then the default
                trial type will be used.
            canonical_names: A mapping of metric names to their
                canonical names(The default metrics for which the metrics are
                proxies.)

        Returns:
            The experiment with the added metrics.
        """
        metrics_to_trial_types = metrics_to_trial_types or {}
        canonical_name = None
        for metric in metrics:
            if canonical_names is not None:
                canonical_name = none_throws(canonical_names).get(metric.name, None)

            self.add_tracking_metric(
                metric=metric,
                trial_type=metrics_to_trial_types.get(
                    metric.name, self._default_trial_type
                ),
                canonical_name=canonical_name,
            )
        return self

    # pyre-fixme[14]: `update_tracking_metric` overrides method defined in
    #  `Experiment` inconsistently.
    def update_tracking_metric(
        self, metric: Metric, trial_type: str, canonical_name: str | None = None
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

        super().update_tracking_metric(metric)
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
    def fetch_data(
        self,
        trial_indices: Iterable[int] | None = None,
        metrics: list[Metric] | None = None,
        combine_with_last_data: bool = False,
        overwrite_existing_data: bool = False,
        **kwargs: Any,
    ) -> Data:
        # TODO: make this more efficient for fetching
        # data for multiple trials of the same type
        # by overriding Experiment._lookup_or_fetch_trials_results
        return self.default_data_constructor.from_multiple_data(
            [
                (
                    trial.fetch_data(**kwargs, metrics=metrics)
                    if trial.status.expecting_data
                    else Data()
                )
                for trial in self.trials.values()
            ]
        )

    @copy_doc(Experiment._fetch_trial_data)
    def _fetch_trial_data(
        self, trial_index: int, metrics: list[Metric] | None = None, **kwargs: Any
    ) -> dict[str, MetricFetchResult]:
        trial = self.trials[trial_index]
        metrics = [
            metric
            for metric in (metrics or self.metrics.values())
            if self.metric_to_trial_type[metric.name] == trial.trial_type
        ]
        # Invoke parent's fetch method using only metrics for this trial_type
        return super()._fetch_trial_data(trial.index, metrics=metrics, **kwargs)

    @property
    def default_trials(self) -> set[int]:
        """Return the indicies for trials of the default type."""
        return {
            idx
            for idx, trial in self.trials.items()
            if trial.trial_type == self.default_trial_type
        }

    @property
    def metric_to_trial_type(self) -> dict[str, str]:
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
    def default_trial_type(self) -> str | None:
        """Default trial type assigned to trials in this experiment."""
        return self._default_trial_type

    def runner_for_trial(self, trial: BaseTrial) -> Runner | None:
        """The default runner to use for a given trial.

        Looks up the appropriate runner for this trial type in the trial_type_to_runner.
        """
        return (
            trial._runner
            if trial._runner
            else self.runner_for_trial_type(trial_type=none_throws(trial.trial_type))
        )

    def runner_for_trial_type(self, trial_type: str) -> Runner | None:
        """The default runner to use for a given trial type.

        Looks up the appropriate runner for this trial type in the trial_type_to_runner.
        """
        if not self.supports_trial_type(trial_type):
            raise ValueError(f"Trial type `{trial_type}` is not supported.")
        return self._trial_type_to_runner[trial_type]

    def metrics_for_trial_type(self, trial_type: str) -> list[Metric]:
        """The default runner to use for a given trial type.

        Looks up the appropriate runner for this trial type in the trial_type_to_runner.
        """
        if not self.supports_trial_type(trial_type):
            raise ValueError(f"Trial type `{trial_type}` is not supported.")
        return [
            self.metrics[metric_name]
            for metric_name, metric_trial_type in self._metric_to_trial_type.items()
            if metric_trial_type == trial_type
        ]

    def supports_trial_type(self, trial_type: str | None) -> bool:
        """Whether this experiment allows trials of the given type.

        Only trial types defined in the trial_type_to_runner are allowed.
        """
        return trial_type in self._trial_type_to_runner.keys()

    def reset_runners(self, runner: Runner) -> None:
        raise NotImplementedError(
            "MultiTypeExperiment does not support resetting all runners."
        )


def filter_trials_by_type(
    trials: Sequence[BaseTrial], trial_type: str | None
) -> list[BaseTrial]:
    """Filter trials by trial type if provided.

    This filters trials by trial type if the experiment is a
    MultiTypeExperiment.

    Args:
        trials: Trials to filter.

    Returns:
        Filtered trials.
    """
    if trial_type is not None:
        return [t for t in trials if t.trial_type == trial_type]
    return list(trials)


def get_trial_indices_for_statuses(
    experiment: Experiment, statuses: set[TrialStatus], trial_type: str | None = None
) -> set[int]:
    """Get trial indices for a set of statuses.

    Args:
        statuses: Set of statuses to get trial indices for.

    Returns:
        Set of trial indices for the given statuses.
    """
    return {
        i
        for i, t in experiment.trials.items()
        if (t.status in statuses)
        and (
            (trial_type is None)
            or ((trial_type is not None) and (t.trial_type == trial_type))
        )
    }

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Self

from ax.core.arm import Arm
from ax.core.experiment import (
    Experiment,
    filter_trials_by_type,
    get_trial_indices_for_statuses,
)
from ax.core.metric import Metric
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
        default_data_type: Any = None,
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
            default_data_type: Deprecated and ignored.
        """

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
            tracking_metrics=tracking_metrics,
            runner=default_runner,
            default_trial_type=default_trial_type,
            default_data_type=default_data_type,
        )

        # Ensure tracking metrics are registered in _metric_to_trial_type.
        # The base __init__ handles _trial_type_to_metric_names.
        for m in tracking_metrics or []:
            if m.name not in self._metric_to_trial_type:
                self._metric_to_trial_type[m.name] = none_throws(
                    self._default_trial_type
                )

    # pyre does not support inferring the type of property setter decorators
    # or the `.fset` attribute on properties.
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator.
    @Experiment.optimization_config.setter
    def optimization_config(self, optimization_config: OptimizationConfig) -> None:
        # pyre-fixme[16]: `Optional` has no attribute `fset`.
        Experiment.optimization_config.fset(self, optimization_config)
        # Base setter handles _trial_type_to_metric_names; update legacy dict.
        for metric_name in optimization_config.metric_names:
            self._metric_to_trial_type[metric_name] = none_throws(
                self.default_trial_type
            )

    def add_tracking_metric(
        self,
        metric: Metric,
        trial_type: str | None = None,
        canonical_name: str | None = None,
    ) -> Self:
        """Add a new metric to the experiment.

        Args:
            metric: The metric to add.
            trial_type: The trial type for which this metric is used.
            canonical_name: The default metric for which this metric is a proxy.
        """
        if trial_type is None:
            trial_type = self._default_trial_type
        self.add_metric(metric, trial_type=trial_type)
        self._metric_to_trial_type[metric.name] = none_throws(trial_type)
        if canonical_name is not None:
            self._metric_to_canonical_name[metric.name] = canonical_name
        return self

    def update_tracking_metric(
        self,
        metric: Metric,
        trial_type: str | None = None,
        canonical_name: str | None = None,
    ) -> Self:
        """Update an existing metric on the experiment.

        Args:
            metric: The metric to add.
            trial_type: The trial type for which this metric is used. Defaults to
                the current trial type of the metric (if set), or the default trial
                type otherwise.
            canonical_name: The default metric for which this metric is a proxy.
        """
        # Default to the existing trial type if not specified
        if trial_type is None:
            trial_type = self._metric_to_trial_type.get(
                metric.name, self._default_trial_type
            )
        self.update_metric(metric, trial_type=trial_type)
        self._metric_to_trial_type[metric.name] = none_throws(trial_type)
        if canonical_name is not None:
            self._metric_to_canonical_name[metric.name] = canonical_name
        return self

    @copy_doc(Experiment.remove_metric)
    def remove_metric(self, metric_name: str) -> Self:
        super().remove_metric(metric_name)
        self._metric_to_trial_type.pop(metric_name, None)
        self._metric_to_canonical_name.pop(metric_name, None)
        return self


# Re-exported from ax.core.experiment for backward compatibility.
__all__ = [
    "MultiTypeExperiment",
    "filter_trials_by_type",
    "get_trial_indices_for_statuses",
]

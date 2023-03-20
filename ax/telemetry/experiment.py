# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Dict, List, Optional, Tuple

from ax.core.base_trial import TrialStatus

from ax.core.experiment import Experiment
from ax.core.map_metric import MapMetric
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace
from ax.core.utils import get_model_times
from ax.telemetry.common import INITIALIZATION_MODELS, OTHER_MODELS

INITIALIZATION_MODEL_STRS: List[str] = [enum.value for enum in INITIALIZATION_MODELS]
OTHER_MODEL_STRS: List[str] = [enum.value for enum in OTHER_MODELS]


@dataclass(frozen=True)
class ExperimentCreatedRecord:
    """
    Record of the Experiment creation event. This can be used for telemetry in settings
    where many Experiments are being created either manually or programatically. In
    order to facilitate easy serialization only include simple types: numbers, strings,
    bools, and None.
    """

    experiment_name: Optional[str]
    experiment_type: Optional[str]

    # SearchSpace info
    num_continuous_range_parameters: int

    # Note: ordered ChoiceParameters and int RangeParameters should both utilize the
    # following fields
    num_int_range_parameters_small: int  # 2 - 3 elements
    num_int_range_parameters_medium: int  # 4 - 7 elements
    num_int_range_parameters_large: int  # 8 or more elements

    # Any RangeParameter can specify log space sampling
    num_log_scale_range_parameters: int

    num_unordered_choice_parameters_small: int  # 2 - 3 elements
    num_unordered_choice_parameters_medium: int  # 4 - 7 elements
    num_unordered_choice_parameters_large: int  # 8 or more elements

    num_fixed_parameters: int

    dimensionality: int
    hierarchical_tree_height: int  # Height of tree for HSS, 1 for base SearchSpace
    num_parameter_constraints: int

    # OptimizationConfig info
    num_objectives: int
    num_tracking_metrics: int
    num_outcome_constraints: int  # Includes ObjectiveThresholds in MOO

    # General Metrics info
    num_map_metrics: int
    metric_cls_to_quantity: Dict[str, int]

    # Runner info
    runner_cls: str

    @classmethod
    def from_experiment(cls, experiment: Experiment) -> ExperimentCreatedRecord:
        (
            num_continuous_range_parameters,
            num_int_range_parameters_small,
            num_int_range_parameters_medium,
            num_int_range_parameters_large,
            num_log_scale_range_parameters,
            num_unordered_choice_parameters_small,
            num_unordered_choice_parameters_medium,
            num_unordered_choice_parameters_large,
            num_fixed_parameters,
        ) = cls._get_param_counts_from_search_space(
            search_space=experiment.search_space
        )

        return cls(
            experiment_name=experiment._name,
            experiment_type=experiment._experiment_type,
            num_continuous_range_parameters=num_continuous_range_parameters,
            num_int_range_parameters_small=num_int_range_parameters_small,
            num_int_range_parameters_medium=num_int_range_parameters_medium,
            num_int_range_parameters_large=num_int_range_parameters_large,
            num_log_scale_range_parameters=num_log_scale_range_parameters,
            num_unordered_choice_parameters_small=num_unordered_choice_parameters_small,
            num_unordered_choice_parameters_medium=(
                num_unordered_choice_parameters_medium
            ),
            num_unordered_choice_parameters_large=num_unordered_choice_parameters_large,
            num_fixed_parameters=num_fixed_parameters,
            dimensionality=sum(
                1 for param in experiment.parameters.values() if param.cardinality() > 1
            ),
            hierarchical_tree_height=experiment.search_space.height
            if isinstance(experiment.search_space, HierarchicalSearchSpace)
            else 1,
            num_parameter_constraints=len(
                experiment.search_space.parameter_constraints
            ),
            num_objectives=len(experiment.optimization_config.objective.metrics)
            if experiment.optimization_config is not None
            else 0,
            num_tracking_metrics=len(experiment.tracking_metrics),
            num_outcome_constraints=len(
                experiment.optimization_config.outcome_constraints
            )
            if experiment.optimization_config is not None
            else 0,
            num_map_metrics=sum(
                1
                for metric in experiment.metrics.values()
                if isinstance(metric, MapMetric)
            ),
            metric_cls_to_quantity={
                cls_name: sum(
                    1
                    for metric in experiment.metrics.values()
                    if metric.__class__.__name__ == cls_name
                )
                for cls_name in {
                    metric.__class__.__name__ for metric in experiment.metrics.values()
                }
            },
            runner_cls=experiment.runner.__class__.__name__,
        )

    @staticmethod
    def _get_param_counts_from_search_space(
        search_space: SearchSpace,
    ) -> Tuple[int, int, int, int, int, int, int, int, int]:
        """
        Return counts of different types of parameters.

        returns:
            num_continuous_range_parameters

            num_int_range_parameters_small
            num_int_range_parameters_medium
            num_int_range_parameters_large

            num_log_scale_range_parameters

            num_unordered_choice_parameters_small
            num_unordered_choice_parameters_medium
            num_unordered_choice_parameters_large

            num_fixed_parameters
        """

        num_continuous_range_parameters = sum(
            1
            for param in search_space.parameters.values()
            if isinstance(param, RangeParameter)
            and param.parameter_type == ParameterType.FLOAT
        )
        num_int_range_parameters_small = sum(
            1
            for param in search_space.parameters.values()
            if (
                isinstance(param, RangeParameter)
                or (isinstance(param, ChoiceParameter) and param.is_ordered)
            )
            and (1 < param.cardinality() <= 3)
        )
        num_int_range_parameters_medium = sum(
            1
            for param in search_space.parameters.values()
            if (
                isinstance(param, RangeParameter)
                or (isinstance(param, ChoiceParameter) and param.is_ordered)
            )
            and (3 < param.cardinality() <= 7)
        )
        num_int_range_parameters_large = sum(
            1
            for param in search_space.parameters.values()
            if (
                isinstance(param, RangeParameter)
                or (isinstance(param, ChoiceParameter) and param.is_ordered)
            )
            and (7 < param.cardinality() < inf)
        )
        num_log_scale_range_parameters = sum(
            1
            for param in search_space.parameters.values()
            if isinstance(param, RangeParameter) and param.log_scale
        )
        num_unordered_choice_parameters_small = sum(
            1
            for param in search_space.parameters.values()
            if (isinstance(param, ChoiceParameter) and not param.is_ordered)
            and (1 < param.cardinality() <= 3)
        )
        num_unordered_choice_parameters_medium = sum(
            1
            for param in search_space.parameters.values()
            if (isinstance(param, ChoiceParameter) and not param.is_ordered)
            and (3 < param.cardinality() <= 7)
        )
        num_unordered_choice_parameters_large = sum(
            1
            for param in search_space.parameters.values()
            if (isinstance(param, ChoiceParameter) and not param.is_ordered)
            and param.cardinality() > 7
        )
        num_fixed_parameters = sum(
            1
            for param in search_space.parameters.values()
            if isinstance(param, FixedParameter)
        )

        return (
            num_continuous_range_parameters,
            num_int_range_parameters_small,
            num_int_range_parameters_medium,
            num_int_range_parameters_large,
            num_log_scale_range_parameters,
            num_unordered_choice_parameters_small,
            num_unordered_choice_parameters_medium,
            num_unordered_choice_parameters_large,
            num_fixed_parameters,
        )


@dataclass(frozen=True)
class ExperimentCompletedRecord:
    """
    Record of the Experiment completion event. This can be used for telemetry in
    settings where many Experiments are being created either manually or
    programatically. In order to facilitate easy serialization only include simple
    types: numbers, strings, bools, and None.
    """

    num_initialization_trials: int
    num_bayesopt_trials: int
    num_other_trials: int

    num_completed_trials: int
    num_failed_trials: int
    num_abandoned_trials: int
    num_early_stopped_trials: int

    total_fit_time: int
    total_gen_time: int

    @classmethod
    def from_experiment(cls, experiment: Experiment) -> ExperimentCompletedRecord:
        trial_count_by_status = {
            status: len(trials)
            for status, trials in experiment.trials_by_status.items()
        }

        model_keys = [
            trial.generator_runs[0]._model_key for trial in experiment.trials.values()
        ]

        fit_time, gen_time = get_model_times(experiment=experiment)

        return cls(
            num_initialization_trials=sum(
                1 for model_key in model_keys if model_key in INITIALIZATION_MODEL_STRS
            ),
            num_bayesopt_trials=sum(
                1
                for model_key in model_keys
                if not (
                    model_key in INITIALIZATION_MODEL_STRS
                    or model_key in OTHER_MODEL_STRS
                )
            ),
            num_other_trials=sum(
                1 for model_key in model_keys if model_key in OTHER_MODEL_STRS
            ),
            num_completed_trials=trial_count_by_status[TrialStatus.COMPLETED],
            num_failed_trials=trial_count_by_status[TrialStatus.FAILED],
            num_abandoned_trials=trial_count_by_status[TrialStatus.ABANDONED],
            num_early_stopped_trials=trial_count_by_status[TrialStatus.EARLY_STOPPED],
            total_fit_time=int(fit_time),
            total_gen_time=int(gen_time),
        )

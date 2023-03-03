# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ExperimentCreatedRecord:
    """
    Record of the Experiment creation event. This can be used for telemetry in settings
    where many Experiments are being created either manually or programatically. In
    order to facilitate easy serialization only include simple types: numbers, strings,
    bools, and None.
    """

    experiment_name: Optional[str]

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
    heirerarchical_tree_height: int  # Height of tree for HSS, 1 for base SearchSpace
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
    # This could be the name of a training job, etc.
    trial_evaluation_identifier: Optional[str]


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

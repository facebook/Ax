# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationSummary:
    """Summary of an experiment's configuration for tier classification.

    Required Keys:
        num_params: Total number of tunable parameters.
        num_binary: Number of binary (0/1 integer) parameters.
        num_categorical_3_5: Number of unordered choice parameters with 3-5 options.
        num_categorical_6_inf: Number of unordered choice parameters with 6+ options.
        num_parameter_constraints: Number of parameter constraints.
        num_objectives: Number of optimization objectives.
        num_outcome_constraints: Number of outcome constraints.
        uses_early_stopping: Whether early stopping is enabled.
        uses_global_stopping: Whether global stopping is enabled.
        all_inputs_are_configs: Whether all inputs are high-level configs
            (as opposed to low-level Ax abstractions).

    Optional Keys:
        max_trials: Maximum number of trials (required if all_inputs_are_configs
            is True).
        tolerated_trial_failure_rate: Maximum tolerated trial failure rate
            (should be <= 0.9).
        max_pending_trials: Maximum number of pending trials.
        min_failed_trials_for_failure_rate_check: Minimum failed trials before
            failure rate is checked.
        non_default_advanced_options: Whether non-default advanced options are set.
        uses_merge_multiple_curves: Whether merge_multiple_curves is used
            (not supported).
    """

    # Required keys
    num_params: int
    num_binary: int
    num_categorical_3_5: int
    num_categorical_6_inf: int
    num_parameter_constraints: int
    num_objectives: int
    num_outcome_constraints: int
    uses_early_stopping: bool
    uses_global_stopping: bool
    all_inputs_are_configs: bool
    # Optional keys
    max_trials: int | None = None
    tolerated_trial_failure_rate: float | None = None
    max_pending_trials: int | None = None
    min_failed_trials_for_failure_rate_check: int | None = None
    non_default_advanced_options: bool | None = None
    uses_merge_multiple_curves: bool | None = None

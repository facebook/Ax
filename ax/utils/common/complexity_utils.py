# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any

from ax.adapter.parameter_utils import can_map_to_binary, is_unordered_choice
from ax.core.experiment import Experiment
from ax.exceptions.core import OptimizationNotConfiguredError
from ax.utils.common.tier_utils import (  # noqa: F401
    check_if_in_standard,
    DEFAULT_TIER_MESSAGES,
    format_tier_message,
    OptimizationSummary,
    TierMessages,
)


def summarize_ax_optimization_complexity(
    experiment: Experiment,
    tier_metadata: dict[str, Any],
    uses_early_stopping: bool = False,
    uses_global_stopping: bool = False,
    tolerated_trial_failure_rate: float | None = 0.5,
    max_pending_trials: int | None = 10,
    min_failed_trials_for_failure_rate_check: int | None = 5,
) -> OptimizationSummary:
    """Summarize the experiment's optimization complexity.

    This function analyzes an experiment's configuration and returns metrics and key
    characteristics that help assess the difficulty of the optimization problem.

    Args:
        experiment: The Ax Experiment.
        tier_metadata: tier-related meta-data from the orchestrator.
        uses_early_stopping: Whether early stopping is enabled.
        uses_global_stopping: Whether global stopping is enabled.
        tolerated_trial_failure_rate: The tolerated trial failure rate.
        max_pending_trials: The maximum number of pending trials.
        min_failed_trials_for_failure_rate_check: The minimum number of failed
            trials before checking the failure rate.

    Returns:
        A dictionary summarizing the experiment.
    """
    search_space = experiment.search_space
    optimization_config = experiment.optimization_config
    if optimization_config is None:
        raise OptimizationNotConfiguredError(
            "Experiment must have an optimization_config."
        )
    params = search_space.tunable_parameters.values()

    max_trials = tier_metadata.get("user_supplied_max_trials", None)
    num_params = len(search_space.tunable_parameters)
    num_binary = sum(can_map_to_binary(p) for p in params)
    num_categorical_3_5 = sum(
        is_unordered_choice(p, min_choices=3, max_choices=5) for p in params
    )
    num_categorical_6_inf = sum(is_unordered_choice(p, min_choices=6) for p in params)
    num_parameter_constraints = len(search_space.parameter_constraints)
    num_objectives = (
        len(optimization_config.objective.metric_names)
        if optimization_config.objective.is_multi_objective
        else 1
    )
    num_outcome_constraints = len(optimization_config.outcome_constraints)

    # Check if any metrics use merge_multiple_curves
    uses_merge_multiple_curves = False
    all_metrics = [
        experiment.get_metric(name) for name in optimization_config.metric_names
    ]
    if hasattr(experiment, "tracking_metrics"):
        all_metrics.extend(experiment.tracking_metrics)
    for metric in all_metrics:
        if getattr(metric, "merge_multiple_curves", False):
            uses_merge_multiple_curves = True
            break

    # Support both new key and old key for backward compatibility
    uses_standard_api = tier_metadata.get("uses_standard_api")
    if uses_standard_api is None:
        uses_standard_api = tier_metadata.get("all_inputs_are_configs", False)

    return OptimizationSummary(
        max_trials=max_trials,
        num_params=num_params,
        num_binary=num_binary,
        num_categorical_3_5=num_categorical_3_5,
        num_categorical_6_inf=num_categorical_6_inf,
        num_parameter_constraints=num_parameter_constraints,
        num_objectives=num_objectives,
        num_outcome_constraints=num_outcome_constraints,
        uses_early_stopping=uses_early_stopping,
        uses_global_stopping=uses_global_stopping,
        uses_merge_multiple_curves=uses_merge_multiple_curves,
        uses_standard_api=uses_standard_api,
        tolerated_trial_failure_rate=tolerated_trial_failure_rate,
        max_pending_trials=max_pending_trials,
        min_failed_trials_for_failure_rate_check=(
            min_failed_trials_for_failure_rate_check
        ),
    )

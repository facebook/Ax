# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any

from ax.adapter.parameter_utils import can_map_to_binary, is_unordered_choice
from ax.core.experiment import Experiment
from ax.exceptions.core import OptimizationNotConfiguredError
from ax.orchestration.orchestrator import OrchestratorOptions
from ax.utils.common.tier_utils import (  # noqa: F401
    check_if_in_standard,
    DEFAULT_TIER_MESSAGES,
    format_tier_message,
    OptimizationSummary,
    TierMessages,
)


def summarize_ax_optimization_complexity(
    experiment: Experiment,
    options: OrchestratorOptions,
    tier_metadata: dict[str, Any],
) -> OptimizationSummary:
    """Summarize the experiment's optimization complexity.

    This function analyzes an experiment's configuration and returns metrics and key
    characteristics that help assess the difficulty of the optimization problem.

    Args:
        experiment: The Ax Experiment.
        options: The orchestrator options.
        tier_metadata: tier-related meta-data from the orchestrator.

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
    uses_early_stopping = options.early_stopping_strategy is not None
    uses_global_stopping = options.global_stopping_strategy is not None

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
        tolerated_trial_failure_rate=options.tolerated_trial_failure_rate,
        max_pending_trials=options.max_pending_trials,
        min_failed_trials_for_failure_rate_check=(
            options.min_failed_trials_for_failure_rate_check
        ),
    )

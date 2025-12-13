# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ax.adapter.adapter_utils import can_map_to_binary, is_unordered_choice
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective
from ax.exceptions.core import OptimizationNotConfiguredError, UserInputError
from ax.service.orchestrator import OrchestratorOptions

WHEELHOUSE_TIER_MESSAGE = """This experiment is in tier 'Wheelhouse'.

Experiments belonging to this tier should not run into any problems! If an issue \
does occur, please post to our github issues page.
"""

ADVANCED_TIER_MESSAGE = """This experiment is in tier 'Advanced'.

This experiment should technically run, but uses advanced features that may not \
be well-tested and/or may not be compatible with other advanced features. We \
encourage users to raise issues encountered in advanced workflows just like \
wheelhouse workflows but it is also possible that reducing the complexity of \
your setup can solve your issue.
"""

UNSUPPORTED_TIER_MESSAGE = """This experiment is in tier 'Unsupported'.

You are pushing Ax beyond its limits. Please post to our github issues page for help \
in improving/simplifying your configuration to conform to a more \
well-supported usage tier if possible.
"""

WIKI_TIER_MESSAGE = "https://ax.dev/docs/why-ax"

UNKNOWN_TIER_MESSAGE = """Failed to determine the tier of this experiment.

Please post on our github issues page or reach out to the Ax user group \
to determine the support tier of your workflow.
"""

NOT_STANDARD_API_MESSAGE = """The experiment summary indicates that this workflow \
is not using a standard API (`uses_standard_api=False`). Tier classification works \
best when the full experiment configuration is known upfront. If you are building a \
tool on top of this function, ensure that `uses_standard_api` is set to `True` in the \
`OptimizationSummary` when your tool uses a standard API.
"""


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
        len(optimization_config.objective.objectives)
        if isinstance(optimization_config.objective, MultiObjective)
        else 1
    )
    num_outcome_constraints = len(optimization_config.outcome_constraints)
    uses_early_stopping = options.early_stopping_strategy is not None
    uses_global_stopping = options.global_stopping_strategy is not None

    # Check if any metrics use merge_multiple_curves
    uses_merge_multiple_curves = False
    all_metrics = list(optimization_config.metrics.values())
    if hasattr(experiment, "tracking_metrics"):
        all_metrics.extend(experiment.tracking_metrics)
    for metric in all_metrics:
        if getattr(metric, "merge_multiple_curves", False):
            uses_merge_multiple_curves = True
            break

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
        all_inputs_are_configs=tier_metadata.get("all_inputs_are_configs", False),
        tolerated_trial_failure_rate=options.tolerated_trial_failure_rate,
        max_pending_trials=options.max_pending_trials,
        min_failed_trials_for_failure_rate_check=(
            options.min_failed_trials_for_failure_rate_check
        ),
    )


def _check_if_is_in_wheelhouse_search_space(
    optimization_summary: OptimizationSummary,
    why_not_is_in_wheelhouse: list[str],
    why_not_supported: list[str],
) -> tuple[bool, bool]:
    """Check if the search space configuration is within supported tiers.

    Evaluates the search space complexity including the number of parameters,
    binary parameters, categorical parameters, and parameter constraints.

    Args:
        optimization_summary: Summary of the experiment. See ``OptimizationSummary``
            for the required and optional keys.
        why_not_is_in_wheelhouse: A list to append reasons for not being in
            the wheelhouse tier.
        why_not_supported: A list to append reasons for not being supported.

    Returns:
        A tuple with information about whether the experiment is in the wheelhouse.
    """
    is_in_wheelhouse, is_supported = True, True

    num_params = optimization_summary.num_params
    if num_params > 50:
        is_in_wheelhouse = False
        why_not_is_in_wheelhouse += [
            f"{num_params} tunable parameter(s) (max in-wheelhouse is 50)"
        ]
        if num_params > 200:
            is_supported = False
            why_not_supported += [
                f"{num_params} tunable parameter(s) (max supported is 200)"
            ]

    num_binary = optimization_summary.num_binary
    if num_binary > 50:
        is_in_wheelhouse = False
        why_not_is_in_wheelhouse += [
            f"{num_binary} binary tunable parameter(s) (max in-wheelhouse is 50)"
        ]
        if num_binary > 100:
            is_supported = False
            why_not_supported += [
                f"{num_binary} binary tunable parameter(s) (max supported is 100)"
            ]

    num_categorical_3_5 = optimization_summary.num_categorical_3_5
    num_categorical_6_inf = optimization_summary.num_categorical_6_inf
    num_categorical_3_inf = num_categorical_3_5 + num_categorical_6_inf
    if num_categorical_3_inf > 0:
        is_in_wheelhouse = False
        why_not_is_in_wheelhouse += [
            f"{num_categorical_3_inf} unordered choice parameter(s) with more "
            "than 3 options (max in-wheelhouse is 0)"
        ]
        if num_categorical_3_5 > 5:
            why_not_supported += [
                f"{num_categorical_3_inf} unordered choice parameters with more "
                "than 3 options (max supported is 5)"
            ]
            is_supported = False
        elif num_categorical_6_inf > 1:
            why_not_supported += [
                f"{num_categorical_6_inf} unordered choice parameters with more "
                "than 5 options (max supported is 1)"
            ]
            is_supported = False

    num_parameter_constraints = optimization_summary.num_parameter_constraints
    if num_parameter_constraints > 2:
        is_in_wheelhouse = False
        why_not_is_in_wheelhouse += [
            f"{num_parameter_constraints} parameter constraints "
            "(max in-wheelhouse is 2)"
        ]
        if num_parameter_constraints > 5:
            is_supported = False
            why_not_supported += [
                f"{num_parameter_constraints} parameter "
                "constraints (max supported is 5)"
            ]
    return is_in_wheelhouse, is_supported


def _check_if_is_in_wheelhouse_optimization_config(
    optimization_summary: OptimizationSummary,
    why_not_is_in_wheelhouse: list[str],
    why_not_supported: list[str],
) -> tuple[bool, bool]:
    """Check if the optimization configuration is within supported tiers.

    Evaluates the optimization config complexity including the number of
    objectives and outcome constraints.

    Args:
        optimization_summary: Summary of the experiment. See ``OptimizationSummary``
            for the required and optional keys.
        why_not_is_in_wheelhouse: A list to append reasons for not being in
            the wheelhouse tier.
        why_not_supported: A list to append reasons for not being supported.

    Returns:
        A tuple with information about whether the experiment is in the wheelhouse.
    """
    is_in_wheelhouse, is_supported = True, True

    num_objectives = optimization_summary.num_objectives
    if num_objectives > 2:
        is_in_wheelhouse = False
        why_not_is_in_wheelhouse += [
            f"{num_objectives} objectives (max in-wheelhouse is 2)"
        ]
        if num_objectives > 4:
            is_supported = False
            why_not_supported += [f"{num_objectives} objectives (max supported is 4)"]

    num_outcome_constraints = optimization_summary.num_outcome_constraints
    if num_outcome_constraints > 2:
        is_in_wheelhouse = False
        why_not_is_in_wheelhouse += [
            f"{num_outcome_constraints} outcome constraints (max in-wheelhouse is 2)"
        ]
        if num_outcome_constraints > 5:
            is_supported = False
            why_not_supported += [
                f"{num_outcome_constraints} outcome constraints (max supported is 5)"
            ]
    return is_in_wheelhouse, is_supported


def _check_if_is_in_wheelhouse_other_settings(
    optimization_summary: OptimizationSummary,
    why_not_is_in_wheelhouse: list[str],
    why_not_supported: list[str],
) -> tuple[bool, bool]:
    """Check if other experiment settings are within supported tiers.

    Evaluates additional settings including max trials, early stopping,
    global stopping, failure rate settings, and advanced options.

    Args:
        optimization_summary: Summary of the experiment. See ``OptimizationSummary``
            for the required and optional keys.
        why_not_is_in_wheelhouse: A list to append reasons for not being in
            the wheelhouse tier.
        why_not_supported: A list to append reasons for not being supported.

    Returns:
        A tuple with information about whether the experiment is in the wheelhouse.
    """
    is_in_wheelhouse, is_supported = True, True
    max_trials = optimization_summary.max_trials
    if not optimization_summary.all_inputs_are_configs:
        is_in_wheelhouse, is_supported = False, False
        why_not_supported.append(NOT_STANDARD_API_MESSAGE)
    elif max_trials is None:
        raise UserInputError("`max_trials` should not be None!")
    elif max_trials is not None and max_trials > 200:
        is_in_wheelhouse = False
        why_not_is_in_wheelhouse += [
            f"{max_trials} total trials (max in-wheelhouse is 200)"
        ]
        if max_trials > 500:
            is_supported = False
            why_not_supported += [f"{max_trials} total trials (max supported is 500)"]

    uses_early_stopping = optimization_summary.uses_early_stopping
    if uses_early_stopping:
        is_in_wheelhouse = False
        why_not_is_in_wheelhouse += ["Early stopping is enabled"]

    uses_global_stopping = optimization_summary.uses_global_stopping
    if uses_global_stopping:
        is_in_wheelhouse = False
        why_not_is_in_wheelhouse += ["Global stopping is enabled"]

    # checking failure rate checking options
    tolerated_trial_failure_rate = optimization_summary.tolerated_trial_failure_rate
    if tolerated_trial_failure_rate is not None and tolerated_trial_failure_rate > 0.9:
        is_in_wheelhouse, is_supported = False, False
        why_not_supported.append(f"{tolerated_trial_failure_rate=} is larger than 0.9.")

    max_pending_trials = optimization_summary.max_pending_trials
    min_failed_trials_for_failure_rate_check = (
        optimization_summary.min_failed_trials_for_failure_rate_check
    )
    if (
        max_pending_trials is not None
        and min_failed_trials_for_failure_rate_check is not None
        and max(2 * max_pending_trials, 5) < min_failed_trials_for_failure_rate_check
    ):
        is_in_wheelhouse, is_supported = False, False
        why_not_supported.append(
            f"{min_failed_trials_for_failure_rate_check=} exceeds "
            f"{max(2 * max_pending_trials, 5)=}. Please reduce "
            "min_failed_trials_for_failure_rate_check below the stated threshold for "
            "this sweep to be in a supported tier."
        )
    non_default_advanced_options = optimization_summary.non_default_advanced_options
    if non_default_advanced_options:
        is_in_wheelhouse, is_supported = False, False
        why_not_supported.append(
            "Non-default advanced_options are set on GenerationStrategyConfig."
        )

    uses_merge_multiple_curves = optimization_summary.uses_merge_multiple_curves
    if uses_merge_multiple_curves:
        is_in_wheelhouse, is_supported = False, False
        why_not_supported.append(
            "Metrics with merge_multiple_curves=True are not supported. "
            "This feature is experimental and caution is advised not to merge "
            "unrelated curves."
        )

    return is_in_wheelhouse, is_supported


def check_if_in_wheelhouse(
    optimization_summary: OptimizationSummary,
) -> tuple[str, list[str] | None, list[str] | None]:
    """Determine the support tier of an experiment based on its configuration.

    Evaluates the experiment summary and classifies it into one of three tiers:

    - **Wheelhouse**: Well-supported configurations that should work without issues.
    - **Advanced**: Technically supported but uses advanced features that may not
      be well-tested or compatible with other advanced features.
    - **Unsupported**: Configurations that push beyond supported limits.

    Args:
        optimization_summary: Summary of the experiment. See ``OptimizationSummary``
            for the required and optional keys. This summary should contain
            information about:

            - Search space: num_params, num_binary, num_categorical_3_5,
              num_categorical_6_inf, num_parameter_constraints
            - Optimization config: num_objectives, num_outcome_constraints
            - Other settings: max_trials, uses_early_stopping, uses_global_stopping,
              all_inputs_are_configs, tolerated_trial_failure_rate, max_pending_trials,
              min_failed_trials_for_failure_rate_check, non_default_advanced_options,
              uses_merge_multiple_curves

    Returns:
        A tuple containing:

        - The tier name: "Wheelhouse", "Advanced", or "Unsupported"
        - A list of reasons for not being in the "Wheelhouse" tier (None if
          in Wheelhouse)
        - A list of reasons for not being supported (None if
          in Wheelhouse or Advanced)
    """

    is_in_wheelhouse, why_not_is_in_wheelhouse = True, []
    is_supported, why_not_supported = True, []

    # Check search space
    search_space_summary = _check_if_is_in_wheelhouse_search_space(
        optimization_summary=optimization_summary,
        why_not_is_in_wheelhouse=why_not_is_in_wheelhouse,
        why_not_supported=why_not_supported,
    )
    is_in_wheelhouse &= search_space_summary[0]
    is_supported &= search_space_summary[1]

    # Check optimization config
    opt_config_summary = _check_if_is_in_wheelhouse_optimization_config(
        optimization_summary=optimization_summary,
        why_not_is_in_wheelhouse=why_not_is_in_wheelhouse,
        why_not_supported=why_not_supported,
    )
    is_in_wheelhouse &= opt_config_summary[0]
    is_supported &= opt_config_summary[1]

    # Check other options
    other_settings_summary = _check_if_is_in_wheelhouse_other_settings(
        optimization_summary=optimization_summary,
        why_not_is_in_wheelhouse=why_not_is_in_wheelhouse,
        why_not_supported=why_not_supported,
    )
    is_in_wheelhouse &= other_settings_summary[0]
    is_supported &= other_settings_summary[1]

    # Return tier and messages
    if is_in_wheelhouse:
        return "Wheelhouse", None, None
    if is_supported:
        return "Advanced", why_not_is_in_wheelhouse, None
    return "Unsupported", why_not_is_in_wheelhouse, why_not_supported


def format_tier_message(
    tier: str,
    why_not_is_in_wheelhouse: Iterable[str] | None,
    why_not_supported: Iterable[str] | None,
) -> str:
    """
    Format the result from `check_if_in_wheelhouse` to a markdown-formatted
    string explaining the tier.

    Takes the tier classification and the reasons for not being in a higher tier,
    and formats them into a readable, markdown-formatted message that can be
    displayed to users.

    Args:
        tier: The tier name ("Wheelhouse", "Advanced", or "Unsupported").
        why_not_is_in_wheelhouse: Reasons for not being in the Wheelhouse tier.
        why_not_supported: Reasons for not being in the Advanced tier.

    Returns:
        A formatted string explaining the tier and the reasons.
    """

    if tier == "Wheelhouse":
        msg = WHEELHOUSE_TIER_MESSAGE
    else:
        if tier == "Advanced":
            msg = ADVANCED_TIER_MESSAGE
        elif tier == "Unsupported":
            msg = UNSUPPORTED_TIER_MESSAGE
        else:
            raise ValueError(f'Got unexpected tier "{tier}".')

        # Provide user-feedback
        if why_not_is_in_wheelhouse:
            why_msg = "\n".join("\t- " + s for s in why_not_is_in_wheelhouse)
            why_msg = (
                "\n\nWhy this experiment is not in the 'Wheelhouse' tier: "
                f"\n{why_msg}\n"
            )
            msg += why_msg
        if why_not_supported:
            why_msg = "\n".join("\t- " + s for s in why_not_supported)
            why_msg = (
                "\n\nWhy this experiment is not in the 'Advanced' tier: "
                f"\n{why_msg}\n"
            )
            msg += why_msg

    msg += (
        "\n\nFor more information about the definition of each tier and what "
        f"level of support you can expect: {WIKI_TIER_MESSAGE}"
    )
    return msg

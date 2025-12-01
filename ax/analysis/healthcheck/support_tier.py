# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, final

import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.healthcheck.healthcheck_analysis import (
    create_healthcheck_analysis_card,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import override


@final
class SupportTierAnalysis(Analysis):
    """
    Healthcheck that evaluates whether the experiment configuration is in the
    "wheelhouse" (fully supported), supported but not wheelhouse, or unsupported.

    This analysis is particularly useful for AxSweep experiments, but can be adapted
    for general Ax experiments as well.

    Status Logic:
    - PASS: Configuration is in wheelhouse (fully supported)
    - WARNING: Configuration is supported but not in wheelhouse (advanced features)
    - FAIL: Configuration is unsupported

    The healthcheck evaluates:
    - Search space complexity (number of parameters, parameter constraints, etc.)
    - Optimization config (number of objectives and outcome constraints)
    - Other settings (early stopping, global stopping, trial limits, etc.)
    """

    def __init__(
        self,
        experiment_summary: dict[str, Any],
        wheelhouse_tier_message: str | None = None,
        advanced_tier_message: str | None = None,
        unsupported_tier_message: str | None = None,
        not_simple_inputs_message: str | None = None,
        wiki_link_message: str | None = None,
    ) -> None:
        """
        Args:
            experiment_summary: Experiment summary dictionary containing all the
                information needed to determine the support tier. Users must provide
                their own summarization logic. Expected keys in the summary dict:
                - max_trials: Maximum number of trials (int | None)
                - num_params: Number of tunable parameters (int)
                - num_binary: Number of binary parameters (int)
                - num_categorical_3_5: Number of categorical parameters with 3-5 options (int)
                - num_categorical_6_inf: Number of categorical parameters with 6+ options (int)
                - num_parameter_constraints: Number of parameter constraints (int)
                - num_objectives: Number of objectives (int)
                - num_outcome_constraints: Number of outcome constraints (int)
                - uses_early_stopping: Whether early stopping is enabled (bool)
                - uses_global_stopping: Whether global stopping is enabled (bool)
                - tolerated_trial_failure_rate: Maximum tolerated trial failure rate (float | None)
                - max_pending_trials: Maximum number of pending trials (int | None)
                - min_failed_trials_for_failure_rate_check: Minimum failed trials
                  before checking failure rate (int | None)
                - all_inputs_are_configs: Whether all inputs use config objects (bool)
                - uses_merge_multiple_curves: Whether merge_multiple_curves is used (bool)
                - non_default_advanced_options: Whether non-default advanced options
                  are set on GenerationStrategyConfig (bool)
            wheelhouse_tier_message: Custom message for Wheelhouse tier.
                If None, uses default generic message.
            advanced_tier_message: Custom message for Advanced tier.
                If None, uses default generic message.
            unsupported_tier_message: Custom message for Unsupported tier.
                If None, uses default generic message.
            not_simple_inputs_message: Custom message when not using simple inputs.
                If None, uses default generic message.
            wiki_link_message: Custom wiki/documentation link to append to messages.
                If None, no link is appended.
        """
        self.experiment_summary = experiment_summary
        self.wheelhouse_tier_message = wheelhouse_tier_message
        self.advanced_tier_message = advanced_tier_message
        self.unsupported_tier_message = unsupported_tier_message
        self.not_simple_inputs_message = not_simple_inputs_message
        self.wiki_link_message = wiki_link_message

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> HealthcheckAnalysisCard:
        # Use the provided experiment summary
        experiment_summary = self.experiment_summary

        # Determine tier
        tier, why_not_wheelhouse, why_not_supported = self._check_tier(
            experiment_summary
        )

        # Build subtitle using custom messages if provided, else defaults
        subtitle = self._format_tier_message(
            tier=tier,
            why_not_wheelhouse=why_not_wheelhouse,
            why_not_supported=why_not_supported,
        )

        # Determine status
        if tier == "Wheelhouse":
            status = HealthcheckStatus.PASS
        elif tier == "Advanced":
            status = HealthcheckStatus.WARNING
        else:  # Unsupported
            status = HealthcheckStatus.FAIL

        # Create dataframe with experiment summary
        df = pd.DataFrame(
            [
                {
                    "Metric": "Support Tier",
                    "Value": tier,
                },
                {
                    "Metric": "Total Parameters",
                    "Value": str(experiment_summary["num_params"]),
                },
                {
                    "Metric": "Objectives",
                    "Value": str(experiment_summary["num_objectives"]),
                },
                {
                    "Metric": "Outcome Constraints",
                    "Value": str(experiment_summary["num_outcome_constraints"]),
                },
                {
                    "Metric": "Parameter Constraints",
                    "Value": str(experiment_summary["num_parameter_constraints"]),
                },
            ]
        )

        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title="Support Tier Healthcheck",
            subtitle=subtitle,
            df=df,
            status=status,
            tier=tier,
        )

    def _check_tier(
        self, experiment_summary: dict[str, Any]
    ) -> tuple[str, list[str], list[str]]:
        """
        Check the support tier of the experiment.

        Returns:
            A tuple containing:
            - The tier string ("Wheelhouse", "Advanced", or "Unsupported")
            - A list of reasons for not being in the Wheelhouse
            - A list of reasons for not being Supported
        """
        is_in_wheelhouse = True
        is_supported = True
        why_not_wheelhouse: list[str] = []
        why_not_supported: list[str] = []

        # ===== Search Space Validation =====
        num_params = experiment_summary["num_params"]
        num_binary = experiment_summary["num_binary"]
        num_categorical_3_5 = experiment_summary["num_categorical_3_5"]
        num_categorical_6_inf = experiment_summary["num_categorical_6_inf"]
        num_parameter_constraints = experiment_summary["num_parameter_constraints"]

        # Total tunable parameters
        if num_params > 50:
            is_in_wheelhouse = False
            why_not_wheelhouse.append(
                f"{num_params} tunable parameter(s) (max in-wheelhouse is 50)"
            )
            if num_params > 200:
                is_supported = False
                why_not_supported.append(
                    f"{num_params} tunable parameter(s) (max supported is 200)"
                )

        # Binary parameters
        if num_binary > 50:
            is_in_wheelhouse = False
            why_not_wheelhouse.append(
                f"{num_binary} binary tunable parameter(s) (max in-wheelhouse is 50)"
            )
            if num_binary > 100:
                is_supported = False
                why_not_supported.append(
                    f"{num_binary} binary tunable parameter(s) (max supported is 100)"
                )

        # Unordered choice parameters
        # Match AxSweep's combined counting logic for wheelhouse message
        num_categorical_3_inf = num_categorical_3_5 + num_categorical_6_inf
        if num_categorical_3_inf > 0:
            is_in_wheelhouse = False
            why_not_wheelhouse.append(
                f"{num_categorical_3_inf} unordered choice parameter(s) with more "
                "than 3 options (max in-wheelhouse is 0)"
            )
            # Check specific thresholds for support tier
            if num_categorical_3_5 > 5:
                is_supported = False
                why_not_supported.append(
                    f"{num_categorical_3_inf} unordered choice parameters with more "
                    "than 3 options (max supported is 5)"
                )
            elif num_categorical_6_inf > 1:
                is_supported = False
                why_not_supported.append(
                    f"{num_categorical_6_inf} unordered choice parameters with more "
                    "than 5 options (max supported is 1)"
                )

        # Parameter constraints
        if num_parameter_constraints > 2:
            is_in_wheelhouse = False
            why_not_wheelhouse.append(
                f"{num_parameter_constraints} parameter constraints "
                "(max in-wheelhouse is 2)"
            )
            if num_parameter_constraints > 5:
                is_supported = False
                why_not_supported.append(
                    f"{num_parameter_constraints} parameter constraints "
                    "(max supported is 5)"
                )

        # ===== Optimization Config Validation =====
        num_objectives = experiment_summary["num_objectives"]
        num_outcome_constraints = experiment_summary["num_outcome_constraints"]

        # Objectives
        if num_objectives > 2:
            is_in_wheelhouse = False
            why_not_wheelhouse.append(
                f"{num_objectives} objectives (max in-wheelhouse is 2)"
            )
            if num_objectives > 4:
                is_supported = False
                why_not_supported.append(
                    f"{num_objectives} objectives (max supported is 4)"
                )

        # Outcome constraints
        if num_outcome_constraints > 2:
            is_in_wheelhouse = False
            why_not_wheelhouse.append(
                f"{num_outcome_constraints} outcome constraints "
                "(max in-wheelhouse is 2)"
            )
            if num_outcome_constraints > 5:
                is_supported = False
                why_not_supported.append(
                    f"{num_outcome_constraints} outcome constraints "
                    "(max supported is 5)"
                )

        # ===== Other Settings Validation =====
        # Check if using simple inputs
        if not experiment_summary["all_inputs_are_configs"]:
            is_in_wheelhouse = False
            is_supported = False
            message = (
                self.not_simple_inputs_message
                if self.not_simple_inputs_message is not None
                else (
                    "Using Ax abstractions (e.g., Experiment, GenerationStrategy) "
                    "as inputs instead of simple config objects"
                )
            )
            why_not_supported.append(message)

        # Check max trials
        max_trials = experiment_summary.get("max_trials")
        if max_trials is not None:
            if max_trials > 200:
                is_in_wheelhouse = False
                why_not_wheelhouse.append(
                    f"{max_trials} total trials (max in-wheelhouse is 200)"
                )
                if max_trials > 500:
                    is_supported = False
                    why_not_supported.append(
                        f"{max_trials} total trials (max supported is 500)"
                    )

        # Check early stopping
        if experiment_summary["uses_early_stopping"]:
            is_in_wheelhouse = False
            why_not_wheelhouse.append("Early stopping is enabled")

        # Check global stopping
        if experiment_summary["uses_global_stopping"]:
            is_in_wheelhouse = False
            why_not_wheelhouse.append("Global stopping is enabled")

        # Check failure rate
        tolerated_trial_failure_rate = experiment_summary.get(
            "tolerated_trial_failure_rate"
        )
        if (
            tolerated_trial_failure_rate is not None
            and tolerated_trial_failure_rate > 0.9
        ):
            is_in_wheelhouse = False
            is_supported = False
            why_not_supported.append(
                f"Tolerated trial failure rate of {tolerated_trial_failure_rate} "
                "is larger than 0.9"
            )

        # Check failure rate check configuration
        max_pending_trials = experiment_summary.get("max_pending_trials")
        min_failed_trials_for_failure_rate_check = experiment_summary.get(
            "min_failed_trials_for_failure_rate_check"
        )
        if (
            max_pending_trials is not None
            and min_failed_trials_for_failure_rate_check is not None
            and max(2 * max_pending_trials, 5)
            < min_failed_trials_for_failure_rate_check
        ):
            is_in_wheelhouse = False
            is_supported = False
            why_not_supported.append(
                f"min_failed_trials_for_failure_rate_check "
                f"({min_failed_trials_for_failure_rate_check}) exceeds "
                f"{max(2 * max_pending_trials, 5)}. Please reduce "
                "min_failed_trials_for_failure_rate_check below the stated threshold."
            )

        # Check advanced options
        if experiment_summary["non_default_advanced_options"]:
            is_in_wheelhouse = False
            is_supported = False
            why_not_supported.append(
                "Non-default advanced_options are set on GenerationStrategyConfig"
            )

        # Check merge_multiple_curves
        if experiment_summary["uses_merge_multiple_curves"]:
            is_in_wheelhouse = False
            is_supported = False
            why_not_supported.append(
                "merge_multiple_curves=True is not supported (experimental feature)"
            )

        # Return tier and messages
        if is_in_wheelhouse:
            return "Wheelhouse", [], []
        if is_supported:
            return "Advanced", why_not_wheelhouse, []
        return "Unsupported", why_not_wheelhouse, why_not_supported

    def _format_tier_message(
        self,
        tier: str,
        why_not_wheelhouse: list[str],
        why_not_supported: list[str],
    ) -> str:
        """
        Format the tier message using custom messages if provided, else defaults.

        Args:
            tier: The support tier ("Wheelhouse", "Advanced", or "Unsupported")
            why_not_wheelhouse: Reasons for not being in Wheelhouse tier
            why_not_supported: Reasons for not being in Advanced tier (unsupported)

        Returns:
            Formatted message string
        """
        # Default messages
        DEFAULT_WHEELHOUSE = (
            "This experiment configuration is in the 'Wheelhouse' tier.\n\n"
            "Experiments in this tier are fully supported and should not "
            "run into any problems. If an issue does occur, please reach "
            "out to the support team."
        )
        DEFAULT_ADVANCED = (
            "This experiment configuration is in the 'Advanced' tier.\n\n"
            "This experiment should technically run, but uses advanced "
            "features that may not be well-tested and/or may not be "
            "compatible with other advanced features."
        )
        DEFAULT_UNSUPPORTED = (
            "This experiment configuration is in the 'Unsupported' tier.\n\n"
            "This configuration pushes beyond supported limits. Please consider "
            "simplifying your experiment configuration."
        )

        # Use custom messages if provided, otherwise use defaults
        if tier == "Wheelhouse":
            msg = self.wheelhouse_tier_message or DEFAULT_WHEELHOUSE
        elif tier == "Advanced":
            msg = self.advanced_tier_message or DEFAULT_ADVANCED
        elif tier == "Unsupported":
            msg = self.unsupported_tier_message or DEFAULT_UNSUPPORTED
        else:
            raise ValueError(f'Got unexpected tier "{tier}".')

        # Add reasons for not being in wheelhouse (for Advanced and Unsupported)
        if why_not_wheelhouse:
            # Check if custom messages are being used (AxSweep style)
            if self.wheelhouse_tier_message is not None:
                # Use AxSweep-style formatting with tabs
                why_msg = "\n".join("\t- " + s for s in why_not_wheelhouse)
                why_msg = (
                    "\n\nWhy this experiment is not in the 'Wheelhouse' tier: "
                    f"\n{why_msg}\n"
                )
            else:
                # Use default markdown formatting
                why_msg = "\n\n**Why not in 'Wheelhouse' tier:**\n"
                why_msg += "".join(f"- {reason}\n" for reason in why_not_wheelhouse)
            msg += why_msg

        # Add reasons for not being supported (for Unsupported only)
        if why_not_supported:
            # Check if custom messages are being used (AxSweep style)
            if self.advanced_tier_message is not None:
                # Use AxSweep-style formatting with tabs
                why_msg = "\n".join("\t- " + s for s in why_not_supported)
                why_msg = (
                    "\n\nWhy this experiment is not in the 'Advanced' tier: "
                    f"\n{why_msg}\n"
                )
            else:
                # Use default markdown formatting
                why_msg = "\n\n**Why not in 'Advanced' tier:**\n"
                why_msg += "".join(f"- {reason}\n" for reason in why_not_supported)
            msg += why_msg

        # Add wiki link if provided
        if self.wiki_link_message:
            msg += (
                "\n\nFor more information about the definition of each tier and what "
                f"level of support you can expect: {self.wiki_link_message}"
            )

        return msg

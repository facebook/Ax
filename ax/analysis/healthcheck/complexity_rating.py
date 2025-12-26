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
from ax.analysis.utils import filter_none
from ax.core.experiment import Experiment
from ax.early_stopping.strategies import BaseEarlyStoppingStrategy
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy
from ax.utils.common.complexity_utils import (
    check_if_in_standard,
    DEFAULT_TIER_MESSAGES,
    format_tier_message,
    summarize_ax_optimization_complexity,
    TierMessages,
)
from pyre_extensions import none_throws, override


@final
class ComplexityRatingAnalysis(Analysis):
    """
    Healthcheck that evaluates whether the experiment configuration is in the
    "standard" (fully tested and supported), "advanced" (technically supported
    but uses features that may not be well-tested or compatible with other advanced
    features), or unsupported tier.

    Status Logic:
    - PASS: Configuration is in standard
    - WARNING: Configuration is advanced
    - FAIL: Configuration is unsupported

    The healthcheck evaluates:
    - Search space complexity (number of parameters, parameter constraints, etc.)
    - Optimization config (number of objectives and outcome constraints)
    - Other settings (early stopping, global stopping, trial limits, etc.)
    """

    def __init__(
        self,
        tier_metadata: dict[str, Any] | None = None,
        tier_messages: TierMessages = DEFAULT_TIER_MESSAGES,
        early_stopping_strategy: BaseEarlyStoppingStrategy | None = None,
        global_stopping_strategy: BaseGlobalStoppingStrategy | None = None,
        tolerated_trial_failure_rate: float | None = None,
        max_pending_trials: int | None = None,
        min_failed_trials_for_failure_rate_check: int | None = None,
    ) -> None:
        """Initialize the ComplexityRatingAnalysis.

        Args:
            tier_metadata: Additional tier-related metadata from the orchestrator.
                Supported keys:
                - 'user_supplied_max_trials': Maximum number of trials.
                - 'uses_standard_api': Whether high-level configs are used (as
                    opposed to low-level Ax abstractions), ensuring the full
                    experiment configuration is known upfront.
            tier_messages: Custom tier-specific messages used when formatting
                the tier result. Defaults to DEFAULT_TIER_MESSAGES which contains
                generic messages suitable for most users. Pass a custom TierMessages
                instance to provide tool-specific descriptions, support SLAs,
                links to docs, or contact information.
            early_stopping_strategy: The early stopping strategy, if any. Used to
                determine if early stopping is enabled. Defaults to None.
            global_stopping_strategy: The global stopping strategy, if any. Used to
                determine if global stopping is enabled. Defaults to None.
            tolerated_trial_failure_rate: Fraction of trials allowed to fail without
                the whole optimization ending. Default value used is 0.5.
            max_pending_trials: Maximum number of pending trials. Default used is 10.
            min_failed_trials_for_failure_rate_check: Minimum failed trials before
                failure rate is checked. Default value used is 5.
        """
        self.tier_metadata: dict[str, Any] = (
            tier_metadata if tier_metadata is not None else {}
        )
        self.tier_messages = tier_messages
        self.early_stopping_strategy = early_stopping_strategy
        self.global_stopping_strategy = global_stopping_strategy
        self.tolerated_trial_failure_rate = tolerated_trial_failure_rate
        self.max_pending_trials = max_pending_trials
        self.min_failed_trials_for_failure_rate_check = (
            min_failed_trials_for_failure_rate_check
        )

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        if experiment is None:
            return "Experiment is required for ComplexityRatingAnalysis."
        return None

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> HealthcheckAnalysisCard:
        """Compute the complexity rating healthcheck for an experiment.

        This method analyzes the experiment's configuration and classifies it
        into one of three tiers: Standard (fully supported), Advanced (supported
        but uses advanced features), or Unsupported (exceeds supported limits).

        Note:
            This method assumes ``validate_applicable_state`` has been called
            and returned None, ensuring ``experiment`` is not None.

        Args:
            experiment: The Ax Experiment to analyze. Must not be None.
            generation_strategy: The generation strategy (unused but required
                by the Analysis interface).
            adapter: The adapter (unused but required by the Analysis interface).

        Returns:
            A ``HealthcheckAnalysisCard`` containing the tier classification,
            a formatted message explaining the tier, and a summary DataFrame
            with key experiment metrics.
        """
        experiment = none_throws(experiment)
        optimization_summary = summarize_ax_optimization_complexity(
            experiment=experiment,
            tier_metadata=self.tier_metadata,
            early_stopping_strategy=self.early_stopping_strategy,
            global_stopping_strategy=self.global_stopping_strategy,
            **filter_none(
                tolerated_trial_failure_rate=self.tolerated_trial_failure_rate,
                max_pending_trials=self.max_pending_trials,
                min_failed_trials_for_failure_rate_check=(
                    self.min_failed_trials_for_failure_rate_check
                ),
            ),
        )

        # Determine tier
        tier, why_not_standard, why_not_supported = check_if_in_standard(
            optimization_summary
        )

        # Build subtitle using custom messages if provided, else defaults
        subtitle = format_tier_message(
            tier=tier,
            why_not_is_in_standard=why_not_standard,
            why_not_supported=why_not_supported,
            tier_messages=self.tier_messages,
        )

        # Determine status
        if tier == "Standard":
            status = HealthcheckStatus.PASS
        elif tier == "Advanced":
            status = HealthcheckStatus.WARNING
        else:  # Unsupported
            status = HealthcheckStatus.FAIL

        # Create dataframe with experiment summary
        df = pd.DataFrame(
            [
                {
                    "Metric": "Optimization Complexity Rating",
                    "Value": tier,
                },
                {
                    "Metric": "Total Parameters",
                    "Value": str(optimization_summary.num_params),
                },
                {
                    "Metric": "Objectives",
                    "Value": str(optimization_summary.num_objectives),
                },
                {
                    "Metric": "Outcome Constraints",
                    "Value": str(optimization_summary.num_outcome_constraints),
                },
                {
                    "Metric": "Parameter Constraints",
                    "Value": str(optimization_summary.num_parameter_constraints),
                },
            ]
        )

        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title="Complexity Rating Healthcheck",
            subtitle=subtitle,
            df=df,
            status=status,
            tier=tier,
        )

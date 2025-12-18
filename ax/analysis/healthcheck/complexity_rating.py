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
from ax.service.orchestrator import OrchestratorOptions
from ax.utils.common.complexity_utils import (
    check_if_in_standard,
    DEFAULT_TIER_MESSAGES,
    format_tier_message,
    summarize_ax_optimization_complexity,
    TierMessages,
)
from pyre_extensions import none_throws, override

# TODO: Enable ComplexityRatingAnalysis in OverviewAnalysis. Currently, this
# analysis depends on OrchestratorOptions to evaluate early stopping, global
# stopping, and other orchestrator-level settings. To enable it in
# OverviewAnalysis, we need to either:
# 1. Extract the relevant settings from OrchestratorOptions into a smaller,
#     analysis-specific data structure that can be passed independently, OR
# 2. Modify summarize_ax_optimization_complexity to make the options parameter
#     optional and handle missing orchestrator settings gracefully (e.g., skip
#     those checks or use defaults).
# For now, this analysis can be used directly other
# orchestrator-aware callers that have access to OrchestratorOptions.


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
        options: OrchestratorOptions | None = None,
        tier_metadata: dict[str, Any] | None = None,
        tier_messages: TierMessages = DEFAULT_TIER_MESSAGES,
    ) -> None:
        """Initialize the ComplexityRatingAnalysis.

        Args:
            options: The orchestrator options used for the optimization.
                Required to evaluate early stopping, global stopping, and
                failure rate settings.
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
        """
        self.options = options
        self.tier_metadata: dict[str, Any] = (
            tier_metadata if tier_metadata is not None else {}
        )
        self.tier_messages = tier_messages

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        if experiment is None:
            return "Experiment is required for ComplexityRatingAnalysis."
        if self.options is None:
            return (
                "OrchestratorOptions is required for ComplexityRatingAnalysis. "
                "Please pass options to the constructor."
            )
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
            and returned None, ensuring ``experiment`` and ``self.options``
            are not None.

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
        options = none_throws(self.options)
        optimization_summary = summarize_ax_optimization_complexity(
            experiment=experiment,
            options=options,
            tier_metadata=self.tier_metadata,
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

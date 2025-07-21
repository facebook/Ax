# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis, ErrorAnalysisCard
from ax.analysis.analysis_card import AnalysisCardGroup
from ax.analysis.diagnostics import DiagnosticAnalysis
from ax.analysis.healthcheck.can_generate_candidates import (
    CanGenerateCandidatesAnalysis,
)
from ax.analysis.healthcheck.constraints_feasibility import (
    ConstraintsFeasibilityAnalysis,
)
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckAnalysisCard
from ax.analysis.healthcheck.metric_fetching_errors import MetricFetchingErrorsAnalysis
from ax.analysis.healthcheck.search_space_analysis import SearchSpaceAnalysis
from ax.analysis.healthcheck.should_generate_candidates import ShouldGenerateCandidates
from ax.analysis.insights import InsightsAnalysis
from ax.analysis.results import ResultsAnalysis
from ax.analysis.trials import AllTrialsAnalysis
from ax.core.experiment import Experiment
from ax.core.trial_status import TrialStatus
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import none_throws, override


HEALTH_CHECK_CARDGROUP_TITLE = "Health Checks"
HEALTH_CHECK_CARDGROUP_SUBTITLE = (
    "Comprehensive health checks designed to identify potential issues in the Ax "
    "experiment. These checks cover areas such as metric fetching, search space "
    "configuration, and candidate generation, with the aim of flagging areas where "
    "user intervention may be necessary to ensure the experiment's robustness "
    "and success."
)

OVERVIEW_CARDGROUP_TITLE = "Overview of the Entire Optimization Process "
OVERVIEW_CARDGROUP_SUBTITLE = (
    "This analysis provides an overview of the entire optimization process. "
    "It includes visualizations of the results obtained so far, insights into "
    "the parameter and metric relationships learned by the Ax model, diagnostics "
    "such as model fit, and health checks to assess the overall health of the "
    "experiment."
)


class OverviewAnalysis(Analysis):
    """
    Top-level Analysis that provides an overview of the entire optimization process,
    including results, insights, and diagnostics. OverviewAnalysis examines the
    Experiment and GenerationStrategy's configuration and their respective current
    states to heuristically determine which Analyses to compute under the hood.

    AnalysisCards will be returned in the following groups:
        * Overview
            * Results
                * Pairs of Modeled and Raw ArmEffectsPlots for objectives and
                    constraints
                * Modeled ScatterPlots for objectives versus objectives and objectives
                    versus constraints
                * ParallelCoordinatesPlot for objectives
                * Summary
            * Insights
                * Sensitivity Plots
                * Slice Plots
                * Contour Plots
            * Diagnostic
                * CrossValidationPlots
            * Health Checks
                * MetricFetchingErrorsAnalysis
                * CanGenerateCandidatesAnalysis
                * ConstraintsFeasibilityAnalysis
                * SearchSpaceAnalysis
                * ShouldGenerateCandidates
            * Trial-Level Analyses
                * Trial 0
                    * ArmEffectsPlot
                ...
    """

    def __init__(
        self,
        can_generate: bool | None = None,
        can_generate_reason: str | None = None,
        can_generate_days_till_fail: int | None = None,
        should_generate: bool | None = None,
        should_generate_reason: str | None = None,
    ) -> None:
        super().__init__()
        self.can_generate = can_generate
        self.can_generate_reason = can_generate_reason
        self.can_generate_days_till_fail = can_generate_days_till_fail
        self.should_generate = should_generate
        self.should_generate_reason = should_generate_reason

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardGroup:
        # Compute the arm effects plots, scatter plots, etc.
        results_group = ResultsAnalysis().compute_or_error_card(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        # Compute the sensitivity plots, slice plots, contour plots, etc.
        insights_group = InsightsAnalysis().compute_or_error_card(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        # Compute the diagnostics section (cross validation plots)
        diagnostics_group = DiagnosticAnalysis().compute_or_error_card(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        health_check_analyses = [
            MetricFetchingErrorsAnalysis(),
            CanGenerateCandidatesAnalysis(
                can_generate_candidates=self.can_generate,
                reason=self.can_generate_reason,
                days_till_fail=self.can_generate_days_till_fail,
            )
            if self.can_generate is not None
            and self.can_generate_reason is not None
            and self.can_generate_days_till_fail is not None
            else None,
            ConstraintsFeasibilityAnalysis(),
            *[
                SearchSpaceAnalysis(trial_index=trial.index)
                for trial in none_throws(experiment).trials_by_status[
                    TrialStatus.CANDIDATE
                ]
            ],
            *[
                ShouldGenerateCandidates(
                    should_generate=self.should_generate,
                    reason=self.should_generate_reason,
                    trial_index=trial.index,
                )
                for trial in none_throws(experiment).trials_by_status[
                    TrialStatus.CANDIDATE
                ]
                if self.should_generate is not None
                and self.should_generate_reason is not None
            ],
        ]

        health_check_cards = [
            analyis.compute_or_error_card(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )
            for analyis in health_check_analyses
            if analyis is not None
        ]

        non_passing_health_checks = [
            card
            for card in health_check_cards
            if (isinstance(card, HealthcheckAnalysisCard) and not card.is_passing())
            or isinstance(card, ErrorAnalysisCard)
        ]

        health_checks_group = (
            AnalysisCardGroup(
                name="HealthchecksAnalysis",
                title=HEALTH_CHECK_CARDGROUP_TITLE,
                subtitle=HEALTH_CHECK_CARDGROUP_SUBTITLE,
                children=non_passing_health_checks,
            )
            if len(non_passing_health_checks) > 0
            else None
        )

        trials_group = AllTrialsAnalysis().compute_or_error_card(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        return self._create_analysis_card_group(
            title=OVERVIEW_CARDGROUP_TITLE,
            subtitle=OVERVIEW_CARDGROUP_SUBTITLE,
            children=[
                group
                for group in [
                    results_group,
                    insights_group,
                    diagnostics_group,
                    health_checks_group,
                    trials_group,
                ]
                if group is not None
            ],
        )

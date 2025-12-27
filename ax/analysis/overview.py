# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, final

from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis, ErrorAnalysisCard
from ax.analysis.diagnostics import DiagnosticAnalysis
from ax.analysis.healthcheck.baseline_improvement import BaselineImprovementAnalysis
from ax.analysis.healthcheck.can_generate_candidates import (
    CanGenerateCandidatesAnalysis,
)
from ax.analysis.healthcheck.complexity_rating import ComplexityRatingAnalysis
from ax.analysis.healthcheck.constraints_feasibility import (
    ConstraintsFeasibilityAnalysis,
)
from ax.analysis.healthcheck.early_stopping_healthcheck import EarlyStoppingAnalysis
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckAnalysisCard
from ax.analysis.healthcheck.metric_fetching_errors import MetricFetchingErrorsAnalysis
from ax.analysis.healthcheck.predictable_metrics import PredictableMetricsAnalysis
from ax.analysis.healthcheck.search_space_analysis import SearchSpaceAnalysis
from ax.analysis.healthcheck.should_generate_candidates import ShouldGenerateCandidates
from ax.analysis.insights import InsightsAnalysis
from ax.analysis.results import ResultsAnalysis
from ax.analysis.trials import AllTrialsAnalysis
from ax.analysis.utils import validate_experiment
from ax.core.analysis_card import AnalysisCardGroup
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.orchestrator import OrchestratorOptions
from pyre_extensions import override


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


@final
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
                * EarlyStoppingAnalysis
                * CanGenerateCandidatesAnalysis
                * ConstraintsFeasibilityAnalysis
                * SearchSpaceAnalysis
                * ShouldGenerateCandidates
                * ComplexityRatingAnalysis
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
        options: OrchestratorOptions | None = None,
        tier_metadata: dict[str, Any] | None = None,
        model_fit_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self.can_generate = can_generate
        self.can_generate_reason = can_generate_reason
        self.can_generate_days_till_fail = can_generate_days_till_fail
        self.should_generate = should_generate
        self.should_generate_reason = should_generate_reason
        self.options = options
        self.tier_metadata = tier_metadata
        self.model_fit_threshold = model_fit_threshold

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        return validate_experiment(
            experiment=experiment,
            require_trials=False,
            require_data=False,
        )

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

        if experiment is None:
            raise UserInputError(
                "OverviewAnalysis requires a non-null experiment to compute candidate "
                "trials. Please provide an experiment."
            )

        candidate_trials = experiment.extract_relevant_trials(
            trial_statuses=[TrialStatus.CANDIDATE]
        )

        # Check if the experiment has MapData and MapMetrics (required for
        # early stopping)
        has_map_data = isinstance(experiment.lookup_data(), MapData)
        has_map_metrics = any(
            isinstance(m, MapMetric) for m in experiment.metrics.values()
        )

        # Check if the experiment has BatchTrials
        has_batch_trials = any(
            isinstance(trial, BatchTrial) for trial in experiment.trials.values()
        )

        health_check_analyses = [
            MetricFetchingErrorsAnalysis(),
            (
                EarlyStoppingAnalysis(
                    early_stopping_strategy=(
                        self.options.early_stopping_strategy if self.options else None
                    ),
                )
                if has_map_data and has_map_metrics
                else None
            ),
            CanGenerateCandidatesAnalysis(
                can_generate_candidates=self.can_generate,
                reason=self.can_generate_reason,
                days_till_fail=self.can_generate_days_till_fail,
            )
            if self.can_generate is not None
            and self.can_generate_reason is not None
            and self.can_generate_days_till_fail is not None
            else None,
            ComplexityRatingAnalysis(
                options=self.options,
                tier_metadata=self.tier_metadata,
            )
            if self.options is not None
            else None,
            ConstraintsFeasibilityAnalysis(),
            PredictableMetricsAnalysis()
            if self.model_fit_threshold is None
            else PredictableMetricsAnalysis(
                model_fit_threshold=self.model_fit_threshold
            ),
            BaselineImprovementAnalysis() if not has_batch_trials else None,
            *[
                SearchSpaceAnalysis(trial_index=trial.index)
                for trial in candidate_trials
            ],
            *[
                ShouldGenerateCandidates(
                    should_generate=self.should_generate,
                    reason=self.should_generate_reason,
                    trial_index=trial.index,
                )
                for trial in candidate_trials
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

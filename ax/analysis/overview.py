# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.analysis_card import AnalysisCardGroup
from ax.analysis.diagnostics import DiagnosticAnalysis
from ax.analysis.insights import InsightsAnalysis
from ax.analysis.results import ResultsAnalysis
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import override


class OverviewAnalysis(Analysis):
    """
    Top-level Analysis that provides an overview of the entire optimization process,
    including results, insights, and diagnostics. OverviewAnalysis examines the
    Experiment and GenerationStrategy's configuration and their respective current
    states to heuristicly determine which Analyses to compute under the hood.

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
            * Healthchecks
                * TODO
            * Trial-level information
                * TODO
    """

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

        return self._create_analysis_card_group(
            children=[results_group, insights_group, diagnostics_group]
        )

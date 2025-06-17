# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-safe

from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.analysis_card import (
    AnalysisCardBase,
    AnalysisCardGroup,
    ErrorAnalysisCard,
)
from ax.analysis.plotly.top_surfaces import TopSurfacesAnalysis
from ax.core.experiment import Experiment
from ax.exceptions.core import DataRequiredError, UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import assert_is_instance, override


class InsightsAnalysis(Analysis):
    """
    An Analysis that provides insights into the optimization process.

    For continuous and mixed seach spaces, this includes sensitivity plots,
    slice plots, and contour plots.

    For bandit experiments, this includes a bandit rollout plot.
    """

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardGroup:
        if experiment is None:
            raise UserInputError("InsightsAnalysis requires an Experiment.")

        if experiment.lookup_data().df.empty:
            raise DataRequiredError(
                "Cannot compute InsightsAnalysis, Experiment has no data."
            )

        # If the Experiment has an OptimizationConfig set, extract the objective and
        # constraint names.
        objective_names = []
        constraint_names = []
        if (optimization_config := experiment.optimization_config) is not None:
            objective_names = optimization_config.objective.metric_names
            constraint_names = [
                constraint.metric.name
                for constraint in optimization_config.outcome_constraints
            ]

        sensitivity_plots: list[AnalysisCardBase] = []
        slice_plots: list[AnalysisCardBase] = []
        contour_plots: list[AnalysisCardBase] = []

        # For each objective and constraint, compute a sensitivity analysis and plot
        # the top 3 surfaces. Collect the bar (sensitivity) plots, slice plots, and
        # contour plots in separate lists.
        for metric_name in [*objective_names, *constraint_names]:
            maybe_top_surfaces_group = TopSurfacesAnalysis(
                metric_name=metric_name,
                top_k=3,
            ).compute_or_error_card(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )

            if isinstance(maybe_top_surfaces_group, ErrorAnalysisCard):
                continue

            top_surfaces_group = assert_is_instance(
                maybe_top_surfaces_group, AnalysisCardGroup
            )

            # Add the top surfaces plots if they were computed
            sensitivity_plots.extend(top_surfaces_group.children[:1])
            slice_plots.extend(top_surfaces_group.children[1:2])
            contour_plots.extend(top_surfaces_group.children[2:3])

        groups = [
            AnalysisCardGroup(name="Sensitivity Plots", children=sensitivity_plots)
            if len(sensitivity_plots) > 0
            else None,
            AnalysisCardGroup(name="Slice Plots", children=slice_plots)
            if len(slice_plots) > 0
            else None,
            AnalysisCardGroup(name="Contour Plots", children=contour_plots)
            if len(contour_plots) > 0
            else None,
        ]

        return self._create_analysis_card_group(
            children=[group for group in groups if group is not None]
        )

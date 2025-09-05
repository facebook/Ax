# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.analysis_card import AnalysisCardGroup
from ax.analysis.plotly.marginal_effects import MarginalEffectsPlot
from ax.analysis.plotly.top_surfaces import TopSurfacesAnalysis
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.utils import is_bandit_experiment
from ax.exceptions.core import DataRequiredError, UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import override


INSIGHTS_CARDGROUP_TITLE = "Insights Analysis"

INSIGHTS_CARDGROUP_SUBTITLE = (
    "Insight Analyses display information to help understand the underlying "
    "experiment i.e parameter and metric relationships learned by the Ax model."
    "Use this information to better understand your experiment space and users."
)


class InsightsAnalysis(Analysis):
    """
    An Analysis that provides insights into the optimization process.

    For continuous and mixed search spaces, this includes sensitivity plots,
    slice plots, and contour plots.

    For bandit experiments, it returns a bandit rollout plot instead.
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

        # Relativize the effects if the status quo is set and there are BatchTrials
        # present.
        relativize = experiment.status_quo is not None and any(
            isinstance(trial, BatchTrial) for trial in experiment.trials.values()
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

        is_bandit = generation_strategy and is_bandit_experiment(
            generation_strategy_name=generation_strategy.name
        )

        # For each objective and constraint, compute a sensitivity analysis and plot
        # the top 3 surfaces. Collect the bar (sensitivity) plots, slice plots, and
        # contour plots in separate lists.
        top_surfaces_groups = (
            [
                TopSurfacesAnalysis(
                    metric_name=metric_name,
                    top_k=3,
                    relativize=relativize,
                ).compute_or_error_card(
                    experiment=experiment,
                    generation_strategy=generation_strategy,
                    adapter=adapter,
                )
                for metric_name in [*objective_names, *constraint_names]
            ]
            if not is_bandit
            else []
        )

        # Add MarginalEffectsPlot for bandit experiments
        marginal_effects_groups = (
            [
                MarginalEffectsPlot(metric_name=metric_name).compute_or_error_card(
                    experiment=experiment,
                    generation_strategy=generation_strategy,
                    adapter=adapter,
                )
                for metric_name in [*objective_names, *constraint_names]
            ]
            if is_bandit
            else []
        )

        return self._create_analysis_card_group(
            title=INSIGHTS_CARDGROUP_TITLE,
            subtitle=INSIGHTS_CARDGROUP_SUBTITLE,
            children=[*top_surfaces_groups, *marginal_effects_groups],
        )

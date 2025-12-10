# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.plotly.marginal_effects import MarginalEffectsPlot
from ax.analysis.plotly.objective_p_feasible_frontier import (
    ObjectivePFeasibleFrontierPlot,
)
from ax.analysis.plotly.p_feasible import PFeasiblePlot
from ax.analysis.plotly.top_surfaces import TopSurfacesAnalysis
from ax.analysis.utils import validate_experiment
from ax.core.analysis_card import AnalysisCardBase, AnalysisCardGroup
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.utils import is_bandit_experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import none_throws, override


INSIGHTS_CARDGROUP_TITLE = "Insights Analysis"

INSIGHTS_CARDGROUP_SUBTITLE = (
    "Insight Analyses display information to help understand the underlying "
    "experiment i.e parameter and metric relationships learned by the Ax model."
    "Use this information to better understand your experiment space and users."
)


@final
class InsightsAnalysis(Analysis):
    """
    An Analysis that provides insights into the optimization process.

    For continuous and mixed search spaces, this includes sensitivity plots,
    slice plots, and contour plots.

    For bandit experiments, it returns a bandit rollout plot instead.
    """

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        experiment_validation_str = validate_experiment(
            experiment=experiment,
            require_trials=False,
            require_data=False,
        )
        if experiment_validation_str is not None:
            return experiment_validation_str

        experiment = none_throws(experiment)
        if experiment.optimization_config is None:
            return "Experiment must have an OptimizationConfig to compute insights."

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardGroup:
        cards: list[AnalysisCardBase] = []

        experiment = none_throws(experiment)
        optimization_config = none_throws(experiment.optimization_config)

        objective_names = optimization_config.objective.metric_names
        constraint_names = [
            constraint.metric.name
            for constraint in optimization_config.outcome_constraints
        ]
        # Relativize the effects if the status quo is set and there are BatchTrials
        # present.
        relativize = experiment.status_quo is not None and any(
            isinstance(trial, BatchTrial) for trial in experiment.trials.values()
        )

        # If there are any constraints, compute the probability of feasiblity related
        # analyses.
        if len(objective_names) == 1 and len(constraint_names) > 0:
            outcome_constraints_card = OutcomeConstraintsAnalysis(
                relativize=relativize
            ).compute_or_error_card(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )
            cards.append(outcome_constraints_card)

        # Add MarginalEffectsPlot for bandit experiments
        if generation_strategy and is_bandit_experiment(
            generation_strategy_name=generation_strategy.name
        ):
            marginal_effects_groups = [
                MarginalEffectsPlot(metric_name=metric_name).compute_or_error_card(
                    experiment=experiment,
                    generation_strategy=generation_strategy,
                    adapter=adapter,
                )
                for metric_name in [*objective_names, *constraint_names]
            ]

            cards.extend(marginal_effects_groups)

        # For non-bandit experiments, for each objective and constraint, compute a
        # sensitivity analysis and plot the top 3 surfaces.
        else:
            top_surfaces_groups = [
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

            cards.extend(top_surfaces_groups)

        return self._create_analysis_card_group(
            title=INSIGHTS_CARDGROUP_TITLE,
            subtitle=INSIGHTS_CARDGROUP_SUBTITLE,
            children=cards,
        )


class OutcomeConstraintsAnalysis(Analysis):
    def __init__(self, relativize: bool = False) -> None:
        self.relativize = relativize

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        experiment_validation_str = validate_experiment(
            experiment=experiment,
            require_trials=True,
            require_data=True,
        )
        if experiment_validation_str is not None:
            return experiment_validation_str

        experiment = none_throws(experiment)
        if experiment.optimization_config is None:
            return "Experiment must have an OptimizationConfig."

        optimization_config = none_throws(experiment.optimization_config)
        if len(optimization_config.objective.metric_names) > 1:
            return "Experiment may not have more than one Objective."

        if len(optimization_config.outcome_constraints) == 0:
            return "Experiment must have at least one OutcomeConstraint."

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardGroup:
        p_feasible_card = PFeasiblePlot(
            use_model_predictions=True,
        ).compute_or_error_card(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        objective_p_feasible_frontier_card = ObjectivePFeasibleFrontierPlot(
            relativize=self.relativize
        ).compute_or_error_card(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        return self._create_analysis_card_group(
            title="Outcome Constraints Analysis",
            subtitle=(
                "Understand which trials are likely to meet outcome constraints, and "
                "show how outcome constraints are affecting the optimization as a "
                "whole."
            ),
            children=[p_feasible_card, objective_p_feasible_frontier_card],
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
from typing import final, Sequence

from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.best_trials import BestTrials
from ax.analysis.plotly.arm_effects import ArmEffectsPlot
from ax.analysis.plotly.bandit_rollout import BanditRollout
from ax.analysis.plotly.scatter import (
    SCATTER_CARDGROUP_SUBTITLE,
    SCATTER_CARDGROUP_TITLE,
    ScatterPlot,
)
from ax.analysis.plotly.utility_progression import UtilityProgressionAnalysis
from ax.analysis.summary import Summary
from ax.analysis.utils import extract_relevant_adapter, validate_experiment
from ax.core.analysis_card import AnalysisCardGroup
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.outcome_constraint import ScalarizedOutcomeConstraint
from ax.core.trial_status import TrialStatus
from ax.core.utils import is_bandit_experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import none_throws, override

RESULTS_CARDGROUP_TITLE = "Results Analysis"

RESULTS_CARDGROUP_SUBTITLE = (
    "Result Analyses provide a high-level overview of the results of the optimization "
    "process so far with respect to the metrics specified in experiment design."
)

ARM_EFFECTS_PAIR_CARDGROUP_TITLE = (
    "Metric Effects: Predicted and observed effects for all arms in the experiment"
)
ARM_EFFECTS_PAIR_CARDGROUP_SUBTITLE = (
    "These pair of plots visualize the metric effects for each arm, with the Ax "
    "model predictions on the left and the raw observed data on the right. The "
    "predicted effects apply shrinkage for noise and adjust for non-stationarity "
    "in the data, so they are more representative of the reproducible effects that "
    "will manifest in a long-term validation experiment. "
)


@final
class ResultsAnalysis(Analysis):
    """
    An Analysis that provides a high-level overview of the results of the optimization
    process so far, e.g. effects on all arms. It produces an analysis card group.
    """

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        return validate_experiment(
            experiment=experiment,
            require_trials=True,
            require_data=True,
        )

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardGroup:
        experiment = none_throws(experiment)

        # If the Experiment has an OptimizationConfig set, extract the objective and
        # constraint names.
        objective_names = []
        constraint_names = []
        if (optimization_config := experiment.optimization_config) is not None:
            objective_names = optimization_config.objective.metric_names
            for oc in optimization_config.outcome_constraints:
                if isinstance(oc, ScalarizedOutcomeConstraint):
                    constraint_names.extend([m.name for m in oc.metrics])
                else:
                    constraint_names.append(oc.metric.name)

        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        # Check if there are BatchTrials present.
        has_batch_trials = any(
            isinstance(trial, BatchTrial) for trial in experiment.trials.values()
        )
        # Relativize the effects if the status quo is set and there are BatchTrials
        # present.
        relativize = experiment.status_quo is not None and has_batch_trials
        # Compute both observed and modeled effects for each objective and constraint.
        arm_effect_pair_group = (
            ArmEffectsPair(
                metric_names=[*objective_names, *constraint_names],
                relativize=relativize,
            ).compute_or_error_card(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=relevant_adapter,
            )
            if len(objective_names) > 0
            else None
        )

        # If there are multiple objectives, compute scatter plots of each combination
        # of two objectives. For MOO experiments, show the Pareto frontier line.
        objective_scatter_group = (
            AnalysisCardGroup(
                name="Objective Scatter Plots",
                title=SCATTER_CARDGROUP_TITLE + " (Objectives)",
                subtitle=SCATTER_CARDGROUP_SUBTITLE,
                children=[
                    ScatterPlot(
                        x_metric_name=x,
                        y_metric_name=y,
                        relativize=relativize,
                        show_pareto_frontier=True,
                    ).compute_or_error_card(
                        experiment=experiment,
                        generation_strategy=generation_strategy,
                        adapter=relevant_adapter,
                    )
                    for x, y in itertools.combinations(objective_names, 2)
                ],
            )
            if len(objective_names) > 1
            else None
        )

        # If there are objectives and constraints, compute scatter plots of each
        # objective versus each constraint. Objectives are always plotted on the x-
        # axis and constraints on the y-axis.
        constraint_scatter_group = (
            AnalysisCardGroup(
                name="Constraint Scatter Plots",
                title=SCATTER_CARDGROUP_TITLE + " (Constraints)",
                subtitle=SCATTER_CARDGROUP_SUBTITLE,
                children=[
                    ScatterPlot(
                        x_metric_name=objective_name,
                        y_metric_name=constraint_name,
                        relativize=relativize,
                    ).compute_or_error_card(
                        experiment=experiment,
                        generation_strategy=generation_strategy,
                        adapter=relevant_adapter,
                    )
                    for objective_name in objective_names
                    for constraint_name in constraint_names
                ],
            )
            if len(objective_names) > 0 and len(constraint_names) > 0
            else None
        )

        # Produce a parallel coordinates plot for each objective.
        # TODO: mpolson mgarrard bring back parallel coordinates after fixing
        # objective_parallel_coordinates_group = (
        #     AnalysisCardGroup(
        #         name="Objective Parallel Coordinates Plots",
        #         children=[
        #             ParallelCoordinatesPlot(
        #                 metric_name=metric_name
        #             ).compute_or_error_card(
        #                 experiment=experiment,
        #                 generation_strategy=generation_strategy,
        #                 adapter=adapter,
        #             )
        #             for metric_name in objective_names
        #         ],
        #     )
        #     if len(objective_names) > 0
        #     else None
        # )

        # Add BanditRollout for experiments with specific generation strategy
        bandit_rollout_card = (
            BanditRollout().compute_or_error_card(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )
            if generation_strategy
            and is_bandit_experiment(generation_strategy_name=generation_strategy.name)
            else None
        )

        # Compute best trials, skip for experiments with ScalarizedOutcomeConstraints or
        # BatchTrials as it is not supported yet
        has_scalarized_outcome_constraints = optimization_config is not None and any(
            isinstance(oc, ScalarizedOutcomeConstraint)
            for oc in optimization_config.outcome_constraints
        )
        best_trials_card = (
            BestTrials().compute_or_error_card(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )
            if not has_batch_trials and not has_scalarized_outcome_constraints
            else None
        )

        # Add utility progression if there are objectives
        # Skip for experiments with ScalarizedOutcomeConstraint as feasibility
        # evaluation for scalarized outcome constraints is not yet implemented
        # Skip for online experiments (those with BatchTrials)
        utility_progression_card = (
            UtilityProgressionAnalysis().compute_or_error_card(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )
            if len(objective_names) > 0
            and not has_scalarized_outcome_constraints
            and not has_batch_trials
            else None
        )

        summary = Summary().compute_or_error_card(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        return self._create_analysis_card_group(
            title=RESULTS_CARDGROUP_TITLE,
            subtitle=RESULTS_CARDGROUP_SUBTITLE,
            children=[
                child
                for child in (
                    arm_effect_pair_group,
                    objective_scatter_group,
                    constraint_scatter_group,
                    bandit_rollout_card,
                    best_trials_card,
                    utility_progression_card,
                    summary,
                )
                if child is not None
            ],
        )


@final
class ArmEffectsPair(Analysis):
    """
    Compute two ArmEffectsPlots in a single AnalysisCardGroup, one plotting model
    predictions and one plotting raw observed data.
    """

    def __init__(
        self,
        metric_names: Sequence[str] | None = None,
        relativize: bool = False,
        trial_index: int | None = None,
        trial_statuses: Sequence[TrialStatus] | None = None,
        additional_arms: Sequence[Arm] | None = None,
        label: str | None = None,
    ) -> None:
        """
        Args:
            metric_names: The names of the metrics to include in the plot. If not
                specified, all metrics in the experiment will be used.
            relativize: Whether to relativize the effects of each arm against the status
                quo arm. If multiple status quo arms are present, relativize each arm
                against the status quo arm from the same trial.
            trial_index: If present, only use arms from the trial with the given index.
            additional_arms: If present, include these arms in the plot in addition to
                the arms in the experiment. These arms will be marked as belonging to a
                trial with index -1.
            label: A label to use in the plot in place of the metric name.
        """

        self.metric_names = metric_names
        self.relativize = relativize
        self.trial_index = trial_index
        self.trial_statuses = trial_statuses
        self.additional_arms = additional_arms
        self.label = label

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardGroup:
        if experiment is None:
            raise UserInputError("ArmEffectsPlot requires an Experiment.")

        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        pairs: list[AnalysisCardGroup] = []
        for metric_name in self.metric_names or [*experiment.metrics.keys()]:
            # TODO: Test for no effects and render a message instead of a flat line.
            predicted_analysis = ArmEffectsPlot(
                metric_name=metric_name,
                use_model_predictions=True,
                relativize=self.relativize,
                trial_index=self.trial_index,
                trial_statuses=self.trial_statuses,
                additional_arms=self.additional_arms,
                label=self.label,
            )

            raw_analysis = ArmEffectsPlot(
                metric_name=metric_name,
                use_model_predictions=False,
                relativize=self.relativize,
                trial_index=self.trial_index,
                trial_statuses=self.trial_statuses,
                additional_arms=self.additional_arms,
                label=self.label,
            )

            pair = AnalysisCardGroup(
                name=f"ArmEffects Pair {metric_name}",
                title=f"Metric Effects Pair for {metric_name}",
                subtitle=None,
                children=[
                    predicted_analysis.compute_or_error_card(
                        experiment=experiment,
                        generation_strategy=generation_strategy,
                        adapter=relevant_adapter,
                    ),
                    raw_analysis.compute_or_error_card(experiment=experiment),
                ],
            )

            pairs.append(pair)

        return self._create_analysis_card_group(
            title=ARM_EFFECTS_PAIR_CARDGROUP_TITLE,
            subtitle=ARM_EFFECTS_PAIR_CARDGROUP_SUBTITLE,
            children=pairs,
        )

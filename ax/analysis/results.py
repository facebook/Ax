# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-safe

import itertools
from typing import Mapping, Sequence

from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.analysis_card import AnalysisCardGroup
from ax.analysis.plotly.arm_effects import ArmEffectsPlot
from ax.analysis.plotly.parallel_coordinates import ParallelCoordinatesPlot
from ax.analysis.plotly.scatter import ScatterPlot
from ax.analysis.summary import Summary
from ax.analysis.utils import extract_relevant_adapter
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import override


class ResultsAnalysis(Analysis):
    """
    An Analysis that provides a high-level overview of the results of the optimization
    process so far, e.g. effects on all arms. It produces an analysis card group.
    """

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardGroup:
        # Ensure we have an Experiment provided by the user to extract the relevant
        # metric names from.
        if experiment is None:
            raise UserInputError("ResultsAnalysis requires an Experiment.")

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

        # Compute both observed and modeled effects for each objective and constraint.
        arm_effect_pair_group = (
            ArmEffectsPair(
                metric_names=[*objective_names, *constraint_names]
            ).compute_or_error_card(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )
            if len(objective_names) > 0
            else None
        )

        # If there are multiple objectives, compute scatter plots of each combination
        # of two objectives.
        objective_scatter_group = (
            AnalysisCardGroup(
                name="Objective Scatter Plots",
                children=[
                    ScatterPlot(
                        x_metric_name=x,
                        y_metric_name=y,
                    ).compute_or_error_card(
                        experiment=experiment,
                        generation_strategy=generation_strategy,
                        adapter=adapter,
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
                children=[
                    ScatterPlot(
                        x_metric_name=objective_name,
                        y_metric_name=constraint_name,
                    ).compute_or_error_card(
                        experiment=experiment,
                        generation_strategy=generation_strategy,
                        adapter=adapter,
                    )
                    for objective_name in objective_names
                    for constraint_name in constraint_names
                ],
            )
            if len(objective_names) > 0 and len(constraint_names) > 0
            else None
        )

        # Produce a parallel coordinates plot for each objective.
        objective_parallel_coordinates_group = (
            AnalysisCardGroup(
                name="Objective Parallel Coordinates Plots",
                children=[
                    ParallelCoordinatesPlot(
                        metric_name=metric_name
                    ).compute_or_error_card(
                        experiment=experiment,
                        generation_strategy=generation_strategy,
                        adapter=adapter,
                    )
                    for metric_name in objective_names
                ],
            )
            if len(objective_names) > 0
            else None
        )

        summary = Summary().compute_or_error_card(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        return self._create_analysis_card_group(
            children=[
                group
                for group in (
                    arm_effect_pair_group,
                    objective_scatter_group,
                    constraint_scatter_group,
                    objective_parallel_coordinates_group,
                    summary,
                )
                if group is not None
            ]
        )


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
        labels: Mapping[str, str] | None = None,
        show_cumulative_best: bool = False,
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
            labels: A mapping from metric names to labels to use in the plot. If a label
                is not provided for a metric, the metric name will be used.
            show_cumulative_best: Whether to draw a line through the best point seen so
                far during the optimization.
        """

        self.metric_names = metric_names
        self.relativize = relativize
        self.trial_index = trial_index
        self.trial_statuses = trial_statuses
        self.additional_arms = additional_arms
        self.labels: Mapping[str, str] = labels or {}
        self.show_cumulative_best = show_cumulative_best

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
                metric_names=[metric_name],
                use_model_predictions=True,
                relativize=self.relativize,
                trial_index=self.trial_index,
                trial_statuses=self.trial_statuses,
                additional_arms=self.additional_arms,
                labels=self.labels,
                show_cumulative_best=self.show_cumulative_best,
            )

            raw_analysis = ArmEffectsPlot(
                metric_names=[metric_name],
                use_model_predictions=False,
                relativize=self.relativize,
                trial_index=self.trial_index,
                trial_statuses=self.trial_statuses,
                additional_arms=self.additional_arms,
                labels=self.labels,
                show_cumulative_best=self.show_cumulative_best,
            )

            pair = AnalysisCardGroup(
                name=f"ArmEffects Pair {metric_name}",
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
            children=pairs,
        )

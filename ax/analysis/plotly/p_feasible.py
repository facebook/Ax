# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import re
from typing import final, Sequence

from ax.adapter.base import Adapter

from ax.analysis.analysis import Analysis
from ax.analysis.plotly.arm_effects import (
    _prepare_figure as _arm_effects_prepare_figure,
)
from ax.analysis.plotly.plotly_analysis import (
    create_plotly_analysis_card,
    PlotlyAnalysisCard,
)
from ax.analysis.utils import (
    extract_relevant_adapter,
    prepare_arm_data,
    validate_adapter_can_predict,
    validate_experiment,
    validate_experiment_has_trials,
    validate_outcome_constraints,
)
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.trial_status import TrialStatus
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import none_throws


@final
class PFeasiblePlot(Analysis):
    """
    Plots the probability than any arm is feasible for all constraints on the
    Experiment's OptimizationConfig.

    The DataFrame computed will contain one row per arm and the following columns:
        - trial_index: The trial index of the arm
        - trial_status: The status of the trial
        - arm_name: The name of the arm
        - generation_node: The name of the ``GenerationNode`` that generated the arm
        - p_feasible_mean: The mean of the probability that the arm is feasible
        - p_feasible_sem: The sem of the probability that the arm is feasible
    """

    def __init__(
        self,
        use_model_predictions: bool = True,
        trial_index: int | None = None,
        trial_statuses: Sequence[TrialStatus] | None = None,
        additional_arms: Sequence[Arm] | None = None,
    ) -> None:
        self.use_model_predictions = use_model_predictions
        self.trial_index = trial_index
        self.trial_statuses = trial_statuses
        self.additional_arms = additional_arms

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

        optimization_config = experiment.optimization_config
        if optimization_config is None:
            return "Experiment must have an OptimizationConfig."

        if len(optimization_config.objective.metrics) > 1:
            return "Only single-objective optimization is supported."

        outcome_constraints_validation_str = validate_outcome_constraints(
            experiment=experiment,
        )
        if outcome_constraints_validation_str is not None:
            return outcome_constraints_validation_str

        outcome_constraint_metrics = [
            outcome_constraint.metric.name
            for outcome_constraint in optimization_config.outcome_constraints
        ]

        # Ensure that we either can predict the outcome constraint metrics or that we
        # have observations for them.
        if self.use_model_predictions:
            adapter_can_predict_validation_str = validate_adapter_can_predict(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
                required_metric_names=outcome_constraint_metrics,
            )
            if adapter_can_predict_validation_str is not None:
                return adapter_can_predict_validation_str
        else:
            if self.additional_arms is not None:
                return (
                    "Cannot compute probability of feasibility with additional arms "
                    "with use_model_predictions=False."
                )

            experiment_has_trials_validation_str = validate_experiment_has_trials(
                experiment=experiment,
                trial_indices=[self.trial_index]
                if self.trial_index is not None
                else None,
                trial_statuses=self.trial_statuses,
                required_metric_names=outcome_constraint_metrics,
            )
            if experiment_has_trials_validation_str is not None:
                return experiment_has_trials_validation_str

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> PlotlyAnalysisCard:
        experiment = none_throws(experiment)
        optimization_config = none_throws(experiment.optimization_config)

        if self.use_model_predictions:
            relevant_adapter = extract_relevant_adapter(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )
        else:
            relevant_adapter = None

        # Compute the data for just the outcome constraint metrics
        arm_data = prepare_arm_data(
            experiment=experiment,
            metric_names=[
                outcome_constrinat.metric.name
                for outcome_constrinat in optimization_config.outcome_constraints
            ],
            use_model_predictions=self.use_model_predictions,
            adapter=relevant_adapter,
            trial_index=self.trial_index,
            trial_statuses=self.trial_statuses,
            additional_arms=self.additional_arms,
            relativize=False,
        )

        fig = _arm_effects_prepare_figure(
            df=arm_data,
            metric_name="p_feasible",
            is_relative=False,
            status_quo_arm_name=(
                experiment.status_quo.name
                if experiment.status_quo is not None
                else None
            ),
            metric_label="% Chance of Feasibility",
        )

        constraints_str = "\n".join(
            re.findall(r"\((.*?)\)", str(constraint))[0]
            for constraint in optimization_config.outcome_constraints
        )

        return create_plotly_analysis_card(
            name=self.__class__.__name__,
            title=(
                ("Predicted" if self.use_model_predictions else "Observed")
                + " Probability of Feasibility"
            ),
            subtitle=(
                "Probability that each arm satisfies all constraints: "
                + constraints_str
            ),
            df=arm_data.loc[
                :,
                [
                    "trial_index",
                    "arm_name",
                    "trial_status",
                    "generation_node",
                    "p_feasible_mean",
                    "p_feasible_sem",
                ],
            ].copy(),
            fig=fig,
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from itertools import chain
from typing import Any

import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel

from ax.analysis.plotly.arm_effects.utils import (
    get_predictions_by_arm,
    prepare_arm_effects_plot,
)

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.utils import is_predictive
from ax.core import OutcomeConstraint
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.exceptions.core import UserInputError
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.transforms.derelativize import Derelativize
from pyre_extensions import assert_is_instance, none_throws


class PredictedEffectsPlot(PlotlyAnalysis):
    """
    Plotly Predicted Effecs plot for a single metric, with one point per unique arm
    across all trials. It plots all observed points, as well as predictions for the
    most recently generated trial.

    This plot is useful for understanding how arms in a candidate trial can be expected
    to perform.

    The DataFrame computed will contain one row per arm and the following columns:
        - source: In-sample or model key that geneerated the candidate
        - arm_name: The name of the arm
        - mean: The observed or predicted mean of the metric specified
        - sem: The observed or predicted sem of the metric specified
        - error_margin: The 95% CI of the metric specified for the arm
        - size_column: The size of the circle in the plot, which represents
            the probability that the arm is feasible (does not violate any
            constraints).
        - parameters: A string representation of the parameters for the arm
            to be viewed in the tooltip.
        - constraints_violated: A string representation of the probability
            each constraint is violated for the arm, to be viewed in the tooltip.
    """

    CARD_NAME = "PredictedEffectsPlot"
    trial_index: int | None = None

    def __init__(self, metric_name: str) -> None:
        """
        Args:
            metric_name: The name of the metric to plot. If not specified the objective
                will be used. Note that the metric cannot be inferred for
                multi-objective or scalarized-objective experiments.
        """
        self.metric_name = metric_name

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> PlotlyAnalysisCard:
        if experiment is None:
            raise UserInputError("PredictedEffectsPlot requires an Experiment.")
        try:
            generation_strategy = assert_is_instance(
                generation_strategy, GenerationStrategy
            )
        except TypeError as e:
            raise UserInputError(
                "PredictedEffectsPlot requires a GenerationStrategy."
            ) from e

        try:
            trial_indices = [
                t.index
                for t in experiment.trials.values()
                if t.status != TrialStatus.ABANDONED
            ]
            candidate_trial = experiment.trials[max(trial_indices)]
            # This is so the card will have a trial_index attribute
            self.trial_index = candidate_trial.index
        except ValueError:
            raise UserInputError(
                f"PredictedEffectsPlot cannot be used for {experiment} "
                "because it has no trials."
            )

        if generation_strategy.model is None:
            generation_strategy._fit_current_model(data=experiment.lookup_data())

        model = none_throws(generation_strategy.model)
        if not is_predictive(model=model):
            raise UserInputError(
                "PredictedEffectsPlot requires a GenerationStrategy which is "
                "in a state where the current model supports prediction.  The current "
                f"model is {model._model_key} and does not support prediction."
            )

        outcome_constraints = (
            []
            if experiment.optimization_config is None
            else Derelativize()
            .transform_optimization_config(
                # TODO[T203521207]: move cloning into transform_optimization_config
                optimization_config=none_throws(experiment.optimization_config).clone(),
                modelbridge=model,
            )
            .outcome_constraints
        )
        df = _prepare_data(
            model=model,
            metric_name=self.metric_name,
            candidate_trial=candidate_trial,
            outcome_constraints=outcome_constraints,
        )
        fig = prepare_arm_effects_plot(
            df=df, metric_name=self.metric_name, outcome_constraints=outcome_constraints
        )

        level = AnalysisCardLevel.HIGH
        nudge = -2
        if experiment.optimization_config is not None:
            if (
                self.metric_name
                in experiment.optimization_config.objective.metric_names
            ):
                nudge = 0
            elif self.metric_name in experiment.optimization_config.metrics:
                nudge = -1

        return self._create_plotly_analysis_card(
            title=f"Predicted Effects for {self.metric_name}",
            subtitle="View a candidate trial and its arms' predicted metric values",
            level=level + nudge,
            df=df,
            fig=fig,
        )


def _prepare_data(
    model: ModelBridge,
    metric_name: str,
    candidate_trial: BaseTrial,
    outcome_constraints: list[OutcomeConstraint],
) -> pd.DataFrame:
    """Prepare data for plotting.  Data should include columns for:
    - source: In-sample or model key that geneerated the candidate
    - arm_name: Name of the arm
    - mean: Predicted metric value
    - error_margin: 1.96 * predicted sem for plotting 95% CI
    - **PARAMETER_NAME: The value of each parameter for the arm.  Will be used
        for the tooltip.
    There will be one row for each arm in the model's training data and one for
    each arm in the generator runs of the candidate trial.  If an arm is in both
    the training data and the candidate trial, it will only appear once for the
    candidate trial.

    Args:
        model: ModelBridge being used for prediction
        metric_name: Name of metric to plot
        candidate_trial: Trial to plot candidates for by generator run
    """
    predictions_for_observed_arms: list[dict[str, Any]] = get_predictions_by_arm(
        model=model,
        metric_name=metric_name,
        outcome_constraints=outcome_constraints,
    )
    candidate_generator_run_predictions: list[list[dict[str, Any]]] = (
        []
        if candidate_trial is None
        else [
            get_predictions_by_arm(
                model=model,
                metric_name=metric_name,
                outcome_constraints=outcome_constraints,
                gr=gr,
                abandoned_arms={a.name for a in candidate_trial.abandoned_arms},
            )
            for gr in candidate_trial.generator_runs
        ]
    )
    df = pd.DataFrame.from_records(
        list(
            chain(
                predictions_for_observed_arms,
                *candidate_generator_run_predictions,
            )
        )
    )
    df.drop_duplicates(subset="arm_name", keep="last", inplace=True)
    return df

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain
from typing import Any

import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.utils import (
    format_constraint_violated_probabilities,
    get_constraint_violated_probabilities,
)
from ax.core import OutcomeConstraint
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.exceptions.core import UserInputError
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.prediction_utils import predict_at_point
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.utils.common.typeutils import checked_cast
from plotly import express as px, graph_objects as go, io as pio
from pyre_extensions import none_throws


class PredictedEffectsPlot(PlotlyAnalysis):
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

        generation_strategy = checked_cast(
            GenerationStrategy,
            generation_strategy,
            exception=UserInputError(
                "PredictedEffectsPlot requires a GenerationStrategy."
            ),
        )

        try:
            trial_indices = [
                t.index
                for t in experiment.trials.values()
                if t.status != TrialStatus.ABANDONED
            ]
            candidate_trial = experiment.trials[max(trial_indices)]
        except ValueError:
            raise UserInputError(
                f"PredictedEffectsPlot cannot be used for {experiment} "
                "because it has no trials."
            )

        if generation_strategy.model is None:
            generation_strategy._fit_current_model(data=experiment.lookup_data())

        model = none_throws(generation_strategy.model)
        outcome_constraints = (
            []
            if experiment.optimization_config is None
            else Derelativize()
            .transform_optimization_config(
                optimization_config=none_throws(experiment.optimization_config),
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
        fig = _prepare_plot(df=df, metric_name=self.metric_name)

        if (
            experiment.optimization_config is None
            or self.metric_name not in experiment.optimization_config.metrics
        ):
            level = AnalysisCardLevel.LOW
        elif self.metric_name in experiment.optimization_config.objective.metric_names:
            level = AnalysisCardLevel.HIGH
        else:
            level = AnalysisCardLevel.MID

        return PlotlyAnalysisCard(
            name="PredictedEffectsPlot",
            title=f"Predicted Effects for {self.metric_name}",
            subtitle="View a candidate trial and its arms' predicted metric values",
            level=level,
            df=df,
            blob=pio.to_json(fig),
            attributes={"trial_index": candidate_trial.index},
        )


def _get_predictions(
    model: ModelBridge,
    metric_name: str,
    outcome_constraints: list[OutcomeConstraint],
    gr: GeneratorRun | None = None,
    trial_index: int | None = None,
) -> list[dict[str, Any]]:
    if gr is None:
        observations = model.get_training_data()
        features = [o.features for o in observations]
        arm_names = [o.arm_name for o in observations]
    else:
        features = [
            ObservationFeatures(parameters=arm.parameters, trial_index=trial_index)
            for arm in gr.arms
        ]
        arm_names = [a.name for a in gr.arms]
    try:
        predictions = [
            predict_at_point(
                model=model,
                obsf=obsf,
                metric_names={metric_name}.union(
                    {constraint.metric.name for constraint in outcome_constraints}
                ),
            )
            for obsf in features
        ]
    except NotImplementedError:
        raise UserInputError(
            "PredictedEffectsPlot requires a GenerationStrategy which is "
            "in a state where the current model supports prediction.  The current "
            f"model is {model._model_key} and does not support prediction."
        )
    constraints_violated_by_constraint = get_constraint_violated_probabilities(
        predictions=predictions,
        outcome_constraints=outcome_constraints,
    )
    probabilities_not_feasible = constraints_violated_by_constraint.pop(
        "any_constraint_violated"
    )
    constraints_violated = [
        {
            c: constraints_violated_by_constraint[c][i]
            for c in constraints_violated_by_constraint
        }
        for i in range(len(features))
    ]

    for i in range(len(features)):
        if (
            model.status_quo is not None
            and features[i].parameters
            == none_throws(model.status_quo).features.parameters
        ):
            probabilities_not_feasible[i] = 0
            constraints_violated[i] = {}
    return [
        {
            "source": "In-sample" if gr is None else gr._model_key,
            "arm_name": arm_names[i],
            "mean": predictions[i][0][metric_name],
            "sem": predictions[i][1][metric_name],
            "error_margin": 1.96 * predictions[i][1][metric_name],
            "constraints_violated": format_constraint_violated_probabilities(
                constraints_violated[i]
            ),
            "size_column": 100 - probabilities_not_feasible[i] * 100,
            "parameters": "<br />  "
            + "<br />  ".join([f"{k}: {v}" for k, v in features[i].parameters.items()]),
        }
        for i in range(len(features))
    ]


def _get_max_observed_trial_index(model: ModelBridge) -> int | None:
    """Returns the max observed trial index to appease multitask models for prediction
    by giving fixed features. This is not necessarily accurate and should eventually
    come from the generation strategy.
    """
    observed_trial_indices = [
        obs.features.trial_index
        for obs in model.get_training_data()
        if obs.features.trial_index is not None
    ]
    if len(observed_trial_indices) == 0:
        return None
    return max(observed_trial_indices)


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
    trial_index = _get_max_observed_trial_index(model)
    df = pd.DataFrame.from_records(
        list(
            chain(
                *[
                    _get_predictions(
                        model=model,
                        metric_name=metric_name,
                        outcome_constraints=outcome_constraints,
                    ),
                    *(
                        []
                        if candidate_trial is None
                        else [
                            _get_predictions(
                                model=model,
                                metric_name=metric_name,
                                outcome_constraints=outcome_constraints,
                                gr=gr,
                                trial_index=trial_index,
                            )
                            for gr in candidate_trial.generator_runs
                        ]
                    ),
                ]
            )
        )
    )
    df.drop_duplicates(subset="arm_name", keep="last", inplace=True)
    return df


def _get_parameter_columns(df: pd.DataFrame) -> dict[str, bool]:
    """Get the names of the columns that represent parameters in df."""
    return {
        col: (col not in ["source", "error_margin", "size_column"])
        for col in df.columns
    }


def _prepare_plot(df: pd.DataFrame, metric_name: str) -> go.Figure:
    """Prepare a plotly figure for the predicted effects based on the data in df."""
    fig = px.scatter(
        df,
        x="arm_name",
        y="mean",
        error_y="error_margin",
        color="source",
        hover_data=_get_parameter_columns(df),
        size="size_column",
        size_max=10,
    )
    if "status_quo" in df["arm_name"].values:
        fig.add_hline(
            y=df[df["arm_name"] == "status_quo"]["mean"].iloc[0],
            line_width=1,
            line_color="red",
        )
    fig.update_layout(
        xaxis={
            "tickangle": 45,
        },
    )
    for trace in fig.data:
        if trace.marker.symbol == "x":
            trace.marker.size = 11  # Larger size for 'x'

    return fig

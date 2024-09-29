# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain
from typing import Any, Optional

import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.exceptions.core import UserInputError
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.prediction_utils import predict_at_point
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
        experiment: Optional[Experiment] = None,
        generation_strategy: Optional[GenerationStrategyInterface] = None,
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
        df = _prepare_data(
            model=model, metric_name=self.metric_name, candidate_trial=candidate_trial
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
        )


def _get_predictions(
    model: ModelBridge,
    metric_name: str,
    gr: Optional[GeneratorRun] = None,
    trial_index: Optional[int] = None,
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
            predict_at_point(model=model, obsf=obsf, metric_names={metric_name})
            for obsf in features
        ]
    except NotImplementedError:
        raise UserInputError(
            "PredictedEffectsPlot requires a GenerationStrategy which is "
            "in a state where the current model supports prediction.  The current "
            f"model is {model._model_key} and does not support prediction."
        )
    return [
        {
            "source": "In-sample" if gr is None else gr._model_key,
            "arm_name": arm_names[i],
            "mean": predictions[i][0][metric_name],
            "error_margin": 1.96 * predictions[i][1][metric_name],
            **features[i].parameters,
        }
        for i in range(len(features))
    ]


def _get_max_observed_trial_index(model: ModelBridge) -> Optional[int]:
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
    model: ModelBridge, metric_name: str, candidate_trial: BaseTrial
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
                    _get_predictions(model, metric_name),
                    *(
                        []
                        if candidate_trial is None
                        else [
                            _get_predictions(model, metric_name, gr, trial_index)
                            for gr in candidate_trial.generator_runs
                        ]
                    ),
                ]
            )
        )
    )
    df.drop_duplicates(subset="arm_name", keep="last", inplace=True)
    return df


def _get_parameter_columns(df: pd.DataFrame) -> list[str]:
    """Get the names of the columns that represent parameters in df."""
    return [
        col
        for col in df.columns
        if col not in ["source", "arm_name", "mean", "error_margin"]
    ]


def _prepare_plot(df: pd.DataFrame, metric_name: str) -> go.Figure:
    """Prepare a plotly figure for the predicted effects based on the data in df."""
    fig = px.scatter(
        df,
        x="arm_name",
        y="mean",
        error_y="error_margin",
        color="source",
        hover_data=_get_parameter_columns(df),
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
    return fig

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from typing import Optional

import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.surface.utils import (
    get_parameter_values,
    is_axis_log_scale,
    select_fixed_value,
)
from ax.analysis.plotly.utils import select_metric
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.core.observation import ObservationFeatures
from ax.exceptions.core import UserInputError
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.generation_strategy import GenerationStrategy
from plotly import express as px, graph_objects as go
from pyre_extensions import none_throws


class SlicePlot(PlotlyAnalysis):
    """
    Plot a 1D "slice" of the surrogate model's predicted outcomes for a given
    parameter, where all other parameters are held fixed at their status-quo value or
    mean if no status quo is available.

    The DataFrame computed will contain the following columns:
        - PARAMETER_NAME: The value of the parameter specified
        - METRIC_NAME_mean: The predected mean of the metric specified
        - METRIC_NAME_sem: The predected sem of the metric specified
    """

    def __init__(
        self,
        parameter_name: str,
        metric_name: str | None = None,
    ) -> None:
        """
        Args:
            parameter_name: The name of the parameter to plot on the x axis.
            metric_name: The name of the metric to plot on the y axis. If not
                specified the objective will be used.
        """
        self.parameter_name = parameter_name
        self.metric_name = metric_name

    def compute(
        self,
        experiment: Optional[Experiment] = None,
        generation_strategy: Optional[GenerationStrategyInterface] = None,
    ) -> PlotlyAnalysisCard:
        if experiment is None:
            raise UserInputError("SlicePlot requires an Experiment")

        if not isinstance(generation_strategy, GenerationStrategy):
            raise UserInputError("SlicePlot requires a GenerationStrategy")

        if generation_strategy.model is None:
            generation_strategy._fit_current_model(None)

        metric_name = self.metric_name or select_metric(experiment=experiment)

        df = _prepare_data(
            experiment=experiment,
            model=none_throws(generation_strategy.model),
            parameter_name=self.parameter_name,
            metric_name=metric_name,
        )

        fig = _prepare_plot(
            df=df,
            parameter_name=self.parameter_name,
            metric_name=metric_name,
            log_x=is_axis_log_scale(
                parameter=experiment.search_space.parameters[self.parameter_name]
            ),
        )

        return self._create_plotly_analysis_card(
            title=f"{self.parameter_name} vs. {metric_name}",
            subtitle=(
                "1D slice of the surrogate model's predicted outcomes for "
                f"{metric_name}"
            ),
            level=AnalysisCardLevel.LOW,
            df=df,
            fig=fig,
        )


def _prepare_data(
    experiment: Experiment,
    model: ModelBridge,
    parameter_name: str,
    metric_name: str,
) -> pd.DataFrame:
    # Choose which parameter values to predict points for.
    xs = get_parameter_values(
        parameter=experiment.search_space.parameters[parameter_name]
    )

    # Construct observation features for each parameter value previously chosen by
    # fixing all other parameters to their status-quo value or mean.
    features = [
        ObservationFeatures(
            parameters={
                parameter_name: x,
                **{
                    parameter.name: select_fixed_value(parameter=parameter)
                    for parameter in experiment.search_space.parameters.values()
                    if parameter.name != parameter_name
                },
            }
        )
        for x in xs
    ]

    predictions = model.predict(observation_features=features)

    return pd.DataFrame.from_records(
        [
            {
                parameter_name: xs[i],
                f"{metric_name}_mean": predictions[0][metric_name][i],
                f"{metric_name}_sem": predictions[1][metric_name][metric_name][i],
            }
            for i in range(len(xs))
        ]
    ).sort_values(by=parameter_name)


def _prepare_plot(
    df: pd.DataFrame,
    parameter_name: str,
    metric_name: str,
    log_x: bool = False,
) -> go.Figure:
    x = df[parameter_name].tolist()
    y = df[f"{metric_name}_mean"].tolist()
    y_upper = (df[f"{metric_name}_mean"] + 1.96 * df[f"{metric_name}_sem"]).tolist()
    y_lower = (df[f"{metric_name}_mean"] - 1.96 * df[f"{metric_name}_sem"]).tolist()

    plotly_blue = px.colors.qualitative.Plotly[0]
    plotly_blue_translucent = "rgba(99, 110, 250, 0.2)"

    # Draw a line at the mean and a shaded region between the upper and lower bounds
    line = go.Scatter(
        x=x,
        y=y,
        line={"color": plotly_blue},
        mode="lines",
        name=metric_name,
        showlegend=False,
    )
    error_band = go.Scatter(
        # Concatenate x values in reverse order to create a closed polygon
        x=x + x[::-1],
        # Concatenate upper and lower bounds in reverse order
        y=y_upper + y_lower[::-1],
        fill="toself",
        fillcolor=plotly_blue_translucent,
        line={"color": "rgba(255,255,255,0)"},  # Make "line" transparent
        hoverinfo="skip",
        showlegend=False,
    )

    fig = go.Figure(
        [line, error_band],
        layout=go.Layout(
            xaxis_title=parameter_name,
            yaxis_title=metric_name,
        ),
    )

    # Set the x-axis scale to log if relevant
    if log_x:
        fig.update_xaxes(
            type="log",
            range=[
                math.log10(df[parameter_name].min()),
                math.log10(df[parameter_name].max()),
            ],
        )
    else:
        fig.update_xaxes(range=[df[parameter_name].min(), df[parameter_name].max()])

    return fig

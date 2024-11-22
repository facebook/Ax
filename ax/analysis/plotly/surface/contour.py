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
from plotly import graph_objects as go
from pyre_extensions import none_throws


class ContourPlot(PlotlyAnalysis):
    """
    Plot a 2D surface of the surrogate model's predicted outcomes for a given pair of
    parameters, where all other parameters are held fixed at their status-quo value or
    mean if no status quo is available.

    The DataFrame computed will contain the following columns:
        - PARAMETER_NAME: The value of the x parameter specified
        - PARAMETER_NAME: The value of the y parameter specified
        - METRIC_NAME: The predected mean of the metric specified
    """

    def __init__(
        self,
        x_parameter_name: str,
        y_parameter_name: str,
        metric_name: str | None = None,
    ) -> None:
        """
        Args:
            y_parameter_name: The name of the parameter to plot on the x-axis.
            y_parameter_name: The name of the parameter to plot on the y-axis.
            metric_name: The name of the metric to plot
        """
        # TODO: Add a flag to specify whether or not to plot markers at the (x, y)
        # coordinates of arms (with hover text). This is fine to exlude for now because
        # this Analysis is only used in the InteractionPlot, but when we want to use it
        # a-la-carte we should add this feature.
        self.x_parameter_name = x_parameter_name
        self.y_parameter_name = y_parameter_name
        self.metric_name = metric_name

    def compute(
        self,
        experiment: Optional[Experiment] = None,
        generation_strategy: Optional[GenerationStrategyInterface] = None,
    ) -> PlotlyAnalysisCard:
        if experiment is None:
            raise UserInputError("ContourPlot requires an Experiment")

        if not isinstance(generation_strategy, GenerationStrategy):
            raise UserInputError("ContourPlot requires a GenerationStrategy")

        if generation_strategy.model is None:
            generation_strategy._fit_current_model(None)

        metric_name = self.metric_name or select_metric(experiment=experiment)

        df = _prepare_data(
            experiment=experiment,
            model=none_throws(generation_strategy.model),
            x_parameter_name=self.x_parameter_name,
            y_parameter_name=self.y_parameter_name,
            metric_name=metric_name,
        )

        fig = _prepare_plot(
            df=df,
            x_parameter_name=self.x_parameter_name,
            y_parameter_name=self.y_parameter_name,
            metric_name=metric_name,
            log_x=is_axis_log_scale(
                parameter=experiment.search_space.parameters[self.x_parameter_name]
            ),
            log_y=is_axis_log_scale(
                parameter=experiment.search_space.parameters[self.y_parameter_name]
            ),
        )

        return self._create_plotly_analysis_card(
            title=(
                f"{self.x_parameter_name}, {self.y_parameter_name} vs. {metric_name}"
            ),
            subtitle=(
                "2D contour of the surrogate model's predicted outcomes for "
                f"{metric_name}"
            ),
            level=AnalysisCardLevel.LOW,
            df=df,
            fig=fig,
        )


def _prepare_data(
    experiment: Experiment,
    model: ModelBridge,
    x_parameter_name: str,
    y_parameter_name: str,
    metric_name: str,
) -> pd.DataFrame:
    # Choose which parameter values to predict points for.
    xs = get_parameter_values(
        parameter=experiment.search_space.parameters[x_parameter_name], density=10
    )
    ys = get_parameter_values(
        parameter=experiment.search_space.parameters[y_parameter_name], density=10
    )

    # Construct observation features for each parameter value previously chosen by
    # fixing all other parameters to their status-quo value or mean.
    features = [
        ObservationFeatures(
            parameters={
                x_parameter_name: x,
                y_parameter_name: y,
                **{
                    parameter.name: select_fixed_value(parameter=parameter)
                    for parameter in experiment.search_space.parameters.values()
                    if not (
                        parameter.name == x_parameter_name
                        or parameter.name == y_parameter_name
                    )
                },
            }
        )
        for x in xs
        for y in ys
    ]

    predictions = model.predict(observation_features=features)

    return pd.DataFrame.from_records(
        [
            {
                x_parameter_name: features[i].parameters[x_parameter_name],
                y_parameter_name: features[i].parameters[y_parameter_name],
                f"{metric_name}_mean": predictions[0][metric_name][i],
            }
            for i in range(len(features))
        ]
    )


def _prepare_plot(
    df: pd.DataFrame,
    x_parameter_name: str,
    y_parameter_name: str,
    metric_name: str,
    log_x: bool,
    log_y: bool,
) -> go.Figure:
    z_grid = df.pivot(
        index=y_parameter_name, columns=x_parameter_name, values=f"{metric_name}_mean"
    )

    fig = go.Figure(
        data=go.Contour(
            z=z_grid.values,
            x=z_grid.columns.values,
            y=z_grid.index.values,
            contours_coloring="heatmap",
            showscale=False,
        ),
        layout=go.Layout(
            xaxis_title=x_parameter_name,
            yaxis_title=y_parameter_name,
        ),
    )

    # Set the x-axis scale to log if relevant
    if log_x:
        fig.update_xaxes(
            type="log",
            range=[
                math.log10(df[x_parameter_name].min()),
                math.log10(df[x_parameter_name].max()),
            ],
        )
    else:
        fig.update_xaxes(range=[df[x_parameter_name].min(), df[x_parameter_name].max()])

    if log_y:
        fig.update_yaxes(
            type="log",
            range=[
                math.log10(df[y_parameter_name].min()),
                math.log10(df[y_parameter_name].max()),
            ],
        )
    else:
        fig.update_yaxes(range=[df[y_parameter_name].min(), df[y_parameter_name].max()])

    return fig

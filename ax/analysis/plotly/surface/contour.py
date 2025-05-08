# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from typing import Sequence

import pandas as pd
from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.surface.utils import (
    get_parameter_values,
    is_axis_log_scale,
    select_fixed_value,
)
from ax.analysis.plotly.utils import (
    METRIC_CONTINUOUS_COLOR_SCALE,
    select_metric,
    truncate_label,
)
from ax.analysis.utils import extract_relevant_adapter
from ax.core.experiment import Experiment
from ax.core.observation import ObservationFeatures
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from plotly import graph_objects as go
from pyre_extensions import none_throws, override


class ContourPlot(PlotlyAnalysis):
    """
    Plot a 2D surface of the surrogate model's predicted outcomes for a given pair of
    parameters, where all other parameters are held fixed at their status-quo value or
    mean if no status quo is available.

    The DataFrame computed will contain the following columns:
        - PARAMETER_NAME: The value of the x parameter specified
        - PARAMETER_NAME: The value of the y parameter specified
        - METRIC_NAME: The predected mean of the metric specified
        - sampled: Whether the parameter values were sampled in at least one trial
    """

    def __init__(
        self,
        x_parameter_name: str,
        y_parameter_name: str,
        metric_name: str | None = None,
        display_sampled: bool = True,
    ) -> None:
        """
        Args:
            y_parameter_name: The name of the parameter to plot on the x-axis.
            y_parameter_name: The name of the parameter to plot on the y-axis.
            metric_name: The name of the metric to plot
            display_sampled: If True, plot "x"s at x coordinates which have been
                sampled in at least one trial.
        """
        self.x_parameter_name = x_parameter_name
        self.y_parameter_name = y_parameter_name
        self.metric_name = metric_name
        self._display_sampled = display_sampled

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[PlotlyAnalysisCard]:
        if experiment is None:
            raise UserInputError("ContourPlot requires an Experiment")

        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        metric_name = self.metric_name or select_metric(experiment=experiment)

        df = _prepare_data(
            experiment=experiment,
            model=relevant_adapter,
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
            display_sampled=self._display_sampled,
        )

        return [
            self._create_plotly_analysis_card(
                title=(
                    f"{self.x_parameter_name}, {self.y_parameter_name} vs. "
                    f"{metric_name}"
                ),
                subtitle=(
                    "The contour plot visualizes the predicted outcomes "
                    f"for {metric_name} across a two-dimensional parameter space, "
                    "with other parameters held fixed at their status_quo value "
                    "(or mean value if status_quo is unavailable). This plot helps "
                    "in identifying regions of optimal performance and understanding "
                    "how changes in the selected parameters influence the predicted "
                    "outcomes. Contour lines represent levels of constant predicted "
                    "values, providing insights into the gradient and potential optima "
                    "within the parameter space."
                ),
                level=AnalysisCardLevel.LOW,
                df=df,
                fig=fig,
                category=AnalysisCardCategory.INSIGHT,
            )
        ]


def _prepare_data(
    experiment: Experiment,
    model: Adapter,
    x_parameter_name: str,
    y_parameter_name: str,
    metric_name: str,
) -> pd.DataFrame:
    sampled = [
        (arm.parameters[x_parameter_name], arm.parameters[y_parameter_name])
        for trial in experiment.trials.values()
        for arm in trial.arms
    ]

    # Choose which parameter values to predict points for.
    unsampled_xs = get_parameter_values(
        parameter=experiment.search_space.parameters[x_parameter_name], density=10
    )
    unsampled_ys = get_parameter_values(
        parameter=experiment.search_space.parameters[y_parameter_name], density=10
    )

    xs = [*[sample[0] for sample in sampled], *unsampled_xs]
    ys = [*[sample[1] for sample in sampled], *unsampled_ys]

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
        # Do not create features for any out of sample points
        if experiment.search_space.check_membership(
            parameterization={
                x_parameter_name: x,
                y_parameter_name: y,
            },
            raise_error=False,
            check_all_parameters_present=False,
        )
    ]

    predictions = model.predict(observation_features=features)

    return none_throws(
        pd.DataFrame.from_records(
            [
                {
                    x_parameter_name: features[i].parameters[x_parameter_name],
                    y_parameter_name: features[i].parameters[y_parameter_name],
                    f"{metric_name}_mean": predictions[0][metric_name][i],
                    "sampled": (
                        features[i].parameters[x_parameter_name],
                        features[i].parameters[y_parameter_name],
                    )
                    in sampled,
                }
                for i in range(len(features))
            ]
        ).drop_duplicates()
    )


def _prepare_plot(
    df: pd.DataFrame,
    x_parameter_name: str,
    y_parameter_name: str,
    metric_name: str,
    log_x: bool,
    log_y: bool,
    display_sampled: bool,
) -> go.Figure:
    z_grid = df.pivot_table(
        index=y_parameter_name,
        columns=x_parameter_name,
        values=f"{metric_name}_mean",
        # aggfunc is required to gracefully handle duplicate values
        aggfunc="mean",
    )

    fig = go.Figure(
        data=go.Contour(
            z=z_grid.values,
            x=z_grid.columns.values,
            y=z_grid.index.values,
            colorscale=METRIC_CONTINUOUS_COLOR_SCALE,
            showscale=False,
        ),
        layout=go.Layout(
            xaxis_title=truncate_label(label=x_parameter_name),
            yaxis_title=truncate_label(label=y_parameter_name),
        ),
    )

    if display_sampled:
        x_sampled = df[df["sampled"]][x_parameter_name].tolist()
        y_sampled = df[df["sampled"]][y_parameter_name].tolist()

        samples = go.Scatter(
            x=x_sampled,
            y=y_sampled,
            mode="markers",
            marker={
                "symbol": "x",
                "color": "black",
            },
            name="Sampled",
            showlegend=False,
        )

        fig.add_trace(samples)

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

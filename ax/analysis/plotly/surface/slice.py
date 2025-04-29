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
    get_scatter_point_color,
    select_metric,
    truncate_label,
)
from ax.analysis.utils import extract_relevant_adapter
from ax.core.experiment import Experiment
from ax.core.observation import ObservationFeatures
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from plotly import express as px, graph_objects as go
from pyre_extensions import none_throws, override


class SlicePlot(PlotlyAnalysis):
    """
    Plot a 1D "slice" of the surrogate model's predicted outcomes for a given
    parameter, where all other parameters are held fixed at their status-quo value or
    mean if no status quo is available.

    The DataFrame computed will contain the following columns:
        - PARAMETER_NAME: The value of the parameter specified
        - METRIC_NAME_mean: The predected mean of the metric specified
        - METRIC_NAME_sem: The predected sem of the metric specified
        - sampled: Whether the parameter value was sampled in at least one trial
    """

    def __init__(
        self,
        parameter_name: str,
        metric_name: str | None = None,
        display_sampled: bool = True,
    ) -> None:
        """
        Args:
            parameter_name: The name of the parameter to plot on the x axis.
            metric_name: The name of the metric to plot on the y axis. If not
                specified the objective will be used.
            display_sampled: If True, plot "x"s at x coordinates which have been
                sampled in at least one trial.
        """
        self.parameter_name = parameter_name
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
            raise UserInputError("SlicePlot requires an Experiment")

        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        metric_name = self.metric_name or select_metric(experiment=experiment)

        df = _prepare_data(
            experiment=experiment,
            model=relevant_adapter,
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
            display_sampled=self._display_sampled,
        )

        return [
            self._create_plotly_analysis_card(
                title=f"{self.parameter_name} vs. {metric_name}",
                subtitle=(
                    "The slice plot provides a one-dimensional view of predicted "
                    f"outcomes for {metric_name} as a function of a single parameter, "
                    "while keeping all other parameters fixed at their status_quo "
                    "value (or mean value if status_quo is unavailable). "
                    "This visualization helps in understanding the sensitivity and "
                    "impact of changes in the selected parameter on the predicted "
                    "metric outcomes."
                ),
                level=AnalysisCardLevel.LOW,
                df=df,
                fig=fig,
                category=AnalysisCardCategory.INSIGHT,
            )
        ]


def compute_slice_adhoc(
    parameter_name: str,
    experiment: Experiment,
    generation_strategy: GenerationStrategy | None = None,
    adapter: Adapter | None = None,
    metric_name: str | None = None,
    display_sampled: bool = True,
) -> list[PlotlyAnalysisCard]:
    """
    Helper method to expose adhoc cross validation plotting. Only for advanced users in
    a notebook setting.

    Args:
        parameter_name: The name of the parameter to plot on the x-axis.
        experiment: The experiment to source data from.
        generation_strategy: Optional. The generation strategy to extract the adapter
            from.
        adapter: Optional. The adapter to use for predictions.
        metric_name: The name of the metric to plot on the y-axis. If not specified
            the objective will be used.
        display_sampled: If True, plot "x"s at x coordinates which have been sampled
            in at least one trial.

    """

    analysis = SlicePlot(
        parameter_name=parameter_name,
        metric_name=metric_name,
        display_sampled=display_sampled,
    )

    return [
        *analysis.compute(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )
    ]


def _prepare_data(
    experiment: Experiment,
    model: Adapter,
    parameter_name: str,
    metric_name: str,
) -> pd.DataFrame:
    sampled_xs = [
        arm.parameters[parameter_name]
        for trial in experiment.trials.values()
        for arm in trial.arms
        # Exclude parameter values which are not valid (ex. None when the parameter is
        # not known in a status quo arm).
        if experiment.search_space.parameters[parameter_name].validate(
            arm.parameters[parameter_name]
        )
    ]
    # Choose which parameter values to predict points for.
    unsampled_xs = get_parameter_values(
        parameter=experiment.search_space.parameters[parameter_name]
    )
    xs = [*sampled_xs, *unsampled_xs]

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

    return none_throws(
        pd.DataFrame.from_records(
            [
                {
                    parameter_name: xs[i],
                    f"{metric_name}_mean": predictions[0][metric_name][i],
                    f"{metric_name}_sem": predictions[1][metric_name][metric_name][i]
                    ** 0.5,  # Convert the variance to the SEM
                    "sampled": xs[i] in sampled_xs,
                }
                for i in range(len(xs))
            ]
        ).drop_duplicates()
    ).sort_values(by=parameter_name)


def _prepare_plot(
    df: pd.DataFrame,
    parameter_name: str,
    metric_name: str,
    log_x: bool,
    display_sampled: bool,
) -> go.Figure:
    x = df[parameter_name].tolist()
    y = df[f"{metric_name}_mean"].tolist()

    # Convert the SEMs to 95% confidence intervals
    y_upper = (df[f"{metric_name}_mean"] + 1.96 * df[f"{metric_name}_sem"]).tolist()
    y_lower = (df[f"{metric_name}_mean"] - 1.96 * df[f"{metric_name}_sem"]).tolist()

    plotly_blue = px.colors.qualitative.Plotly[0]

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
        fillcolor=get_scatter_point_color(hex_color=plotly_blue, ci_transparency=True),
        line={"color": "rgba(255,255,255,0)"},  # Make "line" transparent
        hoverinfo="skip",
        showlegend=False,
    )

    fig = go.Figure(
        [line, error_band],
        layout=go.Layout(
            xaxis_title=truncate_label(label=parameter_name),
            yaxis_title=truncate_label(label=metric_name),
        ),
    )

    if display_sampled:
        sampled = df[df["sampled"]]
        x_sampled = sampled[parameter_name].tolist()
        y_sampled = sampled[f"{metric_name}_mean"].tolist()

        samples = go.Scatter(
            x=x_sampled,
            y=y_sampled,
            mode="markers",
            marker={
                "symbol": "x",
                "color": "black",
            },
            name=f"Sampled {parameter_name}",
            showlegend=False,
        )

        fig.add_trace(samples)

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

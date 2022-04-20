# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Optional

from ax.benchmark.benchmark_result import AggregatedBenchmarkResult
from ax.plot.base import AxPlotConfig, AxPlotTypes
from ax.plot.color import COLORS, DISCRETE_COLOR_SCALE, rgba
from ax.plot.helper import rgb
from plotly import graph_objs as go


def plot_modeling_times(
    aggregated_results: Iterable[AggregatedBenchmarkResult],
) -> AxPlotConfig:
    """Plots wall times of each method's fit and gen calls as a stack bar chart."""

    data = [
        go.Bar(
            name="fit",
            x=[result.name for result in aggregated_results],
            y=[result.fit_time[0] for result in aggregated_results],
            text=["fit" for _ in aggregated_results],
            error_y={
                "type": "data",
                "array": [result.fit_time[1] for result in aggregated_results],
                "visible": True,
            },
            opacity=0.6,
        ),
        go.Bar(
            name="gen",
            x=[result.name for result in aggregated_results],
            y=[result.gen_time[0] for result in aggregated_results],
            text=["gen" for _ in aggregated_results],
            error_y={
                "type": "data",
                "array": [agg.gen_time[1] for agg in aggregated_results],
                "visible": True,
            },
            opacity=0.9,
        ),
    ]

    layout = go.Layout(
        title="Modeling Times",
        showlegend=False,
        yaxis={"title": "Time (s)"},
        xaxis={"title": "Method"},
        barmode="stack",
    )

    return AxPlotConfig(
        data=go.Figure(layout=layout, data=data), plot_type=AxPlotTypes.GENERIC
    )


def plot_optimization_trace(
    aggregated_results: Iterable[AggregatedBenchmarkResult],
    optimum: Optional[float] = None,
) -> AxPlotConfig:
    """Plots optimization trace for each aggregated result with mean and SEM.

    If an optimum is provided (can represent either an optimal value or maximum
    hypervolume in the case of multi-objective problems) it will be plotted as an
    orange dashed line as well.
    """

    x = [*range(max(len(result.optimization_trace) for result in aggregated_results))]

    mean_sem_scatters = [
        [
            go.Scatter(
                x=x,
                y=result.optimization_trace["mean"],
                line={
                    "color": rgba(DISCRETE_COLOR_SCALE[i % len(DISCRETE_COLOR_SCALE)])
                },
                mode="lines",
                name=result.name,
                customdata=result.optimization_trace["sem"],
                hovertemplate="<br><b>Mean:</b> %{y}<br><b>SEM</b>: %{customdata}",
            ),
            go.Scatter(
                x=x,
                y=result.optimization_trace["mean"] + result.optimization_trace["sem"],
                line={"width": 0},
                mode="lines",
                fillcolor=rgba(
                    DISCRETE_COLOR_SCALE[i % len(DISCRETE_COLOR_SCALE)], 0.3
                ),
                fill="tonexty",
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=x,
                y=result.optimization_trace["mean"] - result.optimization_trace["sem"],
                line={"width": 0},
                mode="lines",
                fillcolor=rgba(
                    DISCRETE_COLOR_SCALE[i % len(DISCRETE_COLOR_SCALE)], 0.3
                ),
                fill="tonexty",
                showlegend=False,
                hoverinfo="skip",
            ),
        ]
        for i, result in enumerate(aggregated_results)
    ]

    optimum_scatter = (
        [
            go.Scatter(
                x=x,
                y=[optimum] * len(x),
                mode="lines",
                line={"dash": "dash", "color": rgb(COLORS.ORANGE.value)},
                name="Optimum",
                hovertemplate="Optimum: %{y}",
            )
        ]
        if optimum is not None
        else []
    )

    layout = go.Layout(
        title="Optimization Traces",
        yaxis={"title": "Best Found"},
        xaxis={"title": "Iteration"},
        hovermode="x unified",
    )

    return AxPlotConfig(
        data=go.Figure(
            layout=layout,
            data=[scatter for sublist in mean_sem_scatters for scatter in sublist]
            + optimum_scatter,
        ),
        plot_type=AxPlotTypes.GENERIC,
    )

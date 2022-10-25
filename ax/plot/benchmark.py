# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, List, Optional

from ax.benchmark.benchmark_result import AggregatedBenchmarkResult
from ax.plot.base import AxPlotConfig, AxPlotTypes
from ax.plot.color import COLORS, DISCRETE_COLOR_SCALE, rgba
from ax.plot.helper import rgb
from plotly import graph_objs as go


def plot_modeling_times(
    aggregated_results: List[AggregatedBenchmarkResult],
    labels: Optional[List[str]] = None,
) -> AxPlotConfig:
    """Plots wall times of each method's fit and gen calls as a stack bar chart."""
    if labels is None:
        labels = [result.name for result in aggregated_results]
    data = [
        go.Bar(
            name="fit",
            x=labels,
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
            x=labels,
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
    aggregated_results: List[AggregatedBenchmarkResult],
    optimum: Optional[float] = None,
    by_progression: bool = False,
    final_progression_only: bool = False,
    labels: Optional[List[str]] = None,
) -> AxPlotConfig:
    """Plots optimization trace for each aggregated result with mean and SEM. When
    `by_progression` is True, the results are plotted with progressions on the
    x-axis. In that case, if `final_progression_only` is True, then the value of
    a trial is taken to be the value of its final progression.

    If an optimum is provided (can represent either an optimal value or maximum
    hypervolume in the case of multi-objective problems) it will be plotted as an
    orange dashed line as well.
    """
    if labels is None:
        labels = [result.name for result in aggregated_results]
    x_axes = []
    dfs = []
    for agg_res in aggregated_results:
        if not by_progression:
            x_axes.append([*range(len(agg_res.optimization_trace))])
            dfs.append(agg_res.optimization_trace)
        else:
            optim_trace_by_prog_res = agg_res.optimization_trace_by_progression(
                final_progression_only=final_progression_only
            )
            x_axes.append(optim_trace_by_prog_res["progression"])
            dfs.append(optim_trace_by_prog_res)

    mean_sem_scatters = [
        [
            go.Scatter(
                x=x_axis,
                y=df["mean"],
                line={
                    "color": rgba(DISCRETE_COLOR_SCALE[i % len(DISCRETE_COLOR_SCALE)])
                },
                mode="lines",
                name=label,
                customdata=df["sem"],
                hovertemplate="<br><b>Mean:</b> %{y}<br><b>SEM</b>: %{customdata}",
            ),
            go.Scatter(
                x=x_axis,
                y=df["mean"] + df["sem"],
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
                x=x_axis,
                y=df["mean"] - df["sem"],
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
        for i, (x_axis, df, label) in enumerate(zip(x_axes, dfs, labels))
    ]

    optimum_scatter = (
        [
            go.Scatter(
                x=x_axes[0],
                y=[optimum] * len(x_axes[0]),
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


def plot_progression_trace(
    aggregated_results: List[AggregatedBenchmarkResult],
    labels: Optional[List[str]] = None,
) -> AxPlotConfig:
    if labels is None:
        labels = [result.name for result in aggregated_results]
    x_axes = []
    dfs = []
    for agg_res in aggregated_results:
        progression_trace = agg_res.progression_trace()
        dfs.append(progression_trace)
        x_axes.append([*range(len(progression_trace))])

    mean_sem_scatters = [
        [
            go.Scatter(
                x=x_axis,
                y=df["mean"],
                line={
                    "color": rgba(DISCRETE_COLOR_SCALE[i % len(DISCRETE_COLOR_SCALE)])
                },
                mode="lines",
                name=label,
                customdata=df["sem"],
                hovertemplate="<br><b>Mean:</b> %{y}<br><b>SEM</b>: %{customdata}",
            ),
            go.Scatter(
                x=x_axis,
                y=df["mean"] + df["sem"],
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
                x=x_axis,
                y=df["mean"] - df["sem"],
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
        for i, (x_axis, df, label) in enumerate(zip(x_axes, dfs, labels))
    ]
    layout = go.Layout(
        title="Progression Traces",
        yaxis={"title": "Steps Used"},
        xaxis={"title": "Iteration"},
        hovermode="x unified",
    )
    return AxPlotConfig(
        data=go.Figure(
            layout=layout,
            data=[scatter for sublist in mean_sem_scatters for scatter in sublist],
        ),
        plot_type=AxPlotTypes.GENERIC,
    )


def plot_total_progressions(
    aggregated_results: Iterable[AggregatedBenchmarkResult],
    labels: Optional[List[str]] = None,
) -> AxPlotConfig:
    """Plots total progressions used by each method as a bar chart."""
    if labels is None:
        labels = [result.name for result in aggregated_results]
    total_progressions = [result.total_progression() for result in aggregated_results]
    data = [
        go.Bar(
            name="total progressions",
            x=labels,
            y=[result[0] for result in total_progressions],
            error_y={
                "type": "data",
                "array": [result[1] for result in total_progressions],
                "visible": True,
            },
            opacity=0.6,
        ),
    ]

    layout = go.Layout(
        title="Total Steps",
        showlegend=False,
        yaxis={"title": "Total Steps"},
        xaxis={"title": "Method"},
    )

    return AxPlotConfig(
        data=go.Figure(layout=layout, data=data), plot_type=AxPlotTypes.GENERIC
    )

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objs as go
from ax.plot.base import AxPlotConfig, AxPlotTypes
from ax.plot.color import COLORS, DISCRETE_COLOR_SCALE, rgba


# type aliases
Traces = List[Dict[str, Any]]


def mean_trace_scatter(
    y: np.ndarray,
    trace_color: Tuple[int] = COLORS.STEELBLUE.value,
    legend_label: str = "mean",
    hover_labels: Optional[List[str]] = None,
) -> go.Scatter:
    """Creates a graph object for trace of the mean of the given series across
    runs.

    Args:
        y: (r x t) array with results from  r runs and t trials.
        trace_color: tuple of 3 int values representing an RGB color.
            Defaults to blue.
        legend_label: label for this trace.
        hover_labels: optional, text to show on hover; list where the i-th value
            corresponds to the i-th value in the value of the `y` argument.

    Returns:
        go.Scatter: plotly graph object
    """
    return go.Scatter(
        name=legend_label,
        legendgroup=legend_label,
        x=np.arange(1, y.shape[1] + 1),
        y=np.mean(y, axis=0),
        mode="lines",
        line={"color": rgba(trace_color)},
        fillcolor=rgba(trace_color, 0.3),
        fill="tonexty",
        text=hover_labels,
    )


def sem_range_scatter(
    y: np.ndarray,
    trace_color: Tuple[int] = COLORS.STEELBLUE.value,
    legend_label: str = "",
) -> Tuple[go.Scatter, go.Scatter]:
    """Creates a graph object for trace of mean +/- 2 SEMs for y, across runs.

    Args:
        y: (r x t) array with results from  r runs and t trials.
        trace_color: tuple of 3 int values representing an RGB color.
            Defaults to blue.
        legend_label: Label for the legend group.

    Returns:
        Tuple[go.Scatter]: plotly graph objects for lower and upper bounds
    """
    mean = np.mean(y, axis=0)
    sem = np.std(y, axis=0) / np.sqrt(y.shape[0])
    return (
        go.Scatter(
            x=np.arange(1, y.shape[1] + 1),
            y=mean - 2 * sem,
            legendgroup=legend_label,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="none",
        ),
        go.Scatter(
            x=np.arange(1, y.shape[1] + 1),
            y=mean + 2 * sem,
            legendgroup=legend_label,
            mode="lines",
            line={"width": 0},
            fillcolor=rgba(trace_color, 0.3),
            fill="tonexty",
            showlegend=False,
            hoverinfo="none",
        ),
    )


def optimum_objective_scatter(
    optimum: float, num_iterations: int, optimum_color: Tuple[int] = COLORS.ORANGE.value
) -> go.Scatter:
    """Creates a graph object for the line representing optimal objective.

    Args:
        optimum: value of the optimal objective
        num_iterations: how many trials were in the optimization (used to
            determine the width of the plot)
        trace_color: tuple of 3 int values representing an RGB color.
            Defaults to orange.

    Returns:
        go.Scatter: plotly graph objects for the optimal objective line
    """
    return go.Scatter(
        x=[1, num_iterations],
        y=[optimum] * 2,
        mode="lines",
        line={"dash": "dash", "color": rgba(optimum_color)},
        name="Optimum",
    )


def model_transitions_scatter(
    model_transitions: List[int],
    y_range: List[float],
    generator_change_color: Tuple[int] = COLORS.TEAL.value,
) -> List[go.Scatter]:
    """Creates a graph object for the line(s) representing generator changes.

    Args:
        model_transitions: iterations, before which generators
            changed
        y_range: upper and lower values of the y-range of the plot
        generator_change_color: tuple of 3 int values representing
            an RGB color. Defaults to orange.

    Returns:
        go.Scatter: plotly graph objects for the lines representing generator
            changes
    """
    if len(y_range) != 2:  # pragma: no cover
        raise ValueError("y_range should have two values, lower and upper.")
    data: List[go.Scatter] = []
    for change in model_transitions:
        data.append(
            go.Scatter(
                x=[change] * 2,
                y=y_range,
                mode="lines",
                line={"dash": "dash", "color": rgba(generator_change_color)},
                name="Generator change",
            )
        )
    return data


def optimization_trace_single_method(
    y: np.ndarray,
    optimum: Optional[float] = None,
    model_transitions: Optional[List[int]] = None,
    title: str = "",
    ylabel: str = "",
    hover_labels: Optional[List[str]] = None,
    trace_color: Tuple[int] = COLORS.STEELBLUE.value,
    optimum_color: Tuple[int] = COLORS.ORANGE.value,
    generator_change_color: Tuple[int] = COLORS.TEAL.value,
) -> AxPlotConfig:
    """Plots an optimization trace with mean and 2 SEMs

    Args:
        y: (r x t) array; result to plot, with r runs and t trials
        optimum: value of the optimal objective
        model_transitions: iterations, before which generators
            changed
        title: title for this plot.
        ylabel: label for the Y-axis.
        hover_labels: optional, text to show on hover; list where the i-th value
            corresponds to the i-th value in the value of the `y` argument.
        trace_color: tuple of 3 int values representing an RGB color.
            Defaults to orange.
        optimum_color: tuple of 3 int values representing an RGB color.
            Defaults to orange.
        generator_change_color: tuple of 3 int values representing
            an RGB color. Defaults to orange.

    Returns:
        AxPlotConfig: plot of the optimization trace with IQR
    """
    trace = mean_trace_scatter(y=y, trace_color=trace_color, hover_labels=hover_labels)
    lower, upper = sem_range_scatter(y=y, trace_color=trace_color)

    layout = go.Layout(
        title=title,
        showlegend=True,
        yaxis={"title": ylabel},
        xaxis={"title": "Iteration"},
    )

    data = [lower, trace, upper]

    if optimum is not None:
        data.append(
            optimum_objective_scatter(
                optimum=optimum, num_iterations=y.shape[1], optimum_color=optimum_color
            )
        )

    if model_transitions is not None:  # pragma: no cover
        y_lower = np.min(np.percentile(y, 25, axis=0))
        y_upper = np.max(np.percentile(y, 75, axis=0))
        if optimum is not None and optimum < y_lower:
            y_lower = optimum
        if optimum is not None and optimum > y_upper:
            y_upper = optimum
        data.extend(
            model_transitions_scatter(
                model_transitions=model_transitions,
                y_range=[y_lower, y_upper],
                generator_change_color=generator_change_color,
            )
        )

    return AxPlotConfig(
        data=go.Figure(layout=layout, data=data), plot_type=AxPlotTypes.GENERIC
    )


def optimization_trace_all_methods(
    y_dict: Dict[str, np.ndarray],
    optimum: Optional[float] = None,
    title: str = "",
    ylabel: str = "",
    hover_labels: Optional[List[str]] = None,
    trace_colors: List[Tuple[int]] = DISCRETE_COLOR_SCALE,
    optimum_color: Tuple[int] = COLORS.ORANGE.value,
) -> AxPlotConfig:
    """Plots a comparison of optimization traces with 2-SEM bands for multiple
    methods on the same problem.

    Args:
        y: a mapping of method names to (r x t) arrays, where r is the number
            of runs in the test, and t is the number of trials.
        optimum: value of the optimal objective.
        title: title for this plot.
        ylabel: label for the Y-axis.
        hover_labels: optional, text to show on hover; list where the i-th value
            corresponds to the i-th value in the value of the `y` argument.
        trace_colors: tuples of 3 int values representing
            RGB colors to use for different methods shown in the combination plot.
            Defaults to Ax discrete color scale.
        optimum_color: tuple of 3 int values representing an RGB color.
            Defaults to orange.

    Returns:
        AxPlotConfig: plot of the comparison of optimization traces with IQR
    """
    data: List[go.Scatter] = []

    for i, (method, y) in enumerate(y_dict.items()):
        # If there are more traces than colors, start reusing colors.
        color = trace_colors[i % len(trace_colors)]
        trace = mean_trace_scatter(y=y, trace_color=color, legend_label=method)
        lower, upper = sem_range_scatter(y=y, trace_color=color, legend_label=method)

        data.extend([lower, trace, upper])

    if optimum is not None:
        num_iterations = max(y.shape[1] for y in y_dict.values())
        data.append(
            optimum_objective_scatter(
                optimum=optimum,
                num_iterations=num_iterations,
                optimum_color=optimum_color,
            )
        )

    layout = go.Layout(
        title=title,
        showlegend=True,
        yaxis={"title": ylabel},
        xaxis={"title": "Iteration"},
    )

    return AxPlotConfig(
        data=go.Figure(layout=layout, data=data), plot_type=AxPlotTypes.GENERIC
    )


def optimization_times(
    fit_times: Dict[str, List[float]],
    gen_times: Dict[str, List[float]],
    title: str = "",
) -> AxPlotConfig:
    """Plots wall times for each method as a bar chart.

    Args:
        fit_times: A map from method name to a list of the model fitting times.
        gen_times: A map from method name to a list of the gen times.
        title: Title for this plot.

    Returns: AxPlotConfig with the plot
    """
    # Compute means and SEs
    methods = list(fit_times.keys())
    fit_res: Dict[str, Union[str, List[float]]] = {"name": "Fitting"}
    fit_res["mean"] = [np.mean(fit_times[m]) for m in methods]
    fit_res["2sems"] = [
        2 * np.std(fit_times[m]) / np.sqrt(len(fit_times[m])) for m in methods
    ]
    gen_res: Dict[str, Union[str, List[float]]] = {"name": "Generation"}
    gen_res["mean"] = [np.mean(gen_times[m]) for m in methods]
    gen_res["2sems"] = [
        2 * np.std(gen_times[m]) / np.sqrt(len(gen_times[m])) for m in methods
    ]
    total_mean: List[float] = []
    total_2sems: List[float] = []
    for m in methods:
        totals = np.array(fit_times[m]) + np.array(gen_times[m])
        total_mean.append(np.mean(totals))
        total_2sems.append(2 * np.std(totals) / np.sqrt(len(totals)))
    total_res: Dict[str, Union[str, List[float]]] = {
        "name": "Total",
        "mean": total_mean,
        "2sems": total_2sems,
    }

    # Construct plot
    data: List[go.Bar] = []

    for i, res in enumerate([fit_res, gen_res, total_res]):
        data.append(
            go.Bar(
                x=methods,
                y=res["mean"],
                text=res["name"],
                textposition="auto",
                error_y={"type": "data", "array": res["2sems"], "visible": True},
                marker={
                    "color": rgba(DISCRETE_COLOR_SCALE[i]),
                    "line": {"color": "rgb(0,0,0)", "width": 1.0},
                },
                opacity=0.6,
                name=res["name"],
            )
        )

    layout = go.Layout(
        title=title,
        showlegend=False,
        yaxis={"title": "Time"},
        xaxis={"title": "Method"},
    )

    return AxPlotConfig(
        data=go.Figure(layout=layout, data=data), plot_type=AxPlotTypes.GENERIC
    )

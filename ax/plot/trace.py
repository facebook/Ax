#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objs as go
from ax.core.experiment import Experiment
from ax.plot.base import AxPlotConfig, AxPlotTypes
from ax.plot.color import COLORS, DISCRETE_COLOR_SCALE, rgba
from ax.utils.common.timeutils import timestamps_in_range
from plotly import express as px
from plotly.express.colors import sample_colorscale
from pyre_extensions import none_throws

FIVE_MINUTES = timedelta(minutes=5)


# type aliases
Traces = list[dict[str, Any]]


def map_data_single_trace_scatters(
    x: npt.NDArray,
    y: npt.NDArray,
    legend_label: str,
    xlabel: str = "Trial progression",
    ylabel: str = "Trial performance",
    plot_stopping_marker: bool = False,
    opacity: float = 0.5,
    trace_color: tuple[int] = COLORS.STEELBLUE.value,
    visible: bool = True,
) -> list[go.Scatter]:
    """Plot a single trial's trace from map data.

    Args:
        x: An array of x-values for a single trace.
        y: An array of y-values for a single trace.
        legend_label: Label for this trace used in the legend.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        plot_stopping_marker: Whether to add a red early stopping
            marker for the last data point in this trace. If True,
            this function returns two go.Scatter objects, one for
            the main trace and another for the early stopping marker.
        opacity: Opacity of this trace (excluding early stopping marker).
        trace_color: Color of trace.
        visible: Whether the trace should be visible or not.
    """
    # NOTE: In the hovertemplate, we are not using float formatting for `x`
    # and `y` due to autoformatting + this allows `x` to be time data.
    scatters = [
        go.Scatter(
            name=legend_label,
            text=legend_label,
            x=x,
            y=y,
            mode="lines+markers",
            line={"color": rgba(trace_color)},
            opacity=opacity,
            hovertemplate=f"{legend_label}<br>"
            + f"{xlabel}: "
            + "%{x}<br>"
            + f"{ylabel}: "
            + "%{y}<extra></extra>",
            visible=visible,
        )
    ]
    if plot_stopping_marker:
        scatters.append(
            go.Scatter(
                text=legend_label + " stopped",
                mode="markers",
                x=x[-1:],
                y=y[-1:],
                marker={"color": "Red", "size": 10},
                showlegend=False,
                opacity=1.0,
                hovertemplate=f"{legend_label} stopped<br>"
                + f"{xlabel}: "
                + "%{x}<br>"
                + f"{ylabel}: "
                + "%{y}<extra></extra>",
                visible=visible,
            )
        )
    return scatters


def map_data_multiple_metrics_dropdown_plotly(
    title: str,
    metric_names: list[str],
    xs_by_metric: dict[str, list[npt.NDArray]],
    ys_by_metric: dict[str, list[npt.NDArray]],
    legend_labels_by_metric: dict[str, list[str]],
    stopping_markers_by_metric: dict[str, list[bool]],
    xlabels_by_metric: dict[str, str],
    lower_is_better_by_metric: dict[str, bool | None],
    opacity: float = 0.75,
    color_map: str = "viridis",
    autoset_axis_limits: bool = True,
) -> go.Figure:
    """Plot map data traces for multiple metrics, controlled by a dropdown.
    Each button in the dropdown reveals the plot for a different metric.

    Args:
        title: Title of the plot.
        metric_names: List of metric names.
        xs_by_metric: Maps metric names to a list of x-value arrays.
        ys_by_metric: Maps metric names to a list of y-value arrays.
        legend_labels_by_metric: Maps metric names to legend labels.
        stopping_markers_by_metric: Maps metric names to a list of
            boolean values indicating whether a trace should be plotted
            with a stopping marker.
        xlabels_by_metric: Maps metric names to xlabels.
        lower_is_better_by_metric: Maps metric names to `lower_is_better`
        opacity: The opacity to use when plotting traces.
        color_map: The color map for plotting different trials.
        autoset_axis_limits: Whether to automatically set axis limits.
    """
    data = []
    trace_ranges = {}  # maps metric names to range of associated traces
    layout_yaxis_ranges = {}  # maps metric names to y-axis ranges
    for i, metric_name in enumerate(metric_names):
        colors = sample_colorscale(
            colorscale=color_map,
            samplepoints=np.linspace(1.0, 0.0, len(xs_by_metric[metric_name])),
            colortype="tuple",
        )
        metric_traces = []
        for x, y, legend_label, plot_stopping_marker, color in zip(
            xs_by_metric[metric_name],
            ys_by_metric[metric_name],
            legend_labels_by_metric[metric_name],
            stopping_markers_by_metric[metric_name],
            colors,
        ):
            metric_traces.extend(
                map_data_single_trace_scatters(
                    x=x,
                    y=y,
                    xlabel=xlabels_by_metric[metric_name],
                    ylabel=metric_name,
                    legend_label=legend_label,
                    plot_stopping_marker=plot_stopping_marker,
                    opacity=opacity,
                    visible=(i == 0),
                    trace_color=color,
                )
            )
        trace_ranges[metric_name] = (len(data), len(data) + len(metric_traces))
        data.extend(metric_traces)
        lower_is_better = lower_is_better_by_metric[metric_name]
        if autoset_axis_limits and lower_is_better is not None:
            layout_yaxis_ranges[metric_name] = _autoset_axis_limits(
                y=np.concatenate(ys_by_metric[metric_name]),
                optimization_direction="minimize" if lower_is_better else "maximize",
            )
        else:
            layout_yaxis_ranges[metric_name] = None

    metric_dropdown = []
    for metric_name in metric_names:
        is_visible = [False] * len(data)
        metric_start, metric_end = trace_ranges[metric_name]
        is_visible[metric_start:metric_end] = [True] * (metric_end - metric_start)
        metric_dropdown.append(
            {
                "args": [
                    {"visible": is_visible},
                    {
                        "yaxis.range": layout_yaxis_ranges[metric_name],
                        "yaxis.title": metric_name,
                        "xaxis.title": xlabels_by_metric[metric_name],
                    },
                ],
                "label": metric_name,
                "method": "update",
            }
        )
    layout = go.Layout(
        title=title,
        showlegend=True,
        yaxis={"title": metric_names[0]},
        xaxis={"title": xlabels_by_metric[metric_names[0]]},
        updatemenus=[
            {
                "active": 0,
                "buttons": metric_dropdown,
                "yanchor": "top",
                "xanchor": "left",
                "x": 0,
                "y": 1.125,
            },
        ],
    )
    return go.Figure(
        layout=layout,
        data=data,
        layout_yaxis_range=layout_yaxis_ranges[metric_names[0]],
    )


def mean_trace_scatter(
    y: npt.NDArray,
    trace_color: tuple[int] = COLORS.STEELBLUE.value,
    legend_label: str = "mean",
    hover_labels: list[str] | None = None,
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
    y: npt.NDArray,
    trace_color: tuple[int] = COLORS.STEELBLUE.value,
    legend_label: str = "",
) -> tuple[go.Scatter, go.Scatter]:
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
    y[np.isinf(y)] = np.nan  # avoids RuntimeWarning in np.std due to invalid values
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


def mean_markers_scatter(
    y: npt.NDArray,
    marker_color: tuple[int] = COLORS.LIGHT_PURPLE.value,
    legend_label: str = "",
    hover_labels: list[str] | None = None,
) -> go.Scatter:
    """Creates a graph object for trace of the mean of the given series across
    runs, with errorbars.

    Args:
        y: (r x t) array with results from  r runs and t trials.
        trace_color: tuple of 3 int values representing an RGB color.
            Defaults to light purple.
        legend_label: label for this trace.
        hover_labels: optional, text to show on hover; list where the i-th value
            corresponds to the i-th value in the value of the `y` argument.

    Returns:
        go.Scatter: plotly graph object
    """
    mean = np.mean(y, axis=0)
    sem = np.std(y, axis=0) / np.sqrt(y.shape[0])
    return go.Scatter(
        name=legend_label,
        x=np.arange(1, y.shape[1] + 1),
        y=mean,
        error_y={
            "type": "data",
            "array": sem,
            "visible": True,
        },
        mode="markers",
        marker={"color": rgba(marker_color)},
        text=hover_labels,
    )


def optimum_objective_scatter(
    optimum: float, num_iterations: int, optimum_color: tuple[int] = COLORS.ORANGE.value
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


def optimization_trace_single_method_plotly(
    y: npt.NDArray,
    optimum: float | None = None,
    model_transitions: list[int] | None = None,
    title: str = "",
    ylabel: str = "",
    hover_labels: list[str] | None = None,
    trace_color: tuple[int] = COLORS.STEELBLUE.value,
    optimum_color: tuple[int] = COLORS.ORANGE.value,
    generator_change_color: tuple[int] = COLORS.TEAL.value,
    optimization_direction: str | None = "passthrough",
    plot_trial_points: bool = False,
    trial_points_color: tuple[int] = COLORS.LIGHT_PURPLE.value,
    autoset_axis_limits: bool = True,
) -> go.Figure:
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
        trace_color: tuple of 3 int values representing an RGB color for plotting
            running optimum. Defaults to blue.
        optimum_color: tuple of 3 int values representing an RGB color.
            Defaults to orange.
        generator_change_color: tuple of 3 int values representing
            an RGB color. Defaults to teal.
        optimization_direction: str, "minimize" will plot running minimum,
            "maximize" will plot running maximum, "passthrough" (default) will plot
            y as lines, None does not plot running optimum)
        plot_trial_points: bool, whether to plot the objective for each trial, as
            supplied in y (default False for backward compatibility)
        trial_points_color: tuple of 3 int values representing an RGB color for
            plotting trial points. Defaults to light purple.
        autoset_axis_limits: Automatically try to set the limit for each axis to focus
            on the region of interest.

    Returns:
        go.Figure: plot of the optimization trace with IQR
    """
    if optimization_direction not in {"minimize", "maximize", "passthrough", None}:
        raise ValueError(
            'optimization_direction must be "minimize", "maximize", "passthrough", or '
            "None"
        )
    if (not plot_trial_points) and (optimization_direction is None):
        raise ValueError(
            "If plot_trial_points is False, optimization_direction must not be None."
        )
    data = []
    if plot_trial_points:
        markers = mean_markers_scatter(
            y=y,
            marker_color=trial_points_color,
            hover_labels=hover_labels,
            legend_label="objective value",
        )
        data.extend([markers])

    if optimization_direction is not None:
        legend_label = "best objective so far"
        if optimization_direction == "minimize":
            y_running_optimum = np.minimum.accumulate(y, axis=1)
        elif optimization_direction == "maximize":
            y_running_optimum = np.maximum.accumulate(y, axis=1)
        else:
            y_running_optimum = y
            legend_label = "objective value"
        trace = mean_trace_scatter(
            y=y_running_optimum,
            trace_color=trace_color,
            hover_labels=hover_labels,
            legend_label=legend_label,
        )
        lower, upper = sem_range_scatter(y=y_running_optimum, trace_color=trace_color)
        data.extend([lower, trace, upper])

    if optimum is not None:
        data.append(
            optimum_objective_scatter(
                optimum=optimum, num_iterations=y.shape[1], optimum_color=optimum_color
            )
        )

    layout = go.Layout(
        title=title,
        showlegend=True,
        yaxis={"title": ylabel},
        xaxis={"title": "Iteration"},
    )
    layout_yaxis_range = None
    if autoset_axis_limits and optimization_direction in ["minimize", "maximize"]:
        layout_yaxis_range = _autoset_axis_limits(
            y=y, optimization_direction=optimization_direction
        )
    return go.Figure(layout=layout, data=data, layout_yaxis_range=layout_yaxis_range)


def _autoset_axis_limits(
    y: npt.NDArray,
    optimization_direction: str,
    force_include_value: float | None = None,
) -> list[float]:
    """Provides automatic axis limits based on the data and optimization direction.
    All best points are included in this range, and by default the worst points are
    truncated at some distance below the median, where that distance is given by
    1.5 * (the distance between the median and the best quartile).

    If `force_include_value` is provided, the worst points will be truncated at this
    value if it is worse than the truncation point described above.
    """
    q1 = np.percentile(y, q=25, method="lower").min()
    q2_min = np.percentile(y, q=50, method="linear").min()
    q2_max = np.percentile(y, q=50, method="linear").max()
    q3 = np.percentile(y, q=75, method="higher").max()
    if optimization_direction == "minimize":
        y_lower = y.min()
        y_upper = q2_max + 1.5 * (q2_max - q1)
        if force_include_value is not None:
            y_upper = max(y_upper, force_include_value)
    else:
        y_lower = q2_min - 1.5 * (q3 - q2_min)
        y_upper = y.max()
        if force_include_value is not None:
            y_lower = min(y_lower, force_include_value)
    y_padding = 0.1 * (y_upper - y_lower)
    y_lower, y_upper = y_lower - y_padding, y_upper + y_padding
    return [y_lower, y_upper]


def optimization_trace_single_method(
    y: npt.NDArray,
    optimum: float | None = None,
    model_transitions: list[int] | None = None,
    title: str = "",
    ylabel: str = "",
    hover_labels: list[str] | None = None,
    trace_color: tuple[int] = COLORS.STEELBLUE.value,
    optimum_color: tuple[int] = COLORS.ORANGE.value,
    generator_change_color: tuple[int] = COLORS.TEAL.value,
    optimization_direction: str | None = "passthrough",
    plot_trial_points: bool = False,
    trial_points_color: tuple[int] = COLORS.LIGHT_PURPLE.value,
    autoset_axis_limits: bool = True,
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
        trace_color: tuple of 3 int values representing an RGB color for plotting
            running optimum. Defaults to blue.
        optimum_color: tuple of 3 int values representing an RGB color.
            Defaults to orange.
        generator_change_color: tuple of 3 int values representing
            an RGB color. Defaults to teal.
        optimization_direction: str, "minimize" will plot running minimum,
            "maximize" will plot running maximum, "passthrough" (default) will plot
            y as lines, None does not plot running optimum)
        plot_trial_points: bool, whether to plot the objective for each trial, as
            supplied in y (default False for backward compatibility)
        trial_points_color: tuple of 3 int values representing an RGB color for
            plotting trial points. Defaults to light purple.
        autoset_axis_limits: Automatically try to set the limit for each axis to focus
            on the region of interest.

    Returns:
        AxPlotConfig: plot of the optimization trace with IQR
    """
    return AxPlotConfig(
        # pyre-fixme[6]: For 1st argument expected `Dict[str, typing.Any]` but got
        #  `Figure`.
        data=optimization_trace_single_method_plotly(
            y=y,
            optimum=optimum,
            model_transitions=model_transitions,
            title=title,
            ylabel=ylabel,
            hover_labels=hover_labels,
            trace_color=trace_color,
            optimum_color=optimum_color,
            generator_change_color=generator_change_color,
            optimization_direction=optimization_direction,
            plot_trial_points=plot_trial_points,
            trial_points_color=trial_points_color,
            autoset_axis_limits=autoset_axis_limits,
        ),
        plot_type=AxPlotTypes.GENERIC,
    )


def optimization_trace_all_methods(
    y_dict: dict[str, npt.NDArray],
    optimum: float | None = None,
    title: str = "",
    ylabel: str = "",
    hover_labels: list[str] | None = None,
    trace_colors: list[tuple[int]] = DISCRETE_COLOR_SCALE,
    optimum_color: tuple[int] = COLORS.ORANGE.value,
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
    data: list[go.Scatter] = []

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
        # pyre-fixme[6]: For 1st argument expected `Dict[str, typing.Any]` but got
        #  `Figure`.
        data=go.Figure(layout=layout, data=data),
        plot_type=AxPlotTypes.GENERIC,
    )


def optimization_times(
    fit_times: dict[str, list[float]],
    gen_times: dict[str, list[float]],
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
    fit_res: dict[str, str | list[float]] = {"name": "Fitting"}
    fit_res["mean"] = [np.mean(fit_times[m]) for m in methods]
    fit_res["2sems"] = [
        # pyre-fixme[58]: `*` is not supported for operand types `int` and
        #  `floating[typing.Any]`.
        2 * np.std(fit_times[m]) / np.sqrt(len(fit_times[m]))
        for m in methods
    ]
    gen_res: dict[str, str | list[float]] = {"name": "Generation"}
    gen_res["mean"] = [np.mean(gen_times[m]) for m in methods]
    gen_res["2sems"] = [
        # pyre-fixme[58]: `*` is not supported for operand types `int` and
        #  `floating[typing.Any]`.
        2 * np.std(gen_times[m]) / np.sqrt(len(gen_times[m]))
        for m in methods
    ]
    total_mean: list[float] = []
    total_2sems: list[float] = []
    for m in methods:
        totals = np.array(fit_times[m]) + np.array(gen_times[m])
        total_mean.append(np.mean(totals))
        # pyre-fixme[58]: `*` is not supported for operand types `int` and
        #  `floating[typing.Any]`.
        total_2sems.append(2 * np.std(totals) / np.sqrt(len(totals)))
    total_res: dict[str, str | list[float]] = {
        "name": "Total",
        "mean": total_mean,
        "2sems": total_2sems,
    }

    # Construct plot
    data: list[go.Bar] = []

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
        # pyre-fixme[6]: For 1st argument expected `Dict[str, typing.Any]` but got
        #  `Figure`.
        data=go.Figure(layout=layout, data=data),
        plot_type=AxPlotTypes.GENERIC,
    )


def get_running_trials_per_minute(
    experiment: Experiment,
    show_until_latest_end_plus_timedelta: timedelta = FIVE_MINUTES,
) -> AxPlotConfig:
    trial_runtimes: list[tuple[int, datetime, datetime | None]] = [
        (
            trial.index,
            none_throws(trial._time_run_started),
            trial._time_completed,  # Time trial was completed, failed, or abandoned.
        )
        for trial in experiment.trials.values()
        if trial._time_run_started is not None
    ]

    earliest_start = min(tr[1] for tr in trial_runtimes)
    latest_end = max(none_throws(tr[2]) for tr in trial_runtimes if tr[2] is not None)

    running_during = {
        ts: [
            t[0]  # Trial index.
            for t in trial_runtimes
            # Trial is running during a given timestamp if:
            # 1) it's run start time is at/before the timestamp,
            # 2) it's completion time has not yet come or is after the timestamp.
            if t[1] <= ts and (True if t[2] is None else none_throws(t[2]) >= ts)
        ]
        for ts in timestamps_in_range(
            earliest_start,
            latest_end + show_until_latest_end_plus_timedelta,
            timedelta(seconds=60),
        )
    }

    num_running_at_ts = {ts: len(trials) for ts, trials in running_during.items()}

    scatter = go.Scatter(
        x=list(num_running_at_ts.keys()),
        y=[num_running_at_ts[ts] for ts in num_running_at_ts],
    )

    return AxPlotConfig(
        # pyre-fixme[6]: For 1st argument expected `Dict[str, typing.Any]` but got
        #  `Figure`.
        data=go.Figure(
            layout=go.Layout(title="Number of running trials during experiment"),
            data=[scatter],
        ),
        plot_type=AxPlotTypes.GENERIC,
    )


def plot_objective_value_vs_trial_index(
    exp_df: pd.DataFrame,
    metric_colname: str,
    minimize: bool,
    title: str | None = None,
    hover_data_colnames: list[str] | None = None,
    autoset_axis_limits: bool = True,
) -> go.Figure:
    """Returns a plotly figure showing the optimization trace for a single metric.

    Args:
        exp_df: DataFrame with the following columns
            - "trial_index": Index of each trial.
            - "arm_name": Name of each arm evaluated in the corresponding trial.
            - metric_colname: Name of the objective metric (user-provided).
            - "is_feasible": Whether each arm is feasible (optional). If not
                provided, all arms will be considered feasible.
            - "generation_method": Generation method used to generate each arm
                (optional).
            - hover_data_colnames: Columns to be displayed on hover (user-provided).
        metric_colname: Name of the column in exp_df that contains the
            objective metric values.
        minimize: Optimization direction of the objective.
        title: Title of the plot (optional).
        hover_data_colnames: Names of additional columns to display on hover.
        autoset_axis_limits: Automatically try to set the limit for each axis to focus
            on the region of interest. Will always include first point.
    Returns:
        Optimization trace as a plot.
    """
    # Protect input exp_df from changes.
    exp_df = exp_df.copy()

    # Use completed trials only.
    trial_status_colname = "trial_status"
    if trial_status_colname in exp_df.columns:
        exp_df = exp_df.loc[exp_df[trial_status_colname].str.match("COMPLETED")]

    # Check if feasibility and generation method columns exist.
    is_feasible_colname = "is_feasible" if "is_feasible" in exp_df.columns else None
    generation_method_colname = (
        "generation_method" if "generation_method" in exp_df.columns else None
    )

    scatter = px.scatter(
        data_frame=exp_df,
        x="trial_index",
        y=metric_colname,
        color=is_feasible_colname,
        symbol=generation_method_colname,
        hover_name="arm_name",
        hover_data=hover_data_colnames,
    )
    running_feasible_optimum_df = compute_running_feasible_optimum_df(
        exp_df=exp_df,
        metric_colname=metric_colname,
        minimize=minimize,
        is_feasible_colname=is_feasible_colname,
    )
    line = px.line(
        data_frame=running_feasible_optimum_df,
        x="trial_index",
        y="running_optimum",
        color="Legend",
        line_shape="hv",
    )
    fig = scatter.add_trace(line.data[0])
    if autoset_axis_limits:
        layout_yaxis_range = _autoset_axis_limits(
            y=exp_df[metric_colname].to_numpy(),
            optimization_direction="minimize" if minimize else "maximize",
            force_include_value=running_feasible_optimum_df.loc[0, metric_colname],
        )
        fig.update_layout(yaxis_range=layout_yaxis_range)
    if title is not None:
        fig.update_layout(title=title)
    return fig


def compute_running_feasible_optimum_df(
    exp_df: pd.DataFrame,
    metric_colname: str,
    minimize: bool,
    is_feasible_colname: str | None,
) -> pd.DataFrame:
    """Computes the running feasible optimum for a given metric."""
    # If feasibility column is not provided, assume all arms are feasible.
    if is_feasible_colname is None:
        running_feasible_optimum_df = exp_df
    else:
        running_feasible_optimum_df = exp_df[
            exp_df["is_feasible"].notnull() & exp_df["is_feasible"]
        ]
    running_feasible_optimum_df = running_feasible_optimum_df[
        ["trial_index", metric_colname]
    ].copy()

    # Compute running optimum.
    if minimize:
        running_feasible_optimum_df = (
            running_feasible_optimum_df.groupby("trial_index").min().reset_index()
        )
        running_feasible_optimum_df["running_optimum"] = running_feasible_optimum_df[
            metric_colname
        ].cummin()
    else:
        running_feasible_optimum_df = (
            running_feasible_optimum_df.groupby("trial_index").max().reset_index()
        )
        running_feasible_optimum_df["running_optimum"] = running_feasible_optimum_df[
            metric_colname
        ].cummax()

    # Infill missing/infeasible trials.
    available_trial_indices = exp_df[exp_df[metric_colname].notnull()]["trial_index"]
    new_index = pd.Index(
        np.arange(
            start=available_trial_indices.min(),
            stop=available_trial_indices.max() + 1,
        ),
        name="trial_index",
    )
    running_feasible_optimum_df = (
        running_feasible_optimum_df.set_index("trial_index")
        .reindex(new_index)
        .reset_index()
        .fillna(method="ffill")
    )

    # Add legend column.
    running_feasible_optimum_df["Legend"] = "Running optimum"
    return running_feasible_optimum_df

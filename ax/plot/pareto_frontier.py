#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objs as go
from ax.core.experiment import Experiment
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.exceptions.core import UserInputError
from ax.plot.base import CI_OPACITY, DECIMALS, AxPlotConfig, AxPlotTypes
from ax.plot.color import COLORS, rgba, DISCRETE_COLOR_SCALE
from ax.plot.helper import extend_range, _format_CI, _format_dict
from ax.plot.pareto_utils import ParetoFrontierResults
from ax.utils.common.typeutils import checked_cast, not_none
from scipy.stats import norm


DEFAULT_CI_LEVEL: float = 0.9
VALID_CONSTRAINT_OP_NAMES = {"GEQ", "LEQ"}


def _make_label(
    mean: float, sem: float, name: str, is_relative: bool, Z: Optional[float]
) -> str:
    estimate = str(round(mean, DECIMALS))
    perc = "%" if is_relative else ""
    ci = (
        ""
        if (Z is None or np.isnan(sem))
        else _format_CI(estimate=mean, sd=sem, relative=is_relative, zval=Z)
    )
    return f"{name}: {estimate}{perc} {ci}<br>"


def _filter_outliers(Y: np.ndarray, m: float = 2.0) -> np.ndarray:
    std_filter = abs(Y - np.median(Y, axis=0)) < m * np.std(Y, axis=0)
    return Y[np.all(abs(std_filter), axis=1)]


def scatter_plot_with_pareto_frontier_plotly(
    Y: np.ndarray,
    Y_pareto: np.ndarray,
    metric_x: str,
    metric_y: str,
    reference_point: Tuple[float, float],
    minimize: Union[bool, Tuple[bool, bool]] = True,
) -> go.Figure:
    """Plots a scatter of all points in ``Y`` for ``metric_x`` and ``metric_y``
    with a reference point and Pareto frontier from ``Y_pareto``.

    Points in the scatter are colored in a gradient representing their trial index,
    with metric_x on x-axis and metric_y on y-axis. Reference point is represented
    as a star and Pareto frontier –– as a line. The frontier connects to the reference
    point via projection lines.

    NOTE: Both metrics should have the same minimization setting, passed as `minimize`.

    Args:
        Y: Array of outcomes, of which the first two will be plotted.
        Y_pareto: Array of Pareto-optimal points, first two outcomes in which will be
            plotted.
        metric_x: Name of first outcome in ``Y``.
        metric_Y: Name of second outcome in ``Y``.
        reference_point: Reference point for ``metric_x`` and ``metric_y``.
        minimize: Whether the two metrics in the plot are being minimized or maximized.
    """
    if isinstance(minimize, bool):
        minimize = (minimize, minimize)
    Xs = Y[:, 0]
    Ys = Y[:, 1]

    experimental_points_scatter = go.Scatter(
        x=Xs,
        y=Ys,
        mode="markers",
        marker={
            "color": np.linspace(0, 100, int(len(Xs) * 1.05)),
            "colorscale": "magma",
            "colorbar": {
                "tickvals": [0, 50, 100],
                "ticktext": [
                    1,
                    "iteration",
                    len(Xs),
                ],
            },
        },
        name="Experimental points",
    )
    reference_point_star = go.Scatter(
        x=[reference_point[0]],
        y=[reference_point[1]],
        mode="markers",
        marker={"color": rgba(COLORS.STEELBLUE.value), "size": 25, "symbol": "star"},
    )
    extra_point_x = min(Y_pareto[:, 0]) if minimize[0] else max(Y_pareto[:, 0])
    reference_point_line_1 = go.Scatter(
        x=[extra_point_x, reference_point[0]],
        y=[reference_point[1], reference_point[1]],
        mode="lines",
        marker={"color": rgba(COLORS.STEELBLUE.value)},
    )
    extra_point_y = min(Y_pareto[:, 1]) if minimize[1] else max(Y_pareto[:, 1])
    reference_point_line_2 = go.Scatter(
        x=[reference_point[0], reference_point[0]],
        y=[extra_point_y, reference_point[1]],
        mode="lines",
        marker={"color": rgba(COLORS.STEELBLUE.value)},
    )
    Y_pareto_with_extra = np.concatenate(
        (
            [[extra_point_x, reference_point[1]]],
            Y_pareto,
            [[reference_point[0], extra_point_y]],
        ),
        axis=0,
    )
    pareto_step = go.Scatter(
        x=Y_pareto_with_extra[:, 0],
        y=Y_pareto_with_extra[:, 1],
        mode="lines",
        line_shape="hv",
        marker={"color": rgba(COLORS.STEELBLUE.value)},
    )

    range_x = (
        extend_range(lower=min(Y_pareto[:, 0]), upper=reference_point[0])
        if minimize[0]
        else extend_range(lower=reference_point[0], upper=max(Y_pareto[:, 0]))
    )
    range_y = (
        extend_range(lower=min(Y_pareto[:, 1]), upper=reference_point[1])
        if minimize[1]
        else extend_range(lower=reference_point[1], upper=max(Y_pareto[:, 1]))
    )
    layout = go.Layout(
        title="Observed points with Pareto frontier",
        showlegend=False,
        xaxis={"title": metric_x, "range": range_x},
        yaxis={"title": metric_y, "range": range_y},
    )
    return go.Figure(
        layout=layout,
        data=[
            pareto_step,
            reference_point_line_1,
            reference_point_line_2,
            experimental_points_scatter,
            reference_point_star,
        ],
    )


def scatter_plot_with_pareto_frontier(
    Y: np.ndarray,
    Y_pareto: np.ndarray,
    metric_x: str,
    metric_y: str,
    reference_point: Tuple[float, float],
    minimize: bool = True,
) -> AxPlotConfig:
    return AxPlotConfig(
        data=scatter_plot_with_pareto_frontier_plotly(
            Y=Y,
            Y_pareto=Y_pareto,
            metric_x=metric_x,
            metric_y=metric_y,
            reference_point=reference_point,
        ),
        plot_type=AxPlotTypes.GENERIC,
    )


def _get_single_pareto_trace(
    frontier: ParetoFrontierResults,
    CI_level: float,
    legend_label: str = "mean",
    trace_color: Tuple[int] = COLORS.STEELBLUE.value,
    show_parameterization_on_hover: bool = True,
) -> go.Scatter:
    primary_means = frontier.means[frontier.primary_metric]
    primary_sems = frontier.sems[frontier.primary_metric]
    secondary_means = frontier.means[frontier.secondary_metric]
    secondary_sems = frontier.sems[frontier.secondary_metric]
    absolute_metrics = frontier.absolute_metrics
    all_metrics = frontier.means.keys()
    if frontier.arm_names is None:
        arm_names = [f"Parameterization {i}" for i in range(len(frontier.param_dicts))]
    else:
        arm_names = [f"Arm {name}" for name in frontier.arm_names]

    if CI_level is not None:
        Z = 0.5 * norm.ppf(1 - (1 - CI_level) / 2)
    else:
        Z = None

    labels = []

    for i, param_dict in enumerate(frontier.param_dicts):
        label = f"<b>{arm_names[i]}</b><br>"
        for metric in all_metrics:
            metric_lab = _make_label(
                mean=frontier.means[metric][i],
                sem=frontier.sems[metric][i],
                name=metric,
                is_relative=metric not in absolute_metrics,
                Z=Z,
            )
            label += metric_lab

        parameterization = (
            _format_dict(param_dict, "Parameterization")
            if show_parameterization_on_hover
            else ""
        )
        label += parameterization
        labels.append(label)
    return go.Scatter(
        name=legend_label,
        legendgroup=legend_label,
        x=secondary_means,
        y=primary_means,
        error_x={
            "type": "data",
            "array": Z * np.array(secondary_sems),
            "thickness": 2,
            "color": rgba(trace_color, CI_OPACITY),
        },
        error_y={
            "type": "data",
            "array": Z * np.array(primary_sems),
            "thickness": 2,
            "color": rgba(trace_color, CI_OPACITY),
        },
        mode="markers",
        text=labels,
        hoverinfo="text",
        marker={"color": rgba(trace_color)},
    )


def plot_pareto_frontier(
    frontier: ParetoFrontierResults,
    CI_level: float = DEFAULT_CI_LEVEL,
    show_parameterization_on_hover: bool = True,
) -> AxPlotConfig:
    """Plot a Pareto frontier from a ParetoFrontierResults object.

    Args:
        frontier (ParetoFrontierResults): The results of the Pareto frontier
            computation.
        CI_level (float, optional): The confidence level, i.e. 0.95 (95%)
        show_parameterization_on_hover (bool, optional): If True, show the
            parameterization of the points on the frontier on hover.

    Returns:
        AEPlotConfig: The resulting Plotly plot definition.

    """
    trace = _get_single_pareto_trace(
        frontier=frontier,
        CI_level=CI_level,
        show_parameterization_on_hover=show_parameterization_on_hover,
    )

    shapes = []
    primary_threshold = None
    secondary_threshold = None
    if frontier.objective_thresholds is not None:
        primary_threshold = frontier.objective_thresholds.get(
            frontier.primary_metric, None
        )
        secondary_threshold = frontier.objective_thresholds.get(
            frontier.secondary_metric, None
        )
    absolute_metrics = frontier.absolute_metrics
    rel_x = frontier.secondary_metric not in absolute_metrics
    rel_y = frontier.primary_metric not in absolute_metrics
    if primary_threshold is not None:
        shapes.append(
            {
                "type": "line",
                "xref": "paper",
                "x0": 0.0,
                "x1": 1.0,
                "yref": "y",
                "y0": primary_threshold,
                "y1": primary_threshold,
                "line": {"color": rgba(COLORS.CORAL.value), "width": 3},
            }
        )
    if secondary_threshold is not None:
        shapes.append(
            {
                "type": "line",
                "yref": "paper",
                "y0": 0.0,
                "y1": 1.0,
                "xref": "x",
                "x0": secondary_threshold,
                "x1": secondary_threshold,
                "line": {"color": rgba(COLORS.CORAL.value), "width": 3},
            }
        )

    layout = go.Layout(
        title="Pareto Frontier",
        xaxis={
            "title": frontier.secondary_metric,
            "ticksuffix": "%" if rel_x else "",
            "zeroline": True,
        },
        yaxis={
            "title": frontier.primary_metric,
            "ticksuffix": "%" if rel_y else "",
            "zeroline": True,
        },
        hovermode="closest",
        legend={"orientation": "h"},
        width=750,
        height=500,
        margin=go.layout.Margin(pad=4, l=225, b=75, t=75),  # noqa E741
        shapes=shapes,
    )

    fig = go.Figure(data=[trace], layout=layout)
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)


def plot_multiple_pareto_frontiers(
    frontiers: Dict[str, ParetoFrontierResults],
    CI_level: float = DEFAULT_CI_LEVEL,
    show_parameterization_on_hover: bool = True,
) -> AxPlotConfig:
    """Plot a Pareto frontier from a ParetoFrontierResults object.

    Args:
        frontiers (Dict[str, ParetoFrontierResults]): The results of
            the Pareto frontier computation.
        CI_level (float, optional): The confidence level, i.e. 0.95 (95%)
        show_parameterization_on_hover (bool, optional): If True, show the
            parameterization of the points on the frontier on hover.

    Returns:
        AEPlotConfig: The resulting Plotly plot definition.

    """
    first_frontier = list(frontiers.values())[0]
    traces = []
    for i, (method, frontier) in enumerate(frontiers.items()):
        # Check the two metrics are the same as the first frontier
        if (
            frontier.primary_metric != first_frontier.primary_metric
            or frontier.secondary_metric != first_frontier.secondary_metric
        ):
            raise ValueError("All frontiers should have the same pairs of metrics.")

        trace = _get_single_pareto_trace(
            frontier=frontier,
            legend_label=method,
            trace_color=DISCRETE_COLOR_SCALE[i % len(DISCRETE_COLOR_SCALE)],
            CI_level=CI_level,
            show_parameterization_on_hover=show_parameterization_on_hover,
        )

        traces.append(trace)

    shapes = []
    primary_threshold = None
    secondary_threshold = None
    if frontier.objective_thresholds is not None:
        primary_threshold = frontier.objective_thresholds.get(
            frontier.primary_metric, None
        )
        secondary_threshold = frontier.objective_thresholds.get(
            frontier.secondary_metric, None
        )
    absolute_metrics = frontier.absolute_metrics
    rel_x = frontier.secondary_metric not in absolute_metrics
    rel_y = frontier.primary_metric not in absolute_metrics
    if primary_threshold is not None:
        shapes.append(
            {
                "type": "line",
                "xref": "paper",
                "x0": 0.0,
                "x1": 1.0,
                "yref": "y",
                "y0": primary_threshold,
                "y1": primary_threshold,
                "line": {"color": rgba(COLORS.CORAL.value), "width": 3},
            }
        )
    if secondary_threshold is not None:
        shapes.append(
            {
                "type": "line",
                "yref": "paper",
                "y0": 0.0,
                "y1": 1.0,
                "xref": "x",
                "x0": secondary_threshold,
                "x1": secondary_threshold,
                "line": {"color": rgba(COLORS.CORAL.value), "width": 3},
            }
        )

    layout = go.Layout(
        title="Pareto Frontier",
        xaxis={
            "title": frontier.secondary_metric,
            "ticksuffix": "%" if rel_x else "",
            "zeroline": True,
        },
        yaxis={
            "title": frontier.primary_metric,
            "ticksuffix": "%" if rel_y else "",
            "zeroline": True,
        },
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.20,
            "xanchor": "auto",
            "x": 0.075,
        },
        width=750,
        height=550,
        margin=go.layout.Margin(pad=4, l=225, b=125, t=75),  # noqa E741
        shapes=shapes,
    )

    fig = go.Figure(data=traces, layout=layout)
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)


def interact_pareto_frontier(
    frontier_list: List[ParetoFrontierResults],
    CI_level: float = DEFAULT_CI_LEVEL,
    show_parameterization_on_hover: bool = True,
) -> AxPlotConfig:
    """Plot a pareto frontier from a list of objects"""
    if not frontier_list:
        raise ValueError("Must receive a non-empty list of pareto frontiers to plot.")

    traces = []
    shapes = []
    for frontier in frontier_list:
        config = plot_pareto_frontier(
            frontier=frontier,
            CI_level=CI_level,
            show_parameterization_on_hover=show_parameterization_on_hover,
        )
        traces.append(config.data["data"][0])
        shapes.append(config.data["layout"].get("shapes", []))

    for i, trace in enumerate(traces):
        if i == 0:  # Only the first trace is initially set to visible
            trace["visible"] = True
        else:  # All other plot traces are not visible initially
            trace["visible"] = False

    # TODO (jej): replace dropdown with two dropdowns, one for x one for y.
    dropdown = []
    for i, frontier in enumerate(frontier_list):
        trace_cnt = 1
        # Only one plot trace is visible at a given time.
        visible = [False] * (len(frontier_list) * trace_cnt)
        for j in range(i * trace_cnt, (i + 1) * trace_cnt):
            visible[j] = True
        rel_y = frontier.primary_metric not in frontier.absolute_metrics
        rel_x = frontier.secondary_metric not in frontier.absolute_metrics
        primary_metric = frontier.primary_metric
        secondary_metric = frontier.secondary_metric
        dropdown.append(
            {
                "method": "update",
                "args": [
                    {"visible": visible, "method": "restyle"},
                    {
                        "yaxis.title": primary_metric,
                        "xaxis.title": secondary_metric,
                        "yaxis.ticksuffix": "%" if rel_y else "",
                        "xaxis.ticksuffix": "%" if rel_x else "",
                        "shapes": shapes[i],
                    },
                ],
                "label": f"{primary_metric} vs {secondary_metric}",
            }
        )

    # Set initial layout arguments.
    initial_frontier = frontier_list[0]
    rel_x = initial_frontier.secondary_metric not in initial_frontier.absolute_metrics
    rel_y = initial_frontier.primary_metric not in initial_frontier.absolute_metrics
    secondary_metric = initial_frontier.secondary_metric
    primary_metric = initial_frontier.primary_metric

    layout = go.Layout(
        title="Pareto Frontier",
        xaxis={
            "title": secondary_metric,
            "ticksuffix": "%" if rel_x else "",
            "zeroline": True,
        },
        yaxis={
            "title": primary_metric,
            "ticksuffix": "%" if rel_y else "",
            "zeroline": True,
        },
        updatemenus=[
            {
                "buttons": dropdown,
                "x": 0.075,
                "xanchor": "left",
                "y": 1.1,
                "yanchor": "middle",
            }
        ],
        hovermode="closest",
        legend={"orientation": "h"},
        width=750,
        height=500,
        margin=go.layout.Margin(pad=4, l=225, b=75, t=75),  # noqa E741
        shapes=shapes[0],
    )

    fig = go.Figure(data=traces, layout=layout)
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)


def interact_multiple_pareto_frontier(
    frontier_lists: Dict[str, List[ParetoFrontierResults]],
    CI_level: float = DEFAULT_CI_LEVEL,
    show_parameterization_on_hover: bool = True,
) -> AxPlotConfig:
    """Plot a Pareto frontiers from a list of lists of NamedParetoFrontierResults objects
    that we want to compare.

    Args:
        frontier_lists (Dict[List[ParetoFrontierResults]]): A dictionary of multiple
            lists of Pareto frontier computation results to plot for comparison.
            Each list of ParetoFrontierResults contains a list of the results of
            the same pareto frontier but under different pairs of metrics.
            Different List[ParetoFrontierResults] must contain the the same pairs
            of metrics for this function to work.
        CI_level (float, optional): The confidence level, i.e. 0.95 (95%)
        show_parameterization_on_hover (bool, optional): If True, show the
            parameterization of the points on the frontier on hover.

    Returns:
        AEPlotConfig: The resulting Plotly plot definition.

    """
    if not frontier_lists:
        raise ValueError("Must receive a non-empty list of pareto frontiers to plot.")

    # Check all the lists have the same length
    vals = frontier_lists.values()
    length = len(frontier_lists[next(iter(frontier_lists))])
    if not all(len(item) == length for item in vals):
        raise ValueError("Not all lists in frontier_lists have the same length.")

    # Transform the frontier_lists to lists of frontiers where each list
    # corresponds to one pair of metrics with multiple frontiers
    list_of_frontiers = [
        dict(zip(frontier_lists.keys(), t)) for t in zip(*frontier_lists.values())
    ]
    # Get the traces and shapes for plotting
    traces = []
    shapes = []
    for frontiers in list_of_frontiers:
        config = plot_multiple_pareto_frontiers(
            frontiers=frontiers,
            CI_level=CI_level,
            show_parameterization_on_hover=show_parameterization_on_hover,
        )
        for i in range(len(config.data["data"])):
            traces.append(config.data["data"][i])
        shapes.append(config.data["layout"].get("shapes", []))

    num_frontiers = len(frontier_lists)
    num_metric_pairs = len(list_of_frontiers)
    for i, trace in enumerate(traces):
        if (
            i < num_frontiers
        ):  # Only the traces for metric 1 v.s. metric 2 are initially set to visible
            trace["visible"] = True
        else:  # All other plot traces are not visible initially
            trace["visible"] = False

    dropdown = []
    for i, frontiers in enumerate(list_of_frontiers):
        # Only plot traces for the current pair of metrics are visible at a given time.
        visible = [False] * (num_metric_pairs * num_frontiers)
        for j in range(i * num_frontiers, (i + 1) * num_frontiers):
            visible[j] = True
        # Get the first frontier for reference of metric names
        first_frontier = list(frontiers.values())[0]
        rel_y = first_frontier.primary_metric not in first_frontier.absolute_metrics
        rel_x = first_frontier.secondary_metric not in first_frontier.absolute_metrics
        primary_metric = first_frontier.primary_metric
        secondary_metric = first_frontier.secondary_metric
        dropdown.append(
            {
                "method": "update",
                "args": [
                    {"visible": visible, "method": "restyle"},
                    {
                        "yaxis.title": primary_metric,
                        "xaxis.title": secondary_metric,
                        "yaxis.ticksuffix": "%" if rel_y else "",
                        "xaxis.ticksuffix": "%" if rel_x else "",
                        "shapes": shapes[i],
                    },
                ],
                "label": f"{primary_metric} vs {secondary_metric}",
            }
        )

    # Set initial layout arguments.
    initial_first_frontier = list(list_of_frontiers[0].values())[0]
    rel_x = (
        initial_first_frontier.secondary_metric
        not in initial_first_frontier.absolute_metrics
    )
    rel_y = (
        initial_first_frontier.primary_metric
        not in initial_first_frontier.absolute_metrics
    )
    secondary_metric = initial_first_frontier.secondary_metric
    primary_metric = initial_first_frontier.primary_metric

    layout = go.Layout(
        title="Pareto Frontier",
        xaxis={
            "title": secondary_metric,
            "ticksuffix": "%" if rel_x else "",
            "zeroline": True,
        },
        yaxis={
            "title": primary_metric,
            "ticksuffix": "%" if rel_y else "",
            "zeroline": True,
        },
        updatemenus=[
            {
                "buttons": dropdown,
                "x": 0.075,
                "xanchor": "left",
                "y": 1.1,
                "yanchor": "middle",
            }
        ],
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.20,
            "xanchor": "auto",
            "x": 0.075,
        },
        showlegend=True,
        width=750,
        height=550,
        margin=go.layout.Margin(pad=4, l=225, b=125, t=75),  # noqa E741
        shapes=shapes[0],
    )

    fig = go.Figure(data=traces, layout=layout)
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)


def _pareto_frontier_plot_input_processing(
    experiment: Experiment,
    metric_names: Optional[Tuple[str, str]] = None,
    reference_point: Optional[Tuple[float, float]] = None,
    minimize: Optional[Union[bool, Tuple[bool, bool]]] = None,
) -> Tuple[Tuple[str, str], Tuple[float, float], Tuple[bool, bool]]:
    """Processes inputs for Pareto frontier + scatterplot.

    Args:
        experiment: An Ax experiment.
        metric_names: The names of two metrics to be plotted. Defaults to the metrics
            in the optimization_config.
        reference_point: The 2-dimensional reference point to use when plotting the
            Pareto frontier. Defaults to the value of the objective thresholds of each
            variable.
        minimize: Whether each metric is being minimized. Defaults to the direction
            specified for each variable in the optimization config.

    Returns:
        metric_names: The names of two metrics to be plotted.
        reference_point: The 2-dimensional reference point to use when plotting the
            Pareto frontier.
        minimize: Whether each metric is being minimized.

    """
    optimization_config = _validate_experiment_and_get_optimization_config(
        experiment=experiment,
        metric_names=metric_names,
        reference_point=reference_point,
        minimize=minimize,
    )
    metric_names = _validate_and_maybe_get_default_metric_names(
        metric_names=metric_names, optimization_config=optimization_config
    )
    objective_thresholds = _validate_experiment_and_maybe_get_objective_thresholds(
        optimization_config=optimization_config,
        metric_names=metric_names,
        reference_point=reference_point,
        minimize=minimize,
    )
    reference_point = _validate_and_maybe_get_default_reference_point(
        reference_point=reference_point,
        objective_thresholds=objective_thresholds,
        metric_names=metric_names,
    )
    minimize = _validate_and_maybe_get_default_minimize(
        minimize=minimize,
        objective_thresholds=objective_thresholds,
        metric_names=metric_names,
    )
    return metric_names, reference_point, minimize


def _validate_experiment_and_get_optimization_config(
    experiment: Experiment,
    metric_names: Optional[Tuple[str, str]] = None,
    reference_point: Optional[Tuple[float, float]] = None,
    minimize: Optional[Union[bool, Tuple[bool, bool]]] = None,
) -> Optional[OptimizationConfig]:
    optimization_config = None
    # If any inputs are missing, check that `optimization_config` is specified
    if metric_names is None or reference_point is None or minimize is None:
        if experiment.optimization_config is None:
            raise UserInputError(
                "Inference of defaults failed. Please either specify all args to this "
                "function or provide an experiment with an `optimization_config`."
            )
        optimization_config = not_none(experiment.optimization_config)
    return optimization_config


def _validate_and_maybe_get_default_metric_names(
    metric_names: Optional[Tuple[str, str]],
    optimization_config: Optional[OptimizationConfig],
) -> Tuple[str, str]:
    # Default metric_names is all metrics, producing an error if more than 2
    if metric_names is None:
        metric_names = tuple(not_none(optimization_config).metrics.keys())
    if len(metric_names) != 2:
        raise UserInputError(
            f"Expected 2 metrics but got {len(metric_names)}: {metric_names}. Please "
            "specify `metric_names` of length 2 or provide an experiment whose "
            "`optimization_config` has 2 metrics."
        )
    return metric_names


def _validate_experiment_and_maybe_get_objective_thresholds(
    optimization_config: Optional[OptimizationConfig],
    metric_names: Tuple[str, str],
    reference_point: Optional[Tuple[float, float]],
    minimize: Optional[Union[bool, Tuple[bool, bool]]],
) -> List[ObjectiveThreshold]:
    objective_thresholds = []
    # Validate `objective_thresholds` if either `reference_point` or `minimize` are
    # unspecified
    # `method_str` is non-empty if either `reference_point` or `minimize` are `None`
    method_str = " and ".join(
        {
            "`reference_point`" if reference_point is None else "",
            "`minimize`" if minimize is None else "",
        }
        - {""}
    )
    if method_str:
        objective_thresholds = checked_cast(
            MultiObjectiveOptimizationConfig, optimization_config
        ).objective_thresholds
        if len(objective_thresholds) != len(metric_names):
            raise UserInputError(
                f"For automatic determination of {method_str}, expecting one "
                "`objective_threshold` for each metric. Got "
                f"{len(objective_thresholds)}: {objective_thresholds}. Please specify "
                f"{method_str} or provide an experiment whose `optimization_config` "
                "contains one `objective_threshold` for each metric."
            )
        constraint_metrics = {
            objective_threshold.metric.name
            for objective_threshold in objective_thresholds
        }
        missing_metrics = set(metric_names) - set(constraint_metrics)
        if missing_metrics:
            raise UserInputError(
                f"For automatic determination of {method_str}, expecting one "
                "`objective_threshold` for each metric. Missing `objective_thresholds` "
                f"for {missing_metrics}. Please specify {method_str} or provide an "
                "experiment whose `optimization_config` contains one "
                "`objective_threshold` for each metric."
            )

    return objective_thresholds


def _validate_and_maybe_get_default_reference_point(
    reference_point: Optional[Tuple[float, float]],
    objective_thresholds: List[ObjectiveThreshold],
    metric_names: Tuple[str, str],
) -> Tuple[float, float]:
    if reference_point is None:
        reference_point = {
            objective_threshold.metric.name: objective_threshold.bound
            for objective_threshold in objective_thresholds
        }
        reference_point = tuple(
            reference_point[metric_name] for metric_name in metric_names
        )
    if len(reference_point) != 2:
        raise UserInputError(
            f"Expected 2-dimensional `reference_point` but got {len(reference_point)} "
            f"dimensions: {reference_point}. Please specify `reference_point` of "
            "length 2 or provide an experiment whose optimization_config has one "
            "constraint for each of two metrics."
        )
    return reference_point


def _validate_and_maybe_get_default_minimize(
    minimize: Optional[Union[bool, Tuple[bool, bool]]],
    objective_thresholds: List[ObjectiveThreshold],
    metric_names: Tuple[str, str],
) -> Tuple[bool, bool]:
    if minimize is None:
        # Determine `minimize` from constraints since metrics are not always objectives
        objective_thresholds = not_none(objective_thresholds)
        constraint_op_names = {
            objective_threshold.op.name for objective_threshold in objective_thresholds
        }
        invalid_constraint_op_names = constraint_op_names - VALID_CONSTRAINT_OP_NAMES
        if invalid_constraint_op_names:
            raise ValueError(
                f"Operators of all constraints must be in {VALID_CONSTRAINT_OP_NAMES}. "
                f"Got {invalid_constraint_op_names}.)"
            )
        minimize = {
            objective_threshold.metric.name: objective_threshold.op.name == "LEQ"
            for objective_threshold in objective_thresholds
        }
        minimize = tuple(minimize[metric_name] for metric_name in metric_names)
    # If only one bool provided, use for both dimensions
    elif isinstance(minimize, bool):
        minimize = (minimize, minimize)
    if len(minimize) != 2:
        raise UserInputError(
            f"Expected 2-dimensional `minimize` but got {len(minimize)} dimensions: "
            f"{minimize}. Please specify `minimize` of length 2 or provide an "
            "experiment whose optimization_config has one constraint for each of two "
            "metrics."
        )

    return minimize

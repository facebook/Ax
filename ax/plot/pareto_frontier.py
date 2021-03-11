#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objs as go
from ax.core.data import Data
from ax.plot.base import CI_OPACITY, DECIMALS, AxPlotConfig, AxPlotTypes
from ax.plot.color import COLORS, rgba
from ax.plot.helper import extend_range, _format_CI, _format_dict
from ax.plot.pareto_utils import (
    ParetoFrontierResults,
    observed_pareto_frontier,
)
from scipy.stats import norm


DEFAULT_CI_LEVEL: float = 0.9


def _make_label(
    mean: float, sem: float, name: str, is_relative: bool, Z: Optional[float]
) -> str:
    return "{name}: {estimate}{perc} {ci}<br>".format(
        name=name,
        estimate=round(mean, DECIMALS),
        ci=""
        if Z is None
        else _format_CI(estimate=mean, sd=sem, relative=is_relative, zval=Z),
        perc="%" if is_relative else "",
    )


def scatter_plot_with_pareto_frontier_plotly(
    data: Data,
    metric_x: str,
    metric_y: str,
    reference_point: Tuple[float, float],
) -> go.Figure:
    """Plots a scatter of all points in ``Data`` for ``metric_x`` and ``metric_y``
    with a reference point and Pareto frontier.

    Points in the scatter are colored in a gradient representing their trial index,
    with metric_x on x-axis and metric_y on y-axis. Reference point is represented
    as a star and Pareto frontier –– as a line. The frontier connects to the reference
    point via projection lines.
    """
    Y_pareto, Y = observed_pareto_frontier(
        data=data, reference_point=reference_point, metric_x=metric_x, metric_y=metric_y
    )
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
                "ticktext": ["1", "iteration", f"{len(Xs)}"],
            },
        },
        name="Experimental points",
    )
    reference_point_star = go.Scatter(
        x=[reference_point[1]],
        y=[reference_point[0]],
        mode="markers",
        marker={"color": rgba(COLORS.STEELBLUE.value), "size": 25, "symbol": "star"},
    )
    reference_point_line_1 = go.Scatter(
        x=[min(Y_pareto[:, 0]), reference_point[1]],
        y=[reference_point[0], reference_point[0]],
        mode="lines",
        marker={"color": rgba(COLORS.STEELBLUE.value)},
    )
    reference_point_line_2 = go.Scatter(
        x=[reference_point[1], reference_point[1]],
        y=[min(Y_pareto[:, 1]), reference_point[0]],
        mode="lines",
        marker={"color": rgba(COLORS.STEELBLUE.value)},
    )
    pareto_step = go.Scatter(
        x=Y_pareto[:, 0],
        y=Y_pareto[:, 1],
        mode="lines",
        marker={"color": rgba(COLORS.STEELBLUE.value)},
    )

    layout = go.Layout(
        title="Observed points with Pareto frontier",
        showlegend=False,
        xaxis={
            "title": metric_x,
            "range": extend_range(lower=min(Xs), upper=reference_point[0]),
        },
        yaxis={
            "title": metric_y,
            "range": extend_range(lower=min(Ys), upper=reference_point[1]),
        },
    )
    return go.Figure(
        layout=layout,
        data=[
            experimental_points_scatter,
            reference_point_star,
            reference_point_line_1,
            reference_point_line_2,
            pareto_step,
        ],
    )


def scatter_plot_with_pareto_frontier(
    data: Data,
    metric_x: str,
    metric_y: str,
    reference_point: Tuple[float, float],
) -> AxPlotConfig:
    return AxPlotConfig(
        data=scatter_plot_with_pareto_frontier_plotly(
            data=data,
            metric_x=metric_x,
            metric_y=metric_y,
            reference_point=reference_point,
        ),
        plot_type=AxPlotTypes.GENERIC,
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
    primary_means = frontier.means[frontier.primary_metric]
    primary_sems = frontier.sems[frontier.primary_metric]
    secondary_means = frontier.means[frontier.secondary_metric]
    secondary_sems = frontier.sems[frontier.secondary_metric]
    absolute_metrics = frontier.absolute_metrics

    if CI_level is not None:
        Z = 0.5 * norm.ppf(1 - (1 - CI_level) / 2)
    else:
        Z = None

    labels = []
    rel_x = frontier.secondary_metric not in absolute_metrics
    rel_y = frontier.primary_metric not in absolute_metrics

    for i, param_dict in enumerate(frontier.param_dicts):
        heading = "<b>Parameterization {}</b><br>".format(i + 1)
        x_lab = _make_label(
            mean=secondary_means[i],
            sem=secondary_sems[i],
            name=frontier.secondary_metric,
            is_relative=rel_x,
            Z=Z,
        )
        y_lab = _make_label(
            mean=primary_means[i],
            sem=primary_sems[i],
            name=frontier.primary_metric,
            is_relative=rel_y,
            Z=Z,
        )
        parameterization = (
            _format_dict(param_dict, "Parameterization")
            if show_parameterization_on_hover
            else ""
        )
        labels.append(
            "{heading}<br>{xlab}{ylab}{param_blob}".format(
                heading=heading, xlab=x_lab, ylab=y_lab, param_blob=parameterization
            )
        )

    traces = [
        go.Scatter(
            x=secondary_means,
            y=primary_means,
            error_x={
                "type": "data",
                "array": Z * np.array(secondary_sems),
                "thickness": 2,
                "color": rgba(COLORS.STEELBLUE.value, CI_OPACITY),
            },
            error_y={
                "type": "data",
                "array": Z * np.array(primary_sems),
                "thickness": 2,
                "color": rgba(COLORS.STEELBLUE.value, CI_OPACITY),
            },
            mode="markers",
            text=labels,
            hoverinfo="text",
        )
    ]

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

    traces = [
        plot_pareto_frontier(
            frontier=frontier,
            CI_level=CI_level,
            show_parameterization_on_hover=show_parameterization_on_hover,
        ).data["data"][0]
        for frontier in frontier_list
    ]
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
    )

    fig = go.Figure(data=traces, layout=layout)
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)

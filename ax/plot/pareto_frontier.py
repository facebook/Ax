#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import numpy as np
import plotly.graph_objs as go
from ax.plot.base import CI_OPACITY, DECIMALS, AxPlotConfig, AxPlotTypes
from ax.plot.color import COLORS, rgba
from ax.plot.helper import _format_CI, _format_dict
from ax.plot.pareto_utils import ParetoFrontierResults
from scipy.stats import norm


DEFAULT_CI_LEVEL: float = 0.9


def _make_label(
    mean: float, sem: float, name: str, is_relative: bool, Z: Optional[float]
) -> str:
    estimate = str(round(mean, DECIMALS))
    perc = "%" if is_relative else ""
    ci = (
        ""
        if (Z is None or np.isnan(sem))
        else _format_CI(
            estimate=mean, sd=sem, relative=is_relative, zval=Z  # pyre-ignore
        )
    )
    return f"{name}: {estimate}{perc} {ci}<br>"


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
    all_metrics = frontier.means.keys()
    if frontier.arm_names is None:
        arm_names = [f"Parameterization {i}" for i in range(len(frontier.param_dicts))]
    else:
        arm_names = [f"Arm {name}" for name in frontier.arm_names]

    if CI_level is not None:
        Z = 0.5 * norm.ppf(1 - (1 - CI_level) / 2)
    else:
        Z = None

    primary_threshold = None
    secondary_threshold = None
    if frontier.objective_thresholds is not None:
        primary_threshold = frontier.objective_thresholds.get(
            frontier.primary_metric, None
        )
        secondary_threshold = frontier.objective_thresholds.get(
            frontier.secondary_metric, None
        )

    labels = []
    rel_x = frontier.secondary_metric not in absolute_metrics
    rel_y = frontier.primary_metric not in absolute_metrics

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

    shapes = []
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

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numbers

from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objs as go
from ax.analysis.helpers.color_helpers import rgba

from ax.analysis.helpers.constants import CI_OPACITY, COLORS, DECIMALS, Z

from ax.analysis.helpers.plot_helpers import (
    _format_CI,
    _format_dict,
    arm_name_to_sort_key,
)

from ax.core.types import TParameterization


# Structs for plot data
class PlotMetric(NamedTuple):
    """Struct for metric"""

    metric_name: str
    pred: bool


class PlotInSampleArm(NamedTuple):
    """Struct for in-sample arms (both observed and predicted data)"""

    name: str
    parameters: TParameterization
    y: Dict[str, float]
    y_hat: Dict[str, float]
    se: Dict[str, float]
    se_hat: Dict[str, float]


class PlotData(NamedTuple):
    """Struct for plot data, including metrics and in-sample arms"""

    metrics: List[str]
    in_sample: Dict[str, PlotInSampleArm]


def _error_scatter_data(
    arms: Iterable[PlotInSampleArm],
    y_axis_var: PlotMetric,
    x_axis_var: Optional[PlotMetric] = None,
) -> Tuple[List[float], Optional[List[float]], List[float], List[float]]:
    y_metric_key = "y_hat" if y_axis_var.pred else "y"
    y_sd_key = "se_hat" if y_axis_var.pred else "se"

    arm_names = [a.name for a in arms]
    y = [getattr(a, y_metric_key).get(y_axis_var.metric_name, np.nan) for a in arms]
    y_se = [getattr(a, y_sd_key).get(y_axis_var.metric_name, np.nan) for a in arms]

    # x can be metric for a metric or arm names
    if x_axis_var is None:
        x = arm_names
        x_se = None
    else:
        x_metric_key = "y_hat" if x_axis_var.pred else "y"
        x_sd_key = "se_hat" if x_axis_var.pred else "se"
        x = [getattr(a, x_metric_key).get(x_axis_var.metric_name, np.nan) for a in arms]
        x_se = [getattr(a, x_sd_key).get(x_axis_var.metric_name, np.nan) for a in arms]

    return x, x_se, y, y_se


def _error_scatter_trace(
    arms: Sequence[PlotInSampleArm],
    y_axis_var: PlotMetric,
    x_axis_var: Optional[PlotMetric] = None,
    y_axis_label: Optional[str] = None,
    x_axis_label: Optional[str] = None,
    show_CI: bool = True,
) -> Dict[str, Any]:
    """Plot scatterplot with error bars.

    Args:
        arms (List[Union[PlotInSampleArm, PlotOutOfSampleArm]]):
            a list of in-sample or out-of-sample arms.
            In-sample arms have observed data, while out-of-sample arms
            just have predicted data. As a result,
            when passing out-of-sample arms, pred must be True.
        y_axis_var: name of metric for y-axis, along with whether
            it is observed or predicted.
        x_axis_var: name of metric for x-axis,
            along with whether it is observed or predicted. If None, arm names
            are automatically used.
        y_axis_label: custom label to use for y axis.
            If None, use metric name from `y_axis_var`.
        x_axis_label: custom label to use for x axis.
            If None, use metric name from `x_axis_var` if that is not None.
        show_CI: if True, plot confidence intervals.
    """

    # Opportunistically sort if arm names are in {trial}_{arm} format
    arms = sorted(arms, key=lambda a: arm_name_to_sort_key(a.name), reverse=True)

    x, x_se, y, y_se = _error_scatter_data(
        arms=arms,
        y_axis_var=y_axis_var,
        x_axis_var=x_axis_var,
    )
    labels = []

    arm_names = [a.name for a in arms]

    for i in range(len(arm_names)):
        heading = f"<b>Arm {arm_names[i]}</b><br>"
        x_lab = (
            "{name}: {estimate} {ci}<br>".format(
                name=x_axis_var.metric_name if x_axis_label is None else x_axis_label,
                estimate=(
                    round(x[i], DECIMALS) if isinstance(x[i], numbers.Number) else x[i]
                ),
                ci="" if x_se is None else _format_CI(x[i], x_se[i]),
            )
            if x_axis_var is not None
            else ""
        )
        y_lab = "{name}: {estimate} {ci}<br>".format(
            name=y_axis_var.metric_name if y_axis_label is None else y_axis_label,
            estimate=(
                round(y[i], DECIMALS) if isinstance(y[i], numbers.Number) else y[i]
            ),
            ci="" if y_se is None else _format_CI(y[i], y_se[i]),
        )

        parameterization = _format_dict(arms[i].parameters, "Parameterization")

        labels.append(
            "{arm_name}<br>{xlab}{ylab}{param_blob}".format(
                arm_name=heading,
                xlab=x_lab,
                ylab=y_lab,
                param_blob=parameterization,
            )
        )
        i += 1

    trace = go.Scatter(
        x=x,
        y=y,
        marker={"color": rgba(COLORS.STEELBLUE.value)},
        mode="markers",
        name="In-sample",
        text=labels,
        hoverinfo="text",
    )

    if show_CI:
        if x_se is not None:
            trace.update(
                error_x={
                    "type": "data",
                    "array": np.multiply(x_se, Z),
                    "color": rgba(COLORS.STEELBLUE.value, CI_OPACITY),
                }
            )
        if y_se is not None:
            trace.update(
                error_y={
                    "type": "data",
                    "array": np.multiply(y_se, Z),
                    "color": rgba(COLORS.STEELBLUE.value, CI_OPACITY),
                }
            )

    trace.update(visible=True)
    trace.update(showlegend=True)
    return trace

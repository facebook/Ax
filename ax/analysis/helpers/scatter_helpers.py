#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numbers

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ax.analysis.helpers.color_helpers import rgba

from ax.analysis.helpers.constants import CI_OPACITY, COLORS, DECIMALS, Z

from ax.analysis.helpers.plot_helpers import _format_CI, _format_dict

from ax.core.types import TParameterization


def extract_mean_and_error_from_df(
    df: pd.DataFrame,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Extract mean and error from dataframe.

    Args:
        df: dataframe containing the scatter plot data
    Returns:
        x: list of x values
        x_se: list of x standard error
        y: list of y values
        y_se: list of y standard error
    """
    x = df["x"]
    x_se = df["x_se"]
    y = df["y"]
    y_se = df["y_se"]

    return (x, x_se, y, y_se)


def make_label(
    arm_name: str,
    x_name: str,
    x_val: float,
    x_se: float,
    y_name: str,
    y_val: float,
    y_se: float,
    param_blob: TParameterization,
) -> str:
    """Make label for scatter plot.

    Args:
        arm_name: Name of arm
        x_name: Name of x variable
        x_val: Value of x variable
        x_se: Standard error of x variable
        y_name: Name of y variable
        y_val: Value of y variable
        y_se: Standard error of y variable
        param_blob: Parameterization of arm

    Returns:
        Label for scatter plot.
    """
    heading = f"<b>Arm {arm_name}</b><br>"
    x_lab = "{name}: {estimate} {ci}<br>".format(
        name=x_name,
        estimate=(
            round(x_val, DECIMALS) if isinstance(x_val, numbers.Number) else x_val
        ),
        ci=_format_CI(estimate=x_val, sd=x_se),
    )
    y_lab = "{name}: {estimate} {ci}<br>".format(
        name=y_name,
        estimate=(
            round(y_val, DECIMALS) if isinstance(y_val, numbers.Number) else y_val
        ),
        ci=_format_CI(estimate=y_val, sd=y_se),
    )

    parameterization = _format_dict(param_blob, "Parameterization")

    return "{arm_name}<br>{xlab}{ylab}{param_blob}".format(
        arm_name=heading,
        xlab=x_lab,
        ylab=y_lab,
        param_blob=parameterization,
    )


def error_scatter_trace_from_df(
    df: pd.DataFrame,
    show_CI: bool = True,
    visible: bool = True,
    y_axis_label: Optional[str] = None,
    x_axis_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Plot scatterplot with error bars.

    Args:
        df: dataframe containing the scatter plot data
        show_CI: if True, plot confidence intervals.
        visible: if True, trace will be visible in figure
        y_axis_label: custom label to use for y axis.
            If None, use metric name from `y_axis_var`.
        x_axis_label: custom label to use for x axis.
            If None, use metric name from `x_axis_var` if that is not None.
    """

    x, x_se, y, y_se = extract_mean_and_error_from_df(df)

    labels = []
    arm_names = df["arm_name"]

    metric_name = df["metric_name"].iloc[0]

    print("Data frame: " + str(df))
    print("Arm names: " + str(arm_names))
    print("x" + str(x))
    for _, row in df.iterrows():
        labels.append(
            make_label(
                arm_name=row["arm_name"],
                x_name=metric_name if x_axis_label is None else x_axis_label,
                x_val=row["x"],
                x_se=row["x_se"],
                y_name=(metric_name if y_axis_label is None else y_axis_label),
                y_val=row["y"],
                y_se=row["y_se"],
                param_blob=row["arm_parameters"],
            )
        )

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

    trace.update(visible=visible)
    trace.update(showlegend=True)
    return trace

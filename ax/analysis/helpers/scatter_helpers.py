#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numbers

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import plotly.graph_objs as go
from ax.analysis.helpers.color_helpers import rgba

from ax.analysis.helpers.constants import CI_OPACITY, COLORS, DECIMALS, Z

from ax.analysis.helpers.plot_helpers import _format_CI, _format_dict

from ax.core.types import TParameterization

from ax.utils.stats.statstools import relativize

# disable false positive "SettingWithCopyWarning"
pd.options.mode.chained_assignment = None


def relativize_dataframe(df: pd.DataFrame, status_quo_name: str) -> pd.DataFrame:
    """
    Relativizes the dataframe with respect to "status_quo_name". Assumes as a
    precondition that for each metric in the dataframe, there is a row with
    arm_name == status_quo_name to relativize against.

    Args:
        df: dataframe with the following columns
            {
            "arm_name": name of the arm in the cross validation result
            "metric_name": name of the observed/predicted metric
            "x": value of the observation for the metric for this arm
            "x_se": standard error of the observation for the metric of this arm
            "y": value predicted for the metric for this arm
            "y_se": standard error of predicted metric for this arm
            }
        status_quo_name: name of the status quo arm in the dataframe to use
            for relativization.
    Returns:
        A dataframe containing the same rows as df, with the observation and predicted
        data values relativized with respect to the status quo.
        An additional column "rel" is added to indicate whether the data is relativized.

    """
    metrics = df["metric_name"].unique()

    def _relativize_filtered_dataframe(
        df: pd.DataFrame, metric_name: str, status_quo_name: str
    ) -> pd.DataFrame:
        df = df.loc[df["metric_name"] == metric_name]
        status_quo_row = df.loc[df["arm_name"] == status_quo_name]

        mean_c = status_quo_row["y"].iloc[0]
        sem_c = status_quo_row["y_se"].iloc[0]
        y_rel, y_se_rel = relativize(
            means_t=df["y"].tolist(),
            sems_t=df["y_se"].tolist(),
            mean_c=mean_c,
            sem_c=sem_c,
            as_percent=True,
        )
        df["y"] = y_rel
        df["y_se"] = y_se_rel

        mean_c = status_quo_row["x"].iloc[0]
        sem_c = status_quo_row["x_se"].iloc[0]
        x_rel, x_se_rel = relativize(
            means_t=df["x"].tolist(),
            sems_t=df["x_se"].tolist(),
            mean_c=mean_c,
            sem_c=sem_c,
            as_percent=True,
        )
        df["x"] = x_rel
        df["x_se"] = x_se_rel
        df["rel"] = True
        return df

    return pd.concat(
        [
            _relativize_filtered_dataframe(
                df=df, metric_name=metric, status_quo_name=status_quo_name
            )
            for metric in metrics
        ]
    )


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
    x_axis_values: Optional[Tuple[str, float, float]],
    y_axis_values: Tuple[str, float, float],
    param_blob: TParameterization,
    rel: bool,
) -> str:
    """Make label for scatter plot.

    Args:
        arm_name: Name of arm
        x_axis_values: Optional Tuple of
            x_name: Name of x variable
            x_val: Value of x variable
            x_se: Standard error of x variable
        y_axis_values: Tuple of
            y_name: Name of y variable
            y_val: Value of y variable
            y_se: Standard error of y variable
        param_blob: Parameterization of arm
        rel: whether the data is relativized as a %

    Returns:
        Label for scatter plot.
    """
    heading = f"<b>Arm {arm_name}</b><br>"
    x_lab = ""
    if x_axis_values is not None:
        x_name, x_val, x_se = x_axis_values
        x_lab = "{name}: {estimate}{perc} {ci}<br>".format(
            name=x_name,
            estimate=(
                round(x_val, DECIMALS) if isinstance(x_val, numbers.Number) else x_val
            ),
            ci=_format_CI(estimate=x_val, sd=x_se),
            perc="%" if rel else "",
        )

    y_name, y_val, y_se = y_axis_values
    y_lab = "{name}: {estimate}{perc} {ci}<br>".format(
        name=y_name,
        estimate=(
            round(y_val, DECIMALS) if isinstance(y_val, numbers.Number) else y_val
        ),
        ci=_format_CI(estimate=y_val, sd=y_se),
        perc="%" if rel else "",
    )

    parameterization = _format_dict(param_blob, "Parameterization")

    return "{arm_name}<br>{xlab}{ylab}{param_blob}".format(
        arm_name=heading,
        xlab=x_lab,
        ylab=y_lab,
        param_blob=parameterization,
    )


def error_dot_plot_trace_from_df(
    df: pd.DataFrame,
    show_CI: bool = True,
    visible: bool = True,
) -> Dict[str, Any]:
    """Creates trace for dot plot with confidence intervals.
    Categorizes by arm name.

    Args:
        df: dataframe containing the scatter plot data
        show_CI: if True, plot confidence intervals.
        visible: if True, trace will be visible in figure
    """

    _, _, y, y_se = extract_mean_and_error_from_df(df)

    labels = []

    metric_name = df["metric_name"].iloc[0]

    for _, row in df.iterrows():
        labels.append(
            make_label(
                arm_name=row["arm_name"],
                x_axis_values=(None),
                y_axis_values=(
                    metric_name,
                    row["y"],
                    row["y_se"],
                ),
                param_blob=row["arm_parameters"],
                rel=(False if "rel" not in row else row["rel"]),
            )
        )

    trace = go.Scatter(
        x=df["arm_name"],
        y=y,
        marker={"color": rgba(COLORS.STEELBLUE.value)},
        mode="markers",
        name="In-sample",
        text=labels,
        hoverinfo="text",
    )

    if show_CI:
        if y_se is not None:
            trace.update(
                error_y={
                    "type": "data",
                    "array": np.multiply(y_se, Z),
                    "color": rgba(COLORS.STEELBLUE.value, CI_OPACITY),
                }
            )

    trace.update(visible=visible)
    trace.update(showlegend=False)
    return trace


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

    metric_name = df["metric_name"].iloc[0]

    for _, row in df.iterrows():
        labels.append(
            make_label(
                arm_name=row["arm_name"],
                x_axis_values=(
                    metric_name if x_axis_label is None else x_axis_label,
                    row["x"],
                    row["x_se"],
                ),
                y_axis_values=(
                    (metric_name if y_axis_label is None else y_axis_label),
                    row["y"],
                    row["y_se"],
                ),
                param_blob=row["arm_parameters"],
                rel=(False if "rel" not in row else row["rel"]),
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

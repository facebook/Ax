#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objs as go

from ax.analysis.helpers.constants import Z

from ax.analysis.helpers.layout_helpers import layout_format, updatemenus_format
from ax.analysis.helpers.scatter_helpers import (
    _error_scatter_data,
    _error_scatter_trace,
    PlotData,
    PlotInSampleArm,
    PlotMetric,
)

from ax.modelbridge.cross_validation import CVResult


# Helper functions for plotting model fits
def get_min_max_with_errors(
    x: List[float], y: List[float], sd_x: List[float], sd_y: List[float]
) -> Tuple[float, float]:
    """Get min and max of a bivariate dataset (across variables).

    Args:
        x: point estimate of x variable.
        y: point estimate of y variable.
        sd_x: standard deviation of x variable.
        sd_y: standard deviation of y variable.

    Returns:
        min_: minimum of points, including uncertainty.
        max_: maximum of points, including uncertainty.

    """
    min_ = min(
        min(np.array(x) - np.multiply(sd_x, Z)), min(np.array(y) - np.multiply(sd_y, Z))
    )
    max_ = max(
        max(np.array(x) + np.multiply(sd_x, Z)), max(np.array(y) + np.multiply(sd_y, Z))
    )
    return min_, max_


def get_plotting_limit_ignore_outliers(
    x: List[float], y: List[float], sd_x: List[float], sd_y: List[float]
) -> Tuple[float, float]:
    """Get a range for a bivarite dataset based on the 25th and 75th percentiles
    Used as plotting limit to ignore outliers.

    Args:
        x: point estimate of x variable.
        y: point estimate of y variable.
        sd_x: standard deviation of x variable.
        sd_y: standard deviation of y variable.

    Returns:
        min: lower bound of range
        max: higher bound of range

    """
    min_, max_ = get_min_max_with_errors(x=x, y=y, sd_x=sd_x, sd_y=sd_y)

    x_np = np.array(x)
    # TODO: replace interpolation->method once it becomes standard.
    q1 = np.nanpercentile(x_np, q=25, interpolation="lower").min()
    q3 = np.nanpercentile(x_np, q=75, interpolation="higher").max()
    quartile_difference = q3 - q1

    y_lower = q1 - 1.5 * quartile_difference
    y_upper = q3 + 1.5 * quartile_difference

    # clip outliers from x
    x_np = x_np.clip(y_lower, y_upper).tolist()
    min_robust, max_robust = get_min_max_with_errors(x=x_np, y=y, sd_x=sd_x, sd_y=sd_y)
    y_padding = 0.05 * (max_robust - min_robust)

    return (max(min_robust, min_) - y_padding, min(max_robust, max_) + y_padding)


def diagonal_trace(min_: float, max_: float, visible: bool = True) -> Dict[str, Any]:
    """Diagonal line trace from (min_, min_) to (max_, max_).

    Args:
        min_: minimum to be used for starting point of line.
        max_: maximum to be used for ending point of line.
        visible: if True, trace is set to visible.
    """
    return go.Scatter(
        x=[min_, max_],
        y=[min_, max_],
        line=dict(color="black", width=2, dash="dot"),  # noqa: C408
        mode="lines",
        hoverinfo="none",
        visible=visible,
        showlegend=False,
    )


def default_value_se_raw(se_raw: Optional[List[float]], out_length: int) -> List[float]:
    """
    Takes a list of standard errors and maps edge cases to default list
    of floats.

    """
    new_se_raw = (
        [0.0 if np.isnan(se) else se for se in se_raw]
        if se_raw is not None
        else [0.0] * out_length
    )
    return new_se_raw


def obs_vs_pred_dropdown_plot(
    data: PlotData,
    xlabel: str = "Actual Outcome",
    ylabel: str = "Predicted Outcome",
) -> go.Figure:
    """Plot a dropdown plot of observed vs. predicted values from a model.

    Args:
        data: a name tuple storing observed and predicted data
            from a model.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
    """
    traces = []
    metric_dropdown = []
    layout_axis_range = []

    for i, metric in enumerate(data.metrics):
        y_raw, se_raw, y_hat, se_hat = _error_scatter_data(
            list(data.in_sample.values()),
            y_axis_var=PlotMetric(metric_name=metric, pred=True),
            x_axis_var=PlotMetric(metric_name=metric, pred=False),
        )
        se_raw = default_value_se_raw(se_raw=se_raw, out_length=len(y_raw))

        # Use the min/max of the limits
        min_, max_ = get_plotting_limit_ignore_outliers(
            x=y_raw, y=y_hat, sd_x=se_raw, sd_y=se_hat
        )
        layout_axis_range.append([min_, max_])
        traces.append(
            diagonal_trace(
                min_,
                max_,
                visible=(i == 0),
            )
        )

        traces.append(
            _error_scatter_trace(
                arms=list(data.in_sample.values()),
                show_CI=True,
                x_axis_label=xlabel,
                x_axis_var=PlotMetric(metric_name=metric, pred=False),
                y_axis_label=ylabel,
                y_axis_var=PlotMetric(metric_name=metric, pred=True),
            )
        )

        # only the first two traces are visible (corresponding to first outcome
        # in dropdown)
        is_visible = [False] * (len(data.metrics) * 2)
        is_visible[2 * i] = True
        is_visible[2 * i + 1] = True

        # on dropdown change, restyle
        metric_dropdown.append(
            {
                "args": [
                    {"visible": is_visible},
                    {
                        "xaxis.range": layout_axis_range[-1],
                        "yaxis.range": layout_axis_range[-1],
                    },
                ],
                "label": metric,
                "method": "update",
            }
        )

    updatemenus = updatemenus_format(metric_dropdown=metric_dropdown)
    layout = layout_format(
        layout_axis_range_value=layout_axis_range[0],
        xlabel=xlabel,
        ylabel=ylabel,
        updatemenus=updatemenus,
    )

    return go.Figure(data=traces, layout=layout)


def remap_label(
    cv_results: List[CVResult], label_dict: Dict[str, str]
) -> List[CVResult]:
    """Remaps labels in cv_results according to label_dict.

    Args:
        cv_results: A CVResult for each observation in the training data.
        label_dict: optional map from real metric names to shortened names

    Returns:
        A CVResult for each observation in the training data.
    """
    cv_results = deepcopy(cv_results)  # Copy and edit in-place
    for cv_i in cv_results:
        cv_i.observed.data.metric_names = [
            label_dict.get(m, m) for m in cv_i.observed.data.metric_names
        ]
        cv_i.predicted.metric_names = [
            label_dict.get(m, m) for m in cv_i.predicted.metric_names
        ]
    return cv_results


def get_cv_plot_data(
    cv_results: List[CVResult], label_dict: Optional[Dict[str, str]]
) -> PlotData:
    """Construct PlotData from cv_results, mapping observed to y and se,
    and predicted to y_hat and se_hat.

    Args:
        cv_results: A CVResult for each observation in the training data.
        label_dict: optional map from real metric names to shortened names

    Returns:
        PlotData with the following fields:
            metrics: List[str]
            in_sample: Dict[str, PlotInSampleArm]
                PlotInSample arm have the fields
                {
                    "name"
                    "y"
                    "se"
                    "parameters"
                    "y_hat"
                    "se_hat"
                }

    """
    if len(cv_results) == 0:
        return PlotData(metrics=[], in_sample={})

    if label_dict:
        cv_results = remap_label(cv_results=cv_results, label_dict=label_dict)

    # arm_name -> Arm data
    insample_data: Dict[str, PlotInSampleArm] = {}

    # Get the union of all metric_names seen in predictions
    metric_names = list(
        set().union(*(cv_result.predicted.metric_names for cv_result in cv_results))
    )

    for rid, cv_result in enumerate(cv_results):
        arm_name = cv_result.observed.arm_name
        y, se, y_hat, se_hat = {}, {}, {}, {}

        arm_data = {
            "name": cv_result.observed.arm_name,
            "y": y,
            "se": se,
            "parameters": cv_result.observed.features.parameters,
            "y_hat": y_hat,
            "se_hat": se_hat,
        }
        for i, mname in enumerate(cv_result.observed.data.metric_names):
            y[mname] = cv_result.observed.data.means[i]
            se[mname] = np.sqrt(cv_result.observed.data.covariance[i][i])
        for i, mname in enumerate(cv_result.predicted.metric_names):
            y_hat[mname] = cv_result.predicted.means[i]
            se_hat[mname] = np.sqrt(cv_result.predicted.covariance[i][i])

        # Expected `str` for 2nd anonymous parameter to call `dict.__setitem__` but got
        # `Optional[str]`.
        # pyre-fixme[6]:
        insample_data[f"{arm_name}_{rid}"] = PlotInSampleArm(**arm_data)
    return PlotData(
        metrics=metric_names,
        in_sample=insample_data,
    )

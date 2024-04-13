#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from ax.analysis.helpers.constants import Z

from ax.analysis.helpers.plot_helpers import arm_name_to_sort_key

from ax.modelbridge.cross_validation import CVResult


def error_scatter_data_from_cv_results(
    cv_results: List[CVResult],
    metric_name: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Extract mean and error from CVResults

    Args:
        cv_results: list of cross_validation result objects
        metric_name: metric name to use for prediction
            and observation
    Returns:
        x: list of x values
        x_se: list of x standard error
        y: list of y values
        y_se: list of y standard error
    """
    y = [cv_result.predicted.means_dict[metric_name] for cv_result in cv_results]
    y_se = [
        np.sqrt(cv_result.predicted.covariance_matrix[metric_name][metric_name])
        for cv_result in cv_results
    ]

    x = [cv_result.observed.data.means_dict[metric_name] for cv_result in cv_results]
    x_se = [
        np.sqrt(cv_result.observed.data.covariance_matrix[metric_name][metric_name])
        for cv_result in cv_results
    ]

    return x, x_se, y, y_se


def cv_results_to_df(
    cv_results: List[CVResult],
    metric_name: str,
) -> pd.DataFrame:
    """Create a dataframe with error scatterplot data

    Args:
        cv_results: list of cross validation results
        metric_name: name of metric. Predicted val on y-axis,
            observed val on x-axis.
    """

    # Opportunistically sort if arm names are in {trial}_{arm} format
    cv_results = sorted(
        cv_results,
        key=lambda c: arm_name_to_sort_key(c.observed.arm_name),
        reverse=True,
    )

    x, x_se, y, y_se = error_scatter_data_from_cv_results(
        cv_results=cv_results,
        metric_name=metric_name,
    )

    arm_names = [c.observed.arm_name for c in cv_results]
    records = []

    for i in range(len(arm_names)):

        records.append(
            {
                "arm_name": arm_names[i],
                "metric_name": metric_name,
                "x": x[i],
                "x_se": x_se[i],
                "y": y[i],
                "y_se": y_se[i],
                "arm_parameters": cv_results[i].observed.features.parameters,
            }
        )
    return pd.DataFrame.from_records(records)


# Helper functions for plotting model fits
def get_min_max_with_errors(
    x: List[float], y: List[float], se_x: List[float], se_y: List[float]
) -> Tuple[float, float]:
    """Get min and max of a bivariate dataset (across variables).

    Args:
        x: point estimate of x variable.
        y: point estimate of y variable.
        se_x: standard error of x variable.
        se_y: standard error of y variable.

    Returns:
        min_: minimum of points, including uncertainty.
        max_: maximum of points, including uncertainty.

    """
    min_ = min(
        min(np.array(x) - np.multiply(se_x, Z)), min(np.array(y) - np.multiply(se_y, Z))
    )
    max_ = max(
        max(np.array(x) + np.multiply(se_x, Z)), max(np.array(y) + np.multiply(se_y, Z))
    )
    return min_, max_


def get_plotting_limit_ignore_outliers(
    x: List[float], y: List[float], se_x: List[float], se_y: List[float]
) -> Tuple[List[float], Tuple[float, float]]:
    """Get a range for a bivarite dataset based on the 25th and 75th percentiles
    Used as plotting limit to ignore outliers.

    Args:
        x: point estimate of x variable.
        y: point estimate of y variable.
        se_x: standard error of x variable.
        se_y: standard error of y variable.

    Returns:
        (min, max): layout axis range
        (min, max): diagonal trace range

    """
    se_x = default_value_se_raw(se_raw=se_x, out_length=len(x))

    min_, max_ = get_min_max_with_errors(x=x, y=y, se_x=se_x, se_y=se_y)

    x_np = np.array(x)
    # TODO: replace interpolation->method once it becomes standard.
    q1 = np.nanpercentile(x_np, q=25, interpolation="lower").min()
    q3 = np.nanpercentile(x_np, q=75, interpolation="higher").max()
    quartile_difference = q3 - q1

    y_lower = q1 - 1.5 * quartile_difference
    y_upper = q3 + 1.5 * quartile_difference

    # clip outliers from x
    x_np = x_np.clip(y_lower, y_upper).tolist()
    min_robust, max_robust = get_min_max_with_errors(x=x_np, y=y, se_x=se_x, se_y=se_y)
    y_padding = 0.05 * (max_robust - min_robust)

    layout_range = [
        max(min_robust, min_) - y_padding,
        min(max_robust, max_) + y_padding,
    ]
    diagonal_trace_range = (
        min(min_robust, min_) - y_padding,
        max(max_robust, max_) + y_padding,
    )

    return (layout_range, diagonal_trace_range)


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

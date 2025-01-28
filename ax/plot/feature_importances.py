#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objs as go
from ax.core.parameter import ChoiceParameter
from ax.exceptions.core import NoDataError
from ax.modelbridge import ModelBridge
from ax.plot.base import AxPlotConfig, AxPlotTypes
from ax.plot.helper import compose_annotation
from ax.utils.common.logger import get_logger
from plotly import subplots

logger: Logger = get_logger(__name__)


def plot_feature_importance_plotly(df: pd.DataFrame, title: str) -> go.Figure:
    if df.empty:
        raise NoDataError("No Data on Feature Importances found.")
    df.set_index(df.columns[0], inplace=True)
    data = [
        go.Bar(y=df.index, x=df[column_name], name=column_name, orientation="h")
        for column_name in df.columns
    ]
    fig = subplots.make_subplots(
        rows=len(df.columns),
        cols=1,
        subplot_titles=df.columns,
        print_grid=False,
        shared_xaxes=True,
    )

    for idx, item in enumerate(data):
        fig.append_trace(item, idx + 1, 1)
    fig.layout.showlegend = False
    fig.layout.margin = go.layout.Margin(
        l=8 * min(max(len(idx) for idx in df.index), 75)  # noqa E741
    )
    fig.layout.title = title
    return fig


def plot_feature_importance(df: pd.DataFrame, title: str) -> AxPlotConfig:
    """Wrapper method to convert plot_feature_importance_plotly to
    AxPlotConfig"""
    return AxPlotConfig(
        # pyre-fixme[6]: For 1st argument expected `Dict[str, typing.Any]` but got
        #  `Figure`.
        data=plot_feature_importance_plotly(df, title),
        plot_type=AxPlotTypes.GENERIC,
    )


def plot_feature_importance_by_metric_plotly(model: ModelBridge) -> go.Figure:
    """One plot per feature, showing importances by metric."""
    importances = []
    for metric_name in sorted(model.metric_names):
        try:
            vals: dict[str, Any] = model.feature_importances(metric_name)
            vals["index"] = metric_name
            importances.append(vals)
        except NotImplementedError:
            logger.warning(
                f"Model for {metric_name} does not support feature importances."
            )
    if not importances:
        raise NotImplementedError(
            "Feature importances could not be calculated for any metric"
        )
    df = pd.DataFrame(importances)

    # plot_feature_importance expects index in first column
    df = df.reindex(columns=(["index"] + [a for a in df.columns if a != "index"]))

    plot_fi = plot_feature_importance_plotly(df, "Parameter Sensitivity by Metric")
    num_subplots = len(df.columns)
    num_features = len(df)
    # Include per-subplot margin for subplot titles (feature names).
    plot_fi["layout"]["height"] = num_subplots * (num_features + 1) * 50
    return plot_fi


def plot_feature_importance_by_metric(model: ModelBridge) -> AxPlotConfig:
    """Wrapper method to convert plot_feature_importance_by_metric_plotly to
    AxPlotConfig"""
    return AxPlotConfig(
        # pyre-fixme[6]: For 1st argument expected `Dict[str, typing.Any]` but got
        #  `Figure`.
        data=plot_feature_importance_by_metric_plotly(model),
        plot_type=AxPlotTypes.GENERIC,
    )


def plot_feature_importance_by_feature_plotly(
    model: ModelBridge | None = None,
    sensitivity_values: dict[str, dict[str, float | npt.NDArray]] | None = None,
    relative: bool = False,
    caption: str = "",
    importance_measure: str = "",
    label_dict: dict[str, str] | None = None,
) -> go.Figure:
    """One plot per metric, showing importances by feature.

    If sensitivity values are not all positive, the absolute value will be shown
    and color will indicate positive or negative sign.

    Args:
        model: A model with a ``feature_importances`` method.
        sensitivity_values: The sensitivity values for each metric in a dict format.
            It takes the following format if only the sensitivity value is plotted:
            `{"metric1":{"parameter1":value1,"parameter2":value2 ...} ...}`
            It takes the following format if the sensitivity value and standard error
            are plotted: `{"metric1":{"parameter1":[value1,var,se],
            "parameter2":[[value2,var,se]]...}...}}`.
        relative: Whether to normalize feature importances so that they add to 1.
        caption: An HTML-formatted string to place at the bottom of the plot.
        importance_measure: The name of the importance metric to be added to the title.
        label_dict: A dictionary mapping metric names to short labels.
    Returns a go.Figure of feature importances.
    """
    if sensitivity_values is None:
        if model is None:
            raise ValueError(
                "A model is required when sensitivity values are not provided"
            )
        try:
            sensitivity_values = {
                metric_name: model.feature_importances(metric_name)
                for i, metric_name in enumerate(sorted(model.metric_names))
            }
        except NotImplementedError:
            raise NotImplementedError(
                "Feature importances cannot be computed by the model."
            )

    if label_dict is not None:
        sensitivity_values = {  # pyre-ignore
            label_dict.get(metric_name, metric_name): v
            for metric_name, v in sensitivity_values.items()
        }
    traces = []
    dropdown = []
    categorical_features = []
    if model is not None:
        categorical_features = [
            name
            for name, par in model.model_space.parameters.items()
            if isinstance(par, ChoiceParameter) and not par.is_ordered
        ]

    for i, metric_name in enumerate(sorted(sensitivity_values.keys())):
        importances = sensitivity_values[metric_name]
        factor_col = "Factor"
        importance_col = "Importance"
        sign_col = "Sign"
        error_plot = np.asarray(next(iter(importances.values()))).size > 1
        if error_plot:
            importance_col_se = "SE"
            df = pd.DataFrame(
                [
                    {
                        factor_col: factor,
                        importance_col: np.asarray(importance)[0],
                        importance_col_se: np.asarray(importance)[2],
                        sign_col: (
                            0
                            if factor in categorical_features
                            else 2 * (np.asarray(importance)[0] >= 0).astype(int) - 1
                        ),
                    }
                    for factor, importance in importances.items()
                ]
            )
            df[importance_col] = df[importance_col].abs()
            df = df.sort_values(importance_col)
            error_x = {"type": "data", "array": df[importance_col_se], "visible": True}

        else:
            df = pd.DataFrame(
                [
                    {
                        factor_col: factor,
                        importance_col: importance,
                        sign_col: (
                            0
                            if factor in categorical_features
                            # pyre-fixme[16]: Item `bool` of
                            #  `Union[ndarray[typing.Any, np.dtype[typing.Any]], bool]`
                            #  has no attribute `astype`.
                            else 2 * (importance >= 0).astype(int) - 1
                        ),
                    }
                    for factor, importance in importances.items()
                ]
            )
            df[importance_col] = df[importance_col].abs()
            df = df.sort_values(importance_col)
            error_x = None
        if relative:
            df[importance_col] = df[importance_col].div(df[importance_col].sum())

        colors = {-1: "darkorange", 0: "gray", 1: "steelblue"}
        names = {
            -1: "Decreases metric",
            0: "Affects metric (categorical choice)",
            1: "Increases metric",
        }
        legend_counter = {-1: 0, 0: 0, 1: 0}
        all_positive = all(df[sign_col] >= 0)
        for _, row in df.iterrows():
            traces.append(
                go.Bar(
                    name=names[row[sign_col]],
                    orientation="h",
                    visible=i == 0,
                    x=np.array([row[importance_col]]),
                    y=np.array([row[factor_col]]),
                    error_x=error_x,
                    opacity=0.8,
                    marker_color=colors[row[sign_col]],
                    showlegend=(not all_positive)
                    and (legend_counter[row[sign_col]] == 0),
                    legendgroup=str(row[sign_col]),
                )
            )
            legend_counter[row[sign_col]] += 1

        is_visible = [False] * (len(sensitivity_values) * len(df))
        for j in range(i * len(df), (i + 1) * len(df)):
            is_visible[j] = True
        dropdown.append(
            {"args": ["visible", is_visible], "label": metric_name, "method": "restyle"}
        )
    if not traces:
        raise NotImplementedError("No traces found for metric")

    updatemenus = [
        {
            "x": 0,
            "y": 1,
            "yanchor": "top",
            "xanchor": "left",
            "buttons": dropdown,
            "pad": {
                "t": -40
            },  # hack to put dropdown below title regardless of number of features
        }
    ]
    features = list(list(sensitivity_values.values())[0].keys())

    longest_label = max(len(f) for f in features)
    longest_metric = max(len(m) for m in sensitivity_values.keys())

    layout = go.Layout(
        height=200 + len(features) * 20,
        width=10 * longest_label + max(10 * longest_metric, 400),
        hovermode="closest",
        annotations=compose_annotation(caption=caption),
        title=f"Parameter Sensitivity by {importance_measure}",
        updatemenus=updatemenus,
    )

    if relative:
        layout.update({"xaxis": {"tickformat": ".0%"}})

    return go.Figure(data=traces, layout=layout)


def plot_feature_importance_by_feature(
    model: ModelBridge | None = None,
    sensitivity_values: dict[str, dict[str, float | npt.NDArray]] | None = None,
    relative: bool = False,
    caption: str = "",
    importance_measure: str = "",
    label_dict: dict[str, str] | None = None,
) -> AxPlotConfig:
    """Wrapper method to convert `plot_feature_importance_by_feature_plotly` to
    AxPlotConfig"""
    return AxPlotConfig(
        # pyre-fixme[6]: For 1st argument expected `Dict[str, typing.Any]` but got
        #  `Figure`.
        data=plot_feature_importance_by_feature_plotly(
            model=model,
            sensitivity_values=sensitivity_values,
            relative=relative,
            caption=caption,
            importance_measure=importance_measure,
            label_dict=label_dict,
        ),
        plot_type=AxPlotTypes.GENERIC,
    )


def plot_relative_feature_importance_plotly(model: ModelBridge) -> go.Figure:
    """Create a stacked bar chart of feature importances per metric"""
    importances = []
    for metric_name in sorted(model.metric_names):
        try:
            vals: dict[str, Any] = model.feature_importances(metric_name)
            vals["index"] = metric_name
            importances.append(vals)
        except Exception:
            logger.warning(
                f"Model for {metric_name} does not support feature importances."
            )
    df = pd.DataFrame(importances)
    df.set_index("index", inplace=True)
    df = df.div(df.sum(axis=1), axis=0)
    data = [
        go.Bar(y=df.index, x=df[column_name], name=column_name, orientation="h")
        for column_name in df.columns
    ]
    layout = go.Layout(
        margin=go.layout.Margin(l=250),  # noqa E741
        barmode="group",
        yaxis={"title": ""},
        xaxis={"title": "Relative Parameter importance"},
        showlegend=False,
        title="Relative Parameter Importance per Metric",
    )
    return go.Figure(data=data, layout=layout)


def plot_relative_feature_importance(model: ModelBridge) -> AxPlotConfig:
    """Wrapper method to convert plot_relative_feature_importance_plotly to
    AxPlotConfig"""
    return AxPlotConfig(
        # pyre-fixme[6]: For 1st argument expected `Dict[str, typing.Any]` but got
        #  `Figure`.
        data=plot_relative_feature_importance_plotly(model),
        plot_type=AxPlotTypes.GENERIC,
    )

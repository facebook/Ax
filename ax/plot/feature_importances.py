#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import pandas as pd
import plotly.graph_objs as go
from ax.exceptions.core import NoDataError
from ax.modelbridge import ModelBridge
from ax.plot.base import AxPlotConfig, AxPlotTypes
from ax.utils.common.logger import get_logger
from plotly import subplots


logger = get_logger("FeatureImportance")


def plot_feature_importance(df: pd.DataFrame, title: str) -> AxPlotConfig:
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
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)


def plot_feature_importance_by_metric(model: ModelBridge) -> AxPlotConfig:
    """One plot per feature, showing importances by metric."""
    importances = []
    for metric_name in sorted(model.metric_names):
        try:
            vals: Dict[str, Any] = model.feature_importances(metric_name)
            # pyre-fixme[6]: Expected `float` for 2nd param but got `str`.
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

    plot_fi = plot_feature_importance(df, "Absolute Feature Importances by Metric")
    num_subplots = len(df.columns) - 1  # one column is the index
    num_features = len(df)
    plot_fi.data["layout"]["height"] = num_subplots * num_features * 50
    return plot_fi


def plot_feature_importance_by_feature(
    model: ModelBridge, relative: bool = True
) -> AxPlotConfig:
    """One plot per metric, showing importances by feature."""
    traces = []
    dropdown = []
    for i, metric_name in enumerate(sorted(model.metric_names)):
        try:
            importances = model.feature_importances(metric_name)
        except NotImplementedError:
            logger.warning(
                f"Model for {metric_name} does not support feature importances."
            )
            continue

        df = pd.DataFrame(
            [
                {"Factor": factor, "Importance": importance}
                for factor, importance in importances.items()
            ]
        )

        if relative:
            df["Importance"] = df["Importance"].div(df["Importance"].sum())

        df = df.sort_values("Importance")
        traces.append(
            go.Bar(
                name="Importance",
                orientation="h",
                visible=i == 0,
                x=df["Importance"],
                y=df["Factor"],
            )
        )

        is_visible = [False] * len(sorted(model.metric_names))
        is_visible[i] = True
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
    features = traces[0].y
    title = (
        "Relative Feature Importances" if relative else "Absolute Feature Importances"
    )
    layout = go.Layout(
        height=200 + len(features) * 20,
        hovermode="closest",
        margin=go.layout.Margin(
            l=8 * min(max(len(idx) for idx in features), 75)  # noqa E741
        ),
        showlegend=False,
        title=title,
        updatemenus=updatemenus,
    )

    if relative:
        layout.update({"xaxis": {"tickformat": ".0%"}})

    fig = go.Figure(data=traces, layout=layout)
    plot_fi = AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)
    return plot_fi


def plot_relative_feature_importance(model: ModelBridge) -> AxPlotConfig:
    """Create a stacked bar chart of feature importances per metric"""
    importances = []
    for metric_name in sorted(model.metric_names):
        try:
            vals: Dict[str, Any] = model.feature_importances(metric_name)
            # pyre-fixme[6]: Expected `float` for 2nd param but got `str`.
            vals["index"] = metric_name
            importances.append(vals)
        except Exception:
            logger.warning(
                "Model for {} does not support feature importances.".format(metric_name)
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
        barmode="grouped",
        yaxis={"title": ""},
        xaxis={"title": "Relative Feature importance"},
        showlegend=False,
        title="Relative Feature Importance per Metric",
    )
    fig = go.Figure(data=data, layout=layout)
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)

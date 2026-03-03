# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.analysis.healthcheck.predictable_metrics import DEFAULT_MODEL_FIT_THRESHOLD
from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from plotly import graph_objects as go, io as pio


class MetricR2AnalysisCard(PlotlyAnalysisCard):
    """A PlotlyAnalysisCard that displays a table of metric R² values
    with green highlighting for metrics that meet the model fit threshold."""


def create_metric_r2_analysis_card(
    r2s: dict[str, float],
    threshold: float = DEFAULT_MODEL_FIT_THRESHOLD,
) -> MetricR2AnalysisCard:
    """Create a MetricR2AnalysisCard from a dictionary of metric R² values.

    Args:
        r2s: Dictionary mapping metric names to their R² values.
        threshold: R² threshold for highlighting a metric as having
            good model fit. Defaults to DEFAULT_MODEL_FIT_THRESHOLD.

    Returns:
        A MetricR2AnalysisCard with a table of metric R² values.
    """
    metric_names = list(r2s.keys())
    r2_values = [f"{v:.2f}" for v in r2s.values()]

    fill_colors = [
        "rgba(0, 200, 0, 0.15)" if r2 >= threshold else "white" for r2 in r2s.values()
    ]

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[4, 1],
                header={
                    "values": ["Metric", "R\u00b2"],
                    "align": "left",
                },
                cells={
                    "values": [metric_names, r2_values],
                    "align": "left",
                    "fill_color": [fill_colors, fill_colors],
                },
            )
        ]
    )

    return MetricR2AnalysisCard(
        name="MetricR2Summary",
        title="Summary of model fits",
        subtitle=(
            "R\u00b2 (coefficient of determination) measures how well the model"
            " predicts each metric. Higher values indicate better model fit."
            f" Metrics with R\u00b2 >= {threshold} are highlighted in green."
        ),
        df=pd.DataFrame({"Metric": metric_names, "R\u00b2": list(r2s.values())}),
        blob=pio.to_json(fig),
    )

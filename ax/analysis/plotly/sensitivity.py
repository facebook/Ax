# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Literal, Sequence

import pandas as pd
from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.utils import extract_relevant_adapter
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from ax.modelbridge.torch import TorchAdapter
from ax.utils.sensitivity.sobol_measures import ax_parameter_sens
from plotly import express as px, graph_objects as go
from pyre_extensions import override


class SensitivityAnalysisPlot(PlotlyAnalysis):
    def __init__(
        self,
        metric_names: Sequence[str] | None = None,
        order: Literal["first", "second", "total"] = "total",
        top_k: int | None = None,
    ) -> None:
        self.metric_names = metric_names
        self.order = order
        self.top_k = top_k

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[PlotlyAnalysisCard]:
        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        if not isinstance(relevant_adapter, TorchAdapter):
            raise UserInputError(
                "SensitivityAnalysisPlot requires a TorchAdapter, found "
                f"{type(adapter)}."
            )

        data = _prepare_data(
            adapter=relevant_adapter,
            metric_names=self.metric_names,
            order=self.order,
        )

        cards = []
        for metric_name in data["metric_name"].unique():
            df, fig = _prepare_card_components(
                data=data, metric_name=metric_name, top_k=self.top_k
            )

            card = self._create_plotly_analysis_card(
                title=f"Sensitivity Analysis for {metric_name}",
                subtitle=(
                    f"Understand how each parameter affects {metric_name} according to "
                    f"a {self.order}-order sensitivity analysis."
                ),
                level=AnalysisCardLevel.MID,
                category=AnalysisCardCategory.INSIGHT,
                df=df,
                fig=fig,
            )

            cards.append(card)

        return cards


def _prepare_data(
    adapter: TorchAdapter,
    metric_names: Sequence[str] | None,
    order: Literal["first", "second", "total"],
) -> pd.DataFrame:
    sensitivities = ax_parameter_sens(
        model_bridge=adapter,
        metrics=[*metric_names] if metric_names is not None else None,
        order=order,
    )

    return pd.DataFrame.from_records(
        [
            {
                "metric_name": metric_name,
                "parameter_name": parameter_name,
                "sensitivity": sensitivity,
            }
            for metric_name, sensitivity_dict in sensitivities.items()
            for parameter_name, sensitivity in sensitivity_dict.items()
        ]
    )


def _prepare_card_components(
    data: pd.DataFrame, metric_name: str, top_k: int | None
) -> tuple[pd.DataFrame, go.Figure]:
    plotting_df = data.loc[data["metric_name"] == metric_name][
        ["parameter_name", "sensitivity"]
    ].copy()

    plotting_df["importance"] = plotting_df["sensitivity"].abs()
    plotting_df["direction"] = plotting_df["sensitivity"].apply(
        lambda x: f"Increases {metric_name}" if x >= 0 else f"Decreases {metric_name}"
    )

    blue = px.colors.qualitative.Plotly[0]
    orange = px.colors.qualitative.Plotly[4]
    figure = px.bar(
        plotting_df.sort_values(by="importance", ascending=False)
        .reset_index()
        .head(top_k),
        x="importance",
        y="parameter_name",
        orientation="h",
        color="direction",
        color_discrete_map={
            f"Increases {metric_name}": blue,
            f"Decreases {metric_name}": orange,
        },
        hover_data=["parameter_name", "sensitivity"],
    )

    # Display most important parameters first
    figure.update_layout(yaxis={"categoryorder": "total ascending"})

    return (
        plotting_df[["parameter_name", "sensitivity"]],
        figure,
    )

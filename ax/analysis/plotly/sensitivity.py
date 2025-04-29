# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Literal, Mapping, Sequence

import pandas as pd
from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.utils import truncate_label
from ax.analysis.utils import extract_relevant_adapter
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from ax.modelbridge.torch import TorchAdapter
from ax.utils.sensitivity.sobol_measures import ax_parameter_sens
from plotly import express as px, graph_objects as go
from pyre_extensions import override

# SensitivityAnalysisPlot uses a plotly bar chart which needs especially short labels
MAX_LABEL_LEN: int = 20


class SensitivityAnalysisPlot(PlotlyAnalysis):
    """
    Compute sensitivity for all metrics on a TorchAdapter.

    Sobol measures are always positive regardless of the direction in which the
    parameter influences f. If `signed` is set to True, then the Sobol measure for each
    parameter will be given as its sign the sign of the average gradient with respect to
    that parameter across the search space. Thus, important parameters that, when
    increased, decrease will have large and negative values; unimportant parameters
    will have values close to 0.
    """

    def __init__(
        self,
        metric_names: Sequence[str] | None = None,
        order: Literal["first", "second", "total"] = "total",
        top_k: int | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """
        Args:
            metric_names: The names of the metrics and outcomes for which to compute
                sensitivities. This should preferably be metrics with a good model fit.
                Defaults to all metrics in the experiment.
            order: A string specifying the order of the Sobol indices to be computed.
                Supports "first" and "total" and defaults to "first".
            top_k: Optional limit on the number of parameters to show in the plot.
            labels: A mapping from metric names to labels to use in the plot. If a label
                is not provided for a metric, the metric name will be used.
        """
        self.metric_names = metric_names
        self.order = order
        self.top_k = top_k
        self.labels: dict[str, str] = {**labels} if labels is not None else {}

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
            # If a human readable metric name is provided, use it
            metric_label = self.labels.get(
                metric_name, truncate_label(label=metric_name, n=MAX_LABEL_LEN)
            )
            df, fig = _prepare_card_components(
                data=data,
                metric_name=metric_name,
                top_k=self.top_k,
                metric_label=metric_label,
            )

            card = self._create_plotly_analysis_card(
                title=f"Sensitivity Analysis for {metric_label}",
                subtitle=(
                    f"Understand how each parameter affects {metric_label} according "
                    f"to a {self.order}-order sensitivity analysis."
                ),
                level=AnalysisCardLevel.MID,
                category=AnalysisCardCategory.INSIGHT,
                df=df,
                fig=fig,
            )

            cards.append(card)

        return cards


def compute_sensitivity_adhoc(
    adapter: Adapter | None = None,
    metric_names: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    order: Literal["first", "second", "total"] = "total",
    top_k: int | None = None,
) -> list[PlotlyAnalysisCard]:
    """
    Compute SensitivityAnalysis cards for the given experiment and either Adapter or
    GenerationStrategy.

    Note that cards are not saved to the database when computed adhoc -- they are only
    saved when computed as part of call to ``Client.compute_analyses`` or equivalent.

    Args:
        adapter: The adapter to use to compute the analysis. If not provided, will use
            the current adapter on the ``GenerationStrategy``.
        metric_names: The names of the metrics and outcomes for which to compute
                sensitivities. This should preferably be metrics with a good model fit.
                Defaults to all metrics in the experiment.
        order: A string specifying the order of the Sobol indices to be computed.
            Supports "first" and "total" and defaults to "first".
        top_k: Optional limit on the number of parameters to show in the plot.
        labels: A mapping from metric names to labels to use in the plot. If a label
            is not provided for a metric, the metric name will be used.
    """
    analysis = SensitivityAnalysisPlot(
        metric_names=metric_names,
        order=order,
        top_k=top_k,
        labels=labels,
    )
    return [*analysis.compute(adapter=adapter)]


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
    data: pd.DataFrame,
    metric_name: str,
    metric_label: str,
    top_k: int | None,
) -> tuple[pd.DataFrame, go.Figure]:
    plotting_df = data.loc[data["metric_name"] == metric_name][
        ["parameter_name", "sensitivity"]
    ].copy()

    # If the parameter name is too long, truncate it.
    # If the parameter name is a second order interaction, truncate each parameter name
    # separately then re-combine.
    plotting_df["truncated_parameter_name"] = plotting_df["parameter_name"].apply(
        lambda label: " & ".join(
            truncate_label(label=sub_label, n=MAX_LABEL_LEN // 2)
            for sub_label in label.split(" & ")
        )
        if "&" in label
        else truncate_label(label=label, n=MAX_LABEL_LEN)
    )

    plotting_df["importance"] = plotting_df["sensitivity"].abs()
    plotting_df["direction"] = plotting_df["sensitivity"].apply(
        lambda x: f"Increases {metric_label}" if x >= 0 else f"Decreases {metric_label}"
    )

    blue = px.colors.qualitative.Plotly[0]
    orange = px.colors.qualitative.Plotly[4]
    figure = px.bar(
        plotting_df.sort_values(by="importance", ascending=False)
        .reset_index()
        .head(top_k),
        x="importance",
        y="truncated_parameter_name",
        orientation="h",
        color="direction",
        color_discrete_map={
            f"Increases {metric_label}": blue,
            f"Decreases {metric_label}": orange,
        },
        # Show full parameter name on hover, not truncated name
        hover_data=["parameter_name"],
    )

    # Display most important parameters first
    figure.update_layout(yaxis={"categoryorder": "total ascending"})

    return (
        plotting_df[["parameter_name", "sensitivity"]],
        figure,
    )

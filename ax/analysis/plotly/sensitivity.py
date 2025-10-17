# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Literal, Mapping, Sequence

import pandas as pd
from ax.adapter.base import Adapter
from ax.adapter.torch import TorchAdapter
from ax.analysis.analysis import Analysis
from ax.analysis.analysis_card import AnalysisCard, AnalysisCardGroup
from ax.analysis.plotly.color_constants import COLOR_FOR_DECREASES, COLOR_FOR_INCREASES
from ax.analysis.plotly.plotly_analysis import create_plotly_analysis_card
from ax.analysis.plotly.utils import (
    LEGEND_POSITION,
    MARGIN_REDUCUTION,
    MAX_HOVER_LABEL_LEN,
    select_metric,
    truncate_label,
)
from ax.analysis.utils import extract_relevant_adapter
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.utils.sensitivity.sobol_measures import ax_parameter_sens
from plotly import express as px, graph_objects as go
from pyre_extensions import override

# SensitivityAnalysisPlot uses a plotly bar chart which needs especially short labels
MAX_LABEL_LEN: int = 20

SENSITIVITY_CARDGROUP_TITLE = (
    "Sensitivity Analysis: Understand how each parameter affects metrics"
)

SENSITIVITY_CARDGROUP_SUBTITLE = (
    "These plots showcase the most influential parameters for each metric in the "
    "experiment, highlighting both the direction and magnitude of the metric's "
    "sensitivity to changes in these parameters. This information can be valuable for "
    "understanding metrics that may be oppositely affected by the same parameter, "
    "identifying the most critical parameters to further refine the search space, or "
    "validating underlying assumptions about the experiment's response surface. "
    "Sensitivity is measured using Sobol indices, which are calculated based on the "
    "model fitted to the data."
)


class SensitivityAnalysisPlot(Analysis):
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
        metric_name: str | None = None,
        order: Literal["first", "second", "total"] = "total",
        top_k: int | None = 6,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """
        Args:
            metric_name: The name of the metric to compute sensitivity analysis for.
                If not provided, will compute sensitivity analysis for the objective.
            order: A string specifying the order of the Sobol indices to be computed.
                Supports "first" and "total" and defaults to "first".
            top_k: Optional limit on the number of parameters to show in the plot.
            labels: A mapping from metric names to labels to use in the plot. If a label
                is not provided for a metric, the metric name will be used.
        """
        self.metric_name = metric_name
        self.order = order
        self.top_k = top_k
        self.labels: dict[str, str] = {**labels} if labels is not None else {}

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCard:
        if self.metric_name is None:
            if experiment is None:
                raise UserInputError(
                    "SensitivityAnalysisPlot requires either an a metric name be "
                    "provided or an Experiment be provided to infer the relevant "
                    "metric name."
                )
            metric_name = select_metric(experiment=experiment)
        else:
            metric_name = self.metric_name

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
            metric_name=metric_name,
            order=self.order,
        )

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

        return create_plotly_analysis_card(
            name=self.__class__.__name__,
            title=f"Sensitivity Analysis for {metric_label}",
            subtitle=(
                f"Understand how each parameter affects {metric_label} according "
                f"to a {self.order}-order sensitivity analysis."
            ),
            df=df,
            fig=fig,
        )


def compute_sensitivity_adhoc(
    adapter: Adapter,
    metric_names: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    order: Literal["first", "second", "total"] = "total",
    top_k: int | None = None,
) -> AnalysisCardGroup:
    """
    Compute SensitivityAnalysis cards for the given experiment and either Adapter or
    GenerationStrategy.

    Note that cards are not saved to the database when computed adhoc -- they are only
    saved when computed as part of call to ``Client.compute_analyses`` or equivalent.

    Args:
        adapter: The adapter to use to compute the analysis.
        metric_names: The names of the metrics and outcomes for which to compute
                sensitivities. This should preferably be metrics with a good model fit.
                Defaults to all metrics in the experiment.
        order: A string specifying the order of the Sobol indices to be computed.
            Supports "first" and "total" and defaults to "first".
        top_k: Optional limit on the number of parameters to show in the plot.
        labels: A mapping from metric names to labels to use in the plot. If a label
            is not provided for a metric, the metric name will be used.
    """
    analyis_cards = [
        SensitivityAnalysisPlot(
            metric_name=metric_name,
            order=order,
            top_k=top_k,
            labels=labels,
        ).compute_or_error_card(adapter=adapter)
        for metric_name in (
            metric_names if metric_names is not None else adapter.outcomes
        )
    ]

    return AnalysisCardGroup(
        name="SensitivityAnalysisAdhoc",
        title="Adhoc Sensitivity Analysis",
        subtitle=None,
        children=analyis_cards,
    )


def _prepare_data(
    adapter: TorchAdapter,
    metric_name: str,
    order: Literal["first", "second", "total"],
) -> pd.DataFrame:
    sensitivities = ax_parameter_sens(
        adapter=adapter,
        metrics=[metric_name],
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
    # If the truncated parameter name already exists, append count at end to prevent
    # collisions.
    # TODO: @paschali @mgarrard clean up after implementing parameter canonical names
    param_names = plotting_df["parameter_name"].unique()
    param_to_shortened_name = {}
    shortened_name_count = {}
    for name in param_names:
        shortened_name = (
            " & ".join(
                truncate_label(label=sub_name, n=MAX_LABEL_LEN // 2)
                for sub_name in name.split(" & ")
            )
            if "&" in name
            else truncate_label(label=name, n=MAX_LABEL_LEN)
        )
        # track number of times each shortened name is seen
        if shortened_name not in shortened_name_count:
            shortened_name_count[shortened_name] = 0
        else:
            shortened_name_count[shortened_name] += 1
            shortened_name = shortened_name + f"_{shortened_name_count[shortened_name]}"
        param_to_shortened_name[name] = shortened_name
    plotting_df["truncated_parameter_name"] = plotting_df["parameter_name"].map(
        param_to_shortened_name
    )

    plotting_df["importance"] = plotting_df["sensitivity"].abs()
    plotting_df["direction"] = plotting_df["sensitivity"].apply(
        lambda x: f"Increases {metric_label}" if x >= 0 else f"Decreases {metric_label}"
    )
    figure = px.bar(
        plotting_df.sort_values(by="importance", ascending=False)
        .reset_index()
        .head(top_k),
        x="importance",
        y="truncated_parameter_name",
        orientation="h",
        color="direction",
        color_discrete_map={
            f"Increases {metric_label}": COLOR_FOR_INCREASES,
            f"Decreases {metric_label}": COLOR_FOR_DECREASES,
        },
        # Show longer version of parameter name on hover without overflowing hover
        hover_data=["parameter_name"][:MAX_HOVER_LABEL_LEN],
    )

    figure.update_layout(
        # Display most important parameters first
        yaxis={"categoryorder": "total ascending"},
        # move legend to bottom of plot
        legend=LEGEND_POSITION,
        margin=MARGIN_REDUCUTION,
    )

    return (
        plotting_df[["parameter_name", "sensitivity"]],
        figure,
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any

import numpy as np
import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.utils import select_metric
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.exceptions.core import UserInputError
from plotly import graph_objects as go


class ParallelCoordinatesPlot(PlotlyAnalysis):
    """
    Plotly Parcoords plot for a single metric, with one line per arm and dimensions for
    each parameter in the search space. This plot is useful for understanding how
    thoroughly the search space is explored as well as for identifying if there is any
    clusertering for either good or bad parameterizations.

    The DataFrame computed will contain one row per arm and the following columns:
        - arm_name: The name of the arm
        - METRIC_NAME: The observed mean of the metric specified
        - **PARAMETER_NAME: The value of said parameter for the arm, for each parameter
    """

    def __init__(self, metric_name: str | None = None) -> None:
        """
        Args:
            metric_name: The name of the metric to plot. If not specified the objective
                will be used. Note that the metric cannot be inferred for
                multi-objective or scalarized-objective experiments.
        """

        self.metric_name = metric_name

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> PlotlyAnalysisCard:
        if experiment is None:
            raise UserInputError("ParallelCoordinatesPlot requires an Experiment")

        metric_name = self.metric_name or select_metric(experiment=experiment)

        df = _prepare_data(experiment=experiment, metric=metric_name)
        fig = _prepare_plot(df=df, metric_name=metric_name)

        return self._create_plotly_analysis_card(
            title=f"Parallel Coordinates for {metric_name}",
            subtitle="View arm parameterizations with their respective metric values",
            level=AnalysisCardLevel.HIGH,
            df=df,
            fig=fig,
        )


def _prepare_data(experiment: Experiment, metric: str) -> pd.DataFrame:
    data_df = experiment.lookup_data().df
    filtered_df = data_df.loc[data_df["metric_name"] == metric]

    if filtered_df.empty:
        raise ValueError(f"No data found for metric {metric}")

    records = [
        {
            "arm_name": arm.name,
            **arm.parameters,
            metric: _find_mean_by_arm_name(df=filtered_df, arm_name=arm.name),
        }
        for trial in experiment.trials.values()
        for arm in trial.arms
    ]

    return pd.DataFrame.from_records(records).dropna()


def _prepare_plot(df: pd.DataFrame, metric_name: str) -> go.Figure:
    # ParCoords requires that the dimensions are specified on continuous scales, so
    # ChoiceParameters and FixedParameters must be preprocessed to allow for
    # appropriate plotting.
    parameter_dimensions = [
        _get_parameter_dimension(series=df[col])
        for col in df.columns
        if col != "arm_name" and col != metric_name
    ]

    return go.Figure(
        go.Parcoords(
            line={"color": df[metric_name], "showscale": True},
            dimensions=[
                *parameter_dimensions,
                {
                    "label": _truncate_label(label=metric_name),
                    "values": df[metric_name].tolist(),
                },
            ],
            # Rotate the labels to allow them to be longer withoutoverlapping
            labelangle=-45,
        )
    )


def _find_mean_by_arm_name(
    df: pd.DataFrame,
    arm_name: str,
) -> float:
    # Given a dataframe with arm_name and mean columns, find the mean for a given
    # arm_name. If an arm_name is not found (as can happen if the arm is still running
    # or has failed) return NaN.
    series = df.loc[df["arm_name"] == arm_name]["mean"]

    if series.empty:
        return np.nan

    return series.item()


def _get_parameter_dimension(series: pd.Series) -> dict[str, Any]:
    # For numeric parameters allow Plotly to infer tick attributes. Note: booleans are
    # considered numeric, but in this case we want to treat them as categorical.
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return {
            "tickvals": None,
            "ticktext": None,
            "label": _truncate_label(label=str(series.name)),
            "values": series.tolist(),
        }

    # For non-numeric parameters, sort, map onto an integer scale, and provide
    # corresponding tick attributes
    mapping = {v: k for k, v in enumerate(sorted(series.unique()))}

    return {
        "tickvals": [_truncate_label(label=str(val)) for val in mapping.values()],
        "ticktext": [_truncate_label(label=str(key)) for key in mapping.keys()],
        "label": _truncate_label(label=str(series.name)),
        "values": series.map(mapping).tolist(),
    }


def _truncate_label(label: str, n: int = 18) -> str:
    if len(label) > n:
        return label[:n] + "..."
    return label

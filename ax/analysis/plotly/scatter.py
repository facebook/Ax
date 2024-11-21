# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.exceptions.core import DataRequiredError, UserInputError
from plotly import express as px, graph_objects as go


class ScatterPlot(PlotlyAnalysis):
    """
    Plotly Scatter plot for any two metrics. Each arm is represented by a single point,
    whose color indicates the arm's trial index. Optionally, the Pareto frontier can be
    shown. This plot is useful for understanding the relationship and/or tradeoff
    between two metrics.

    The DataFrame computed will contain one row per arm and the following columns:
        - trial_index: The trial index of the arm
        - arm_name: The name of the arm
        - X_METRIC_NAME: The observed mean of the metric specified
        - Y_METRIC_NAME: The observed mean of the metric specified
        - is_optimal: Whether the arm is on the Pareto frontier
    """

    def __init__(
        self, x_metric_name: str, y_metric_name: str, show_pareto_frontier: bool = False
    ) -> None:
        """
        Args:
            x_metric_name: The name of the metric to plot on the x-axis.
            y_metric_name: The name of the metric to plot on the y-axis.
            show_pareto_frontier: Whether to show the Pareto frontier for the two
                metrics. Optimization direction is inferred from the Experiment.
        """

        self.x_metric_name = x_metric_name
        self.y_metric_name = y_metric_name

        self.show_pareto_frontier = show_pareto_frontier

    def compute(
        self,
        experiment: Optional[Experiment] = None,
        generation_strategy: Optional[GenerationStrategyInterface] = None,
    ) -> PlotlyAnalysisCard:
        if experiment is None:
            raise UserInputError("ScatterPlot requires an Experiment")

        df = _prepare_data(
            experiment=experiment,
            x_metric_name=self.x_metric_name,
            y_metric_name=self.y_metric_name,
        )
        fig = _prepare_plot(
            df=df,
            x_metric_name=self.x_metric_name,
            y_metric_name=self.y_metric_name,
            show_pareto_frontier=self.show_pareto_frontier,
            x_lower_is_better=experiment.metrics[self.x_metric_name].lower_is_better
            or False,
        )

        return self._create_plotly_analysis_card(
            title=f"Observed {self.x_metric_name} vs. {self.y_metric_name}",
            subtitle="Compare arms by their observed metric values",
            level=AnalysisCardLevel.HIGH,
            df=df,
            fig=fig,
        )


def _prepare_data(
    experiment: Experiment, x_metric_name: str, y_metric_name: str
) -> pd.DataFrame:
    """
    Extract the relevant data from the experiment and prepare it into a dataframe
    formatted in the way expected by _prepare_plot.

    Args:
        experiment: The experiment to extract data from.
        x_metric_name: The name of the metric to plot on the x-axis.
        y_metric_name: The name of the metric to plot on the y-axis.
    """

    # Lookup the data that has already been fetched and attached to the experiment
    data = experiment.lookup_data().df

    # Filter for only rows with the relevant metric names
    metric_name_mask = data["metric_name"].isin([x_metric_name, y_metric_name])
    filtered = data[metric_name_mask][
        ["trial_index", "arm_name", "metric_name", "mean"]
    ]

    # Pivot the data so that each row is an arm and the columns are the metric names
    pivoted: pd.DataFrame = filtered.pivot_table(
        index=["trial_index", "arm_name"], columns="metric_name", values="mean"
    ).dropna()
    pivoted.reset_index(inplace=True)
    pivoted.columns.name = None

    if pivoted.empty:
        raise DataRequiredError(
            f"No observations have data for both {x_metric_name} and {y_metric_name}. "
            "Please ensure that the data has been fetched and attached to the "
            "experiment."
        )

    # Add a column indicating whether the arm is on the Pareto frontier. This is
    # calculated by comparing each arm to all other arms in the experiment and
    # creating a mask.
    # If directional guidance is not specified, we assume that we intendt to maximize
    # the metric.
    x_lower_is_better: bool = experiment.metrics[x_metric_name].lower_is_better or False
    y_lower_is_better: bool = experiment.metrics[y_metric_name].lower_is_better or False

    def is_optimal(row: pd.Series) -> bool:
        x_mask = (
            (pivoted[x_metric_name] < row[x_metric_name])
            if x_lower_is_better
            else (pivoted[x_metric_name] > row[x_metric_name])
        )
        y_mask = (
            (pivoted[y_metric_name] < row[y_metric_name])
            if y_lower_is_better
            else (pivoted[y_metric_name] > row[y_metric_name])
        )
        return not (x_mask & y_mask).any()

    pivoted["is_optimal"] = pivoted.apply(
        is_optimal,
        axis=1,
    )

    return pivoted


def _prepare_plot(
    df: pd.DataFrame,
    x_metric_name: str,
    y_metric_name: str,
    show_pareto_frontier: bool,
    x_lower_is_better: bool,
) -> go.Figure:
    """
    Prepare a scatter plot for the given DataFrame.

    Args:
        df: The DataFrame to plot. Must contain the following columns:
            - trial_index: The trial index of the arm
            - arm_name: The name of the arm
            - X_METRIC_NAME: The observed mean of some metric to plot on the x-axis
            - Y_METRIC_NAME: The observed mean of the metric to plot on the y-axis
            - is_optimal: Whether the arm is on the Pareto frontier (this can be
                omitted if show_pareto_frontier=False)
        x_metric_name: The name of the metric to plot on the x-axis
        y_metric_name: The name of the metric to plot on the y-axis
        show_pareto_frontier: Whether to draw the Pareto frontier for the two metrics
        x_lower_is_better: Whether the metric on the x-axis is being minimized (only
            relevant if show_pareto_frontier=True)
    """
    fig = px.scatter(
        df,
        x=x_metric_name,
        y=y_metric_name,
        color="trial_index",
        hover_data=["trial_index", "arm_name", x_metric_name, y_metric_name],
    )

    if show_pareto_frontier:
        # Must sort to ensure we draw the line through optimal points in the correct
        # order.
        frontier_df = df[df["is_optimal"]].sort_values(by=x_metric_name)

        fig.add_trace(
            go.Scatter(
                x=frontier_df[x_metric_name],
                y=frontier_df[y_metric_name],
                mode="lines",
                line_shape="hv" if x_lower_is_better else "vh",
                showlegend=False,
            )
        )

    return fig

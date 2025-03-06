# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import pandas as pd
from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.core.experiment import Experiment
from ax.exceptions.core import DataRequiredError, UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from plotly import graph_objects as go


class ScatterPlot(PlotlyAnalysis):
    """
    Plotly Scatter plot for any two metrics. Each arm is represented by a single point,
    whose color indicates the arm's trial index. Only completed trials are shown.

    Optionally, the Pareto frontier can be shown. This plot is useful for understanding
    the relationship and/or tradeoff between two metrics.

    The DataFrame computed will contain one row per arm and the following columns:
        - trial_index: The trial index of the arm
        - arm_name: The name of the arm
        - X_METRIC_NAME: The observed mean of the metric specified
        - Y_METRIC_NAME: The observed mean of the metric specified
        - is_optimal: Whether the arm is on the Pareto frontier
    """

    def __init__(
        self,
        x_metric_name: str,
        y_metric_name: str,
        show_pareto_frontier: bool = False,
        trial_index: int | None = None,
    ) -> None:
        """
        Args:
            x_metric_name: The name of the metric to plot on the x-axis.
            y_metric_name: The name of the metric to plot on the y-axis.
            show_pareto_frontier: Whether to show the Pareto frontier for the two
                metrics. Optimization direction is inferred from the Experiment.
            trial_index: Optional trial index to filter the data to. If not specified,
                all trials will be included.
        """

        self.x_metric_name = x_metric_name
        self.y_metric_name = y_metric_name
        self.show_pareto_frontier = show_pareto_frontier
        self.trial_index = trial_index

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
    ) -> PlotlyAnalysisCard:
        if experiment is None:
            raise UserInputError("ScatterPlot requires an Experiment")

        df = _prepare_data(
            experiment=experiment,
            x_metric_name=self.x_metric_name,
            y_metric_name=self.y_metric_name,
            trial_index=self.trial_index,
        )
        fig = _prepare_plot(
            df=df,
            x_metric_name=self.x_metric_name,
            y_metric_name=self.y_metric_name,
            show_pareto_frontier=self.show_pareto_frontier,
            x_lower_is_better=experiment.metrics[self.x_metric_name].lower_is_better
            or False,
            trial_index=self.trial_index,
        )

        return self._create_plotly_analysis_card(
            title=f"Observed {self.x_metric_name} vs. {self.y_metric_name}",
            subtitle="Compare arms by their observed metric values",
            level=AnalysisCardLevel.HIGH,
            df=df,
            fig=fig,
            category=AnalysisCardCategory.INSIGHT,
        )


def _prepare_data(
    experiment: Experiment,
    x_metric_name: str,
    y_metric_name: str,
    trial_index: int | None = None,
) -> pd.DataFrame:
    """
    Extract the relevant data from the experiment and prepare it into a dataframe
    formatted in the way expected by _prepare_plot.

    Args:
        experiment: The experiment to extract data from.
        x_metric_name: The name of the metric to plot on the x-axis.
        y_metric_name: The name of the metric to plot on the y-axis.
        trial_index: Optional trial index to filter the data to. If not specified,
                all trials will be included.
    """
    # Lookup the data that has already been fetched and attached to the experiment
    data = experiment.lookup_data().df

    # Filter for only rows with the relevant metric names and only completed trials
    metric_name_mask = data["metric_name"].isin([x_metric_name, y_metric_name])
    status_mask = data["trial_index"].apply(
        lambda trial_index: experiment.trials[trial_index].status.is_completed
    )
    filtered = data[metric_name_mask & status_mask][
        ["trial_index", "arm_name", "metric_name", "mean", "sem"]
    ]

    # filter data to trial index if specified
    if trial_index is not None:
        filtered = filtered[filtered["trial_index"] == trial_index]

    # Pivot the data so that each row is an arm and the columns are the metric names
    # and the SEMs for each metric.
    pivoted_mean: pd.DataFrame = filtered.pivot_table(
        index=["trial_index", "arm_name"], columns="metric_name", values="mean"
    ).dropna()
    pivoted_sem: pd.DataFrame = filtered.pivot_table(
        index=["trial_index", "arm_name"], columns="metric_name", values="sem"
    ).dropna()
    pivoted = pivoted_mean.join(pivoted_sem, rsuffix="_sem")
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
    # If directional guidance is not specified, assume higher is better
    x_lower_is_better: bool = experiment.metrics[x_metric_name].lower_is_better or False
    y_lower_is_better: bool = experiment.metrics[y_metric_name].lower_is_better or False

    pivoted["is_optimal"] = pivoted.apply(
        lambda row: not (
            (
                (pivoted[x_metric_name] < row[x_metric_name])
                if x_lower_is_better
                else (pivoted[x_metric_name] > row[x_metric_name])
            )
            & (
                (pivoted[y_metric_name] < row[y_metric_name])
                if y_lower_is_better
                else (pivoted[y_metric_name] > row[y_metric_name])
            )
        ).any(),
        axis=1,
    )

    return pivoted


def _prepare_plot(
    df: pd.DataFrame,
    x_metric_name: str,
    y_metric_name: str,
    show_pareto_frontier: bool,
    x_lower_is_better: bool,
    trial_index: int | None = None,
) -> go.Figure:
    """
    Prepare a scatter plot for the given DataFrame.

    Args:
        df: The DataFrame to plot. Must contain the following columns:
            - trial_index: The trial index of the arm
            - arm_name: The name of the arm
            - X_METRIC_NAME: The observed mean of some metric to plot on the x-axis
            - Y_METRIC_NAME: The observed mean of the metric to plot on the y-axis
            - X_METRIC_NAME_SEM: The SEM of the observed mean of the x-axis metric
            - Y_METRIC_NAME_SEM: The SEM of the observed mean of the y-axis metric
            - is_optimal: Whether the arm is on the Pareto frontier (this can be
                omitted if show_pareto_frontier=False)
        x_metric_name: The name of the metric to plot on the x-axis
        y_metric_name: The name of the metric to plot on the y-axis
        show_pareto_frontier: Whether to draw the Pareto frontier for the two metrics
        x_lower_is_better: Whether the metric on the x-axis is being minimized (only
            relevant if show_pareto_frontier=True)
        trial_index: Optional trial index to filter the data to. If not specified,
                all trials will be included.
    """
    fig = go.Figure(
        go.Scatter(
            x=df[x_metric_name],
            y=df[y_metric_name],
            mode="markers",
            marker={
                "color": "rgba(0, 0, 255, 0.3)",  # partially transparent blue
            },
            error_x={
                "type": "data",
                "array": df[f"{x_metric_name}_sem"] * 1.96,
                "visible": True,
                "color": "rgba(0, 0, 255, 0.2)",  # Semi-transparent blue
            },
            error_y={
                "type": "data",
                "array": df[f"{y_metric_name}_sem"] * 1.96,
                "visible": True,
                "color": "rgba(0, 0, 255, 0.2)",  # Semi-transparent blue
            },
            hoverlabel={
                "bgcolor": "rgba(0, 0, 255, 0.2)",  # partially transparent blue
                "font": {"color": "black"},
            },
            hoverinfo="text",
            text=df.apply(
                lambda row: (
                    f"Trial: {row['trial_index']}<br>"
                    + f"Arm: {row['arm_name']}<br>"
                    + f"{x_metric_name}: {row[x_metric_name]}<br>"
                    + f"{y_metric_name}: {row[y_metric_name]}"
                ),
                axis=1,
            ),
            showlegend=False,
        )
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

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import plotly.express as px
from ax.analysis.analysis import AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.utils import select_metric
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from plotly import graph_objects as go


class ProgressionPlot(PlotlyAnalysis):
    """
    Plotly Scatter showing a timerseries-like metric's progression, with one line for
    each arm. The plot also includes a marker on the terminal step of any trial that
    was early stopped.

    The DataFrame computed will contain one row per arm and the following columns:
        - trial_index: The trial index of the arm
        - arm_name: The name of the arm
        - METRIC_NAME: The observed mean of the metric specified
        - progression: The progression at which the metric was observed
    """

    def __init__(self, metric_name: str | None = None) -> None:
        """
        Args:
            metric_name: The name of the metric to plot. If not specified the objective
                will be used. Note that the metric cannot be inferred for
                multi-objective or scalarized-objective experiments.
        """

        self._metric_name = metric_name

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
    ) -> PlotlyAnalysisCard:
        if experiment is None:
            raise UserInputError("ProgressionPlot requires an Experiment")

        data = experiment.lookup_data()
        if not isinstance(data, MapData):
            raise UserInputError("ProgressionPlot requires MapData")
        if len(data.map_key_infos) != 1:
            raise UserInputError("ProgressionPlot requires a single map key on MapData")
        map_key = data.map_key_infos[0].key

        metric_name = self._metric_name or select_metric(experiment=experiment)

        # Collect the data necessary to plot each progression curve.
        map_df = data.map_df
        df = map_df.loc[
            map_df["metric_name"] == metric_name,
            ["trial_index", "arm_name", "mean", map_key],
        ].rename(columns={map_key: "progression", "mean": metric_name})

        # Get the terminal step of each trial that was early stopped so we can place a
        # marker to inform the user.
        data_df = data.df  # Collect dataframe with only the terminal observations
        terminal_points = data_df.loc[
            data_df["trial_index"].isin(
                [
                    trial.index
                    for trial in experiment.trials_by_status[TrialStatus.EARLY_STOPPED]
                ]
            ),
            ["mean", map_key],
        ].rename(columns={map_key: "progression", "mean": metric_name})

        # Plot the progression lines with one curve for each arm.
        fig = px.line(df, x="progression", y=metric_name, color="arm_name")

        # Add a marker for each terminal point on early stopped trials.
        fig.add_trace(
            go.Scatter(
                x=terminal_points["progression"],
                y=terminal_points[metric_name],
                mode="markers",
                showlegend=False,
                line_color="red",
                hoverinfo="none",
            )
        )

        return self._create_plotly_analysis_card(
            title=f"{metric_name} by progression",
            subtitle="Observe how the metric changes as each trial progresses",
            level=AnalysisCardLevel.MID,
            df=df,
            fig=fig,
        )

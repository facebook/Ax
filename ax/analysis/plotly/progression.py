# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Sequence

import numpy as np
import plotly.express as px
from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.utils import select_metric
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter

from plotly import graph_objects as go
from pyre_extensions import assert_is_instance, override


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
        - wallclock_time: The wallclock time at which the metric was observed, in
            seconds and starting at 0 from the first trial's start time.
    """

    def __init__(
        self, metric_name: str | None = None, by_wallclock_time: bool = False
    ) -> None:
        """
        Args:
            metric_name: The name of the metric to plot. If not specified the objective
                will be used. Note that the metric cannot be inferred for
                multi-objective or scalarized-objective experiments.
            wallclock_time: If True, plot the relative wallclock time instead of the
                progression on the x-axis.
        """

        self._metric_name = metric_name
        self._by_wallclock_time = by_wallclock_time

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[PlotlyAnalysisCard]:
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
            ["trial_index", "mean", map_key],
        ].rename(columns={map_key: "progression", "mean": metric_name})

        # Add the wallclock time column
        wallclock_series = _calculate_wallclock_timeseries(
            experiment=experiment, metric_name=metric_name
        )

        # If there is a nan in the wallclock time dict's keys then lookup will fail. If
        # this happens, set the wallclock time to nan and continue.
        df["wallclock_time"] = df.apply(
            lambda row: wallclock_series[row["trial_index"]].get(
                row["progression"], np.nan
            ),
            axis=1,
        )
        if len(terminal_points) > 0:
            terminal_points["wallclock_time"] = terminal_points.apply(
                lambda row: wallclock_series[row["trial_index"]].get(
                    row["progression"], np.nan
                ),
                axis=1,
            )

        # Plot the progression lines with one curve for each arm.
        if self._by_wallclock_time:
            x_axis_name = "wallclock_time"
        else:
            x_axis_name = "progression"

        fig = px.line(df, x=x_axis_name, y=metric_name, color="arm_name")

        # Add a marker for each terminal point on early stopped trials.
        if len(terminal_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=terminal_points[x_axis_name],
                    y=terminal_points[metric_name],
                    mode="markers",
                    marker={"symbol": "x", "color": "red"},
                    showlegend=False,
                    hoverinfo="none",
                )
            )

        return [
            self._create_plotly_analysis_card(
                title=f"{metric_name} by {x_axis_name.replace('_', ' ')}",
                subtitle=(
                    "The progression plot tracks the evolution of each metric "
                    "over the course of the experiment. This visualization is "
                    "typically used to monitor the improvement of metrics over "
                    "Trial iterations, but can also be useful in informing decisions "
                    "about early stopping for Trials."
                ),
                level=AnalysisCardLevel.MID,
                df=df,
                fig=fig,
                category=AnalysisCardCategory.INSIGHT,
            )
        ]


def _calculate_wallclock_timeseries(
    experiment: Experiment,
    metric_name: str,
) -> dict[int, dict[float, float]]:
    """
    Calculate a mapping from each trial index and progression to the time since the
    first trial started, in seconds. Assume that the first trial started at t=0, and
    that progressions are linearly spaced between the start and completion times of
    each trial.

    If a trial does not have either a start or completion time the wallclock time
    cannot be calculated and the value will be nan (which will not be plotted).

    Returns:
        trial_index => (progression => timestamp)
    """
    # Find the earliest start time.
    start_time = min(
        trial.time_run_started.timestamp()
        for trial in experiment.trials.values()
        if trial.time_run_started is not None
    )
    # Calculate all start and completion times relative to the earliest start time.
    # Give nan for trials that don't have a start or completion time.
    relative_timestamps = {
        idx: (
            trial.time_run_started.timestamp() - start_time
            if trial.time_run_started is not None
            else np.nan,
            trial.time_completed.timestamp() - start_time
            if trial.time_completed is not None
            else np.nan,
        )
        for idx, trial in experiment.trials.items()
    }

    data = assert_is_instance(experiment.lookup_data(), MapData)
    df = data.map_df[data.map_df["metric_name"] == metric_name]
    map_key = data.map_key_infos[0].key

    return {
        trial_index: dict(
            zip(
                df[df["trial_index"] == trial_index][map_key].to_numpy(),
                # Map the progressions to linspace if the start and completion times
                # are both available, otherwise map to nans
                np.linspace(
                    relative_timestamps[trial_index][0],
                    relative_timestamps[trial_index][1],
                    len(df[df["trial_index"] == trial_index]),
                )
                if (
                    relative_timestamps[trial_index][0] is not None
                    and relative_timestamps[trial_index][1] is not None
                )
                else np.full(len(df[df["trial_index"] == trial_index]), np.nan),
            )
        )
        for trial_index in experiment.trials.keys()
    }

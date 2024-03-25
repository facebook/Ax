# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional, Tuple

import pandas as pd

from ax.analysis.base_plotly_visualization import BasePlotlyVisualization

from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial

from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.trial import Trial

from plotly import express as px, graph_objs as go


class ParallelCoordinatesPlot(BasePlotlyVisualization):
    def __init__(
        self,
        experiment: Experiment,
        objective_name: Optional[str] = None,
    ) -> None:
        """
        Args:
        experiment: Experiment containing trials to plot
        objective_name: Objective name used to color lines between
            parallel plots
        """

        if not objective_name:
            if not experiment.optimization_config:
                raise ValueError("No objective specified for parallel coordinate")
            objective_name = experiment.optimization_config.objective.metric.name

        self.objective_name = objective_name

        super().__init__(experiment=experiment)

    def get_df(self) -> pd.DataFrame:
        """Strip variables not desired in the final plot
         and truncate names for readability

        Returns:
            df.DataFrame: data frame ready for ingestion by plotly
        """

        data_df = self.experiment.lookup_data().df
        filtered_df = data_df.loc[data_df["metric_name"] == self.objective_name]

        if filtered_df.empty:
            raise ValueError(f"No data found for metric {self.objective_name}")

        def map_trial_to_arms(trial: BaseTrial) -> List[Tuple[Arm, Optional[int]]]:
            if isinstance(trial, BatchTrial):
                return [(arm, trial.index) for arm in trial.arms]
            if isinstance(trial, Trial):
                if not trial.arm:
                    raise ValueError("Trial does not contain an arm")
                return [(trial.arm, trial.index)]
            raise ValueError(f"Unsupported trial type {type(trial)}")

        records = [
            {
                self.objective_name: filtered_df.loc[data_df["arm_name"] == arm.name][
                    "mean"
                ].item(),
                "index": index,
                **arm.parameters,
            }
            for t in self.experiment.trials.values()
            for arm, index in map_trial_to_arms(t)
        ]

        return pd.DataFrame.from_records(records)

    def get_fig(self) -> go.Figure:
        """Plot trials as a parallel coordinates graph

        Returns:
            go.Figure: Parellel coordinates plot of all experiment trials
        """
        df = self.get_df()

        return px.parallel_coordinates(
            df,
            color=self.objective_name,
            dimensions=self.experiment.search_space.parameters.keys(),
        )

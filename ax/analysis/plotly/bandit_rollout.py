# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

import pandas as pd
import plotly.express as px
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.plotly.color_constants import DISCRETE_ARM_SCALE
from ax.analysis.plotly.plotly_analysis import (
    create_plotly_analysis_card,
    PlotlyAnalysisCard,
)
from ax.analysis.plotly.utils import STALE_FAIL_REASON
from ax.analysis.utils import validate_experiment
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.trial_status import NON_STALE_ABANDONED_STATUSES, TrialStatus
from ax.generation_strategy.generation_strategy import GenerationStrategy
from plotly import graph_objects as go
from pyre_extensions import assert_is_instance, none_throws, override


@final
class BanditRollout(Analysis):
    """
    BanditRollout visualizes the distribution of weights across different trials
    and arms in an experiment using a bar plot.
    This class is useful for understanding how weights are allocated to different
    arms over the course of an experiment,
    providing insights into the exploration and exploitation dynamics of a bandit
    algorithm.

    The DataFrame computed will contain one row per arm and the following columns:
        - trial_index: The trial index during which the arm was run
        - arm_name: The name of the arm
        - weight: The weight assigned to the arm in the trial
        - normalized_weight: The weight normalized within each trial
    """

    def __init__(self) -> None:
        """
        Initialize the BanditRollout analysis class.

        This class visualizes the distribution of weights across different trials
        and arms in an experiment using a bar plot, helping to understand the
        exploration and exploitation dynamics of bandit algorithms.
        """
        super().__init__()

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        BanditRollout requires an Experiment with BatchTrials.
        """
        if (
            experiment_invalid_reason := validate_experiment(
                experiment=experiment,
                require_trials=True,
                require_data=False,
            )
        ) is not None:
            return experiment_invalid_reason

        experiment = none_throws(experiment)

        if not all(
            isinstance(trial, BatchTrial) for trial in experiment.trials.values()
        ):
            return "Only Experiments with BatchTrials are supported."

    def _prepare_data(self, experiment: Experiment) -> pd.DataFrame:
        """
        Prepare the data for plotting.

        This function takes an experiment as input, extracts the arm weights from each
        trial, and returns a DataFrame with the trial index, arm name, weight, and
        normalized weight. The normalized weight is calculated by dividing the weight
        of each arm by the total weight of all arms in the same trial.

        Args:
            experiment (Experiment): The experiment to extract data from.

        Returns:
            pd.DataFrame: A DataFrame containing the trial index, arm name, weight, and
                normalized weight for each arm in the experiment.
        """

        data_df: pd.DataFrame = pd.DataFrame()
        trial_index: list[int] = []
        arm_name: list[str] = []
        arm_weight: list[float] = []

        trials = experiment.extract_relevant_trials(
            trial_statuses=list(NON_STALE_ABANDONED_STATUSES)
        )

        for trial in trials:
            batch_trial = assert_is_instance(trial, BatchTrial)
            # Exclude failed trials that failed due to staleness
            if (
                batch_trial.status == TrialStatus.FAILED
                and batch_trial.status_reason is not None
                and batch_trial.status_reason == STALE_FAIL_REASON
            ):
                continue
            for arm, weight in batch_trial.arm_weights.items():
                trial_index.append(trial.index)
                arm_name.append(arm.name)
                arm_weight.append(weight)

        data_df["trial_index"] = trial_index
        data_df["arm_name"] = arm_name
        data_df["arm_weight"] = arm_weight

        data_df["normalized_weight"] = data_df.groupby("trial_index")[
            "arm_weight"
        ].transform(lambda x: x / x.sum())

        return data_df

    def _prepare_plot(self, df: pd.DataFrame, experiment_name: str) -> go.Figure:
        """
        Prepare the plot for rendering.

        This function takes a DataFrame as input, groups the data by trial index
        and arm name, sums the normalized weight, and returns a bar plot using
        Plotly Express. The x-axis represents the trial index, the color
        represents the arm name, and the y-axis represents the normalized weight.

        Args:
            df (pd.DataFrame): The DataFrame containing the trial index, arm name,
                weight, and normalized weight for each arm in the experiment.

        Returns:
            go.Figure: A bar plot representing the distribution of weights across
                different trials and arms in the experiment.
        """
        fig = px.bar(
            df,
            x=df["trial_index"].astype("str"),
            color="arm_name",
            y="normalized_weight",
            title=f"Bandit Rollout Weights by Trial for {experiment_name}",
            labels={
                "x": "Trial Index",
                "normalized_weight": "Normalized Weight",
                "arm_name": "Arm Name",
            },
            color_discrete_sequence=DISCRETE_ARM_SCALE,
        )
        return fig

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> PlotlyAnalysisCard:
        experiment = none_throws(experiment)

        df = self._prepare_data(
            experiment=experiment,
        )
        fig = self._prepare_plot(df=df, experiment_name=experiment.name)

        return create_plotly_analysis_card(
            name=self.__class__.__name__,
            title=f"Bandit Rollout Weights by Trial for {experiment.name}",
            subtitle=(
                "The Bandit Rollout visualization provides a comprehensive "
                "overview of the allocation of weights across different trials "
                "and arms. By representing each trial as a distinct axis, this "
                "plot allows for the examination of exploration and exploitation "
                "dynamics over time. It aids in identifying trends and patterns in "
                "arm performance, offering insights into the effectiveness of the "
                "bandit algorithm. Observing the distribution of weights can "
                "reveal correlations and interactions that contribute to the "
                "success or failure of various strategies, enhancing the "
                "understanding of experimental results."
            ),
            df=df,
            fig=fig,
        )

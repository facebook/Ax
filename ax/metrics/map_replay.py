#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict
from logging import Logger
from typing import Any

import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.map_data import MAP_KEY, MapData
from ax.core.map_metric import MapMetric, MapMetricFetchResult
from ax.core.metric import MetricFetchE
from ax.core.trial import Trial
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Err, Ok
from pyre_extensions import none_throws

logger: Logger = get_logger(__name__)


class MapDataReplayMetric(MapMetric):
    """A metric for replaying historical map data."""

    def __init__(
        self,
        name: str,
        map_data: MapData,
        metric_name: str,
        max_steps_validation: int | None = 200,
        lower_is_better: bool | None = None,
    ) -> None:
        """Inits MapDataReplayMetric.

        Args:
            name: The name of the metric.
            map_data: Historical data to use for replaying. It is assumed that
                there is a single curve (arm) per trial (i.e., no batch trials).
            metric_name: The metric to replay from `map_data`.
            max_steps_validation: If not None, we check to see that the inferred
            scaling factor and offset does not lead to a number of replay steps
                that is larger than `max_steps_validation` for any trial.
            lower_is_better: If True, lower metric values are considered
                desirable.
        """
        self.map_data = map_data
        self.max_steps_validation = max_steps_validation
        self.metric_name: str = metric_name
        # Store pre-processed DataFrame sorted by trial_index and step
        self._replay_df: pd.DataFrame = _prepare_replay_dataframe(
            map_data=map_data, metric_name=self.metric_name
        )
        # Pre-group by trial_index for O(1) trial lookups instead of O(n) filtering
        self._trial_groups: dict[int, pd.DataFrame] = {
            int(trial_idx): group
            for trial_idx, group in self._replay_df.groupby("trial_index")
        }
        # Pre-compute trial statistics using vectorized groupby, then extract
        # offset and scaling_factor once, and store only last_step as a dict
        trial_stats = _compute_trial_stats(self._replay_df)
        self.offset: float = trial_stats["first_step"].min()
        self.scaling_factor: float = _compute_scaling_factor(
            trial_stats=trial_stats, offset=self.offset
        )
        # Store only last_step as dict for O(1) lookups in hot paths
        # Explicitly convert keys to int for consistency with _trial_groups
        self._trial_last_step: dict[int, float] = {
            int(k): float(v) for k, v in trial_stats["last_step"].items()
        }
        self._trial_index_to_step: dict[int, int] = defaultdict(int)
        super().__init__(name=name, lower_is_better=lower_is_better)
        self._validate_replay_feasibility(trial_stats=trial_stats)

    @classmethod
    def is_available_while_running(cls) -> bool:
        return True

    def _validate_replay_feasibility(self, trial_stats: pd.DataFrame) -> None:
        """Check that the offset and scaling factor results in a reasonable number
        of steps for all trials (i.e., we don't want an intractable number of trials
        if (trial_max_step - offset) / scaling_factor is too large).

        Args:
            trial_stats: DataFrame with trial statistics (first_step, last_step,
                num_points). Passed in to avoid recomputing or storing it.
        """
        if self.max_steps_validation is None:
            return

        # Vectorized computation of max steps per trial
        max_steps_per_trial = (
            trial_stats["last_step"] - self.offset
        ) / self.scaling_factor
        max_steps = max_steps_per_trial.max()

        # Find violating trials
        violating = max_steps_per_trial[max_steps_per_trial > self.max_steps_validation]
        if not violating.empty:
            trial_idx = violating.index[0]
            max_steps_trial = violating.iloc[0]
            raise ValueError(
                f"For trial {trial_idx}, the computed offset {self.offset} and "
                f"scaling factor {self.scaling_factor} lead to "
                f"{max_steps_trial} steps, which is larger than "
                f"{self.max_steps_validation} steps to replay."
            )
        logger.debug(
            f"Validated MapReplayMetric {self.name} with "
            f"{len(trial_stats)} trials, scaling factor = "
            f"{self.scaling_factor:.2f}, and offset = {self.offset:.2f}, "
            f"resulting in maximum steps = {max_steps}."
        )

    def has_trial_data(self, trial_idx: int) -> bool:
        """Check if any replay data exists for a given trial."""
        # Use pre-grouped dict for O(1) lookup instead of checking DataFrame index
        return trial_idx in self._trial_groups

    def more_replay_available(self, trial_idx: int) -> bool:
        """Check if more replay data is available for a given trial."""
        trial_max_step = self._trial_last_step.get(trial_idx)
        if trial_max_step is None:
            return False
        current_step = (
            self.offset + self._trial_index_to_step[trial_idx] * self.scaling_factor
        )
        return current_step < trial_max_step

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MapMetricFetchResult:
        try:
            if not isinstance(trial, Trial):
                raise RuntimeError(
                    "Only (non-batch) Trials are supported by "
                    f"{self.__class__.__name__}."
                )
            trial_idx = trial.index
            # Increment the step counter if we can.
            if trial.status.is_running and self.more_replay_available(
                trial_idx=trial_idx
            ):
                self._trial_index_to_step[trial_idx] += 1
            trial_scaled_step = (
                self.offset + self._trial_index_to_step[trial_idx] * self.scaling_factor
            )
            logger.info(f"Trial {trial_idx} is at step {trial_scaled_step}.")

            # Use pre-grouped data for O(1) lookup instead of filtering full DataFrame
            trial_group = self._trial_groups.get(trial_idx)
            if trial_group is None:
                return Ok(value=MapData())

            # Filter only the trial's subset (much smaller than full DataFrame)
            trial_data = trial_group[trial_group[MAP_KEY] <= trial_scaled_step]

            if trial_data.empty:
                return Ok(value=MapData())

            # Create the result DataFrame in one operation
            result_df = pd.DataFrame(
                {
                    "arm_name": none_throws(trial.arm).name,
                    "metric_name": self.name,
                    "mean": trial_data["mean"].values,
                    "sem": trial_data["sem"].values,
                    "trial_index": trial.index,
                    "metric_signature": self.signature,
                    MAP_KEY: trial_data[MAP_KEY].values,
                }
            )

            return Ok(value=MapData(df=result_df))

        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )


def _prepare_replay_dataframe(map_data: MapData, metric_name: str) -> pd.DataFrame:
    """Prepare a pre-sorted DataFrame for efficient replay lookups.

    Filters the data to the specified metric and sorts by trial_index and step.
    This allows efficient vectorized filtering during fetch_trial_data.
    """
    df = map_data.full_df
    df = df[df["metric_name"] == metric_name]
    # Sort once upfront for efficient lookups
    return df.sort_values(
        by=["trial_index", MAP_KEY], ascending=True, ignore_index=True
    )


def _compute_trial_stats(replay_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-trial statistics using vectorized groupby operations.

    Returns a DataFrame indexed by trial_index with columns:
    - first_step: the first (minimum) step value for each trial
    - last_step: the last (maximum) step value for each trial
    - num_points: the number of data points per trial
    """
    stats = replay_df.groupby("trial_index")[MAP_KEY].agg(
        first_step="first",  # Data is pre-sorted, so first/last are min/max
        last_step="last",
        num_points="count",
    )
    return stats


def _compute_scaling_factor(trial_stats: pd.DataFrame, offset: float) -> float:
    """Compute the scaling factor for replay data using vectorized operations.

    The scaling factor is:
    `mean_{trial in trials} (max_steps_trial - offset) / num_points_trial`.
    """
    # Vectorized computation of per-trial scaling factors
    valid_mask = (trial_stats["num_points"] > 0) & (trial_stats["last_step"] > offset)
    if not valid_mask.any():
        return 1.0

    scaling_factors = (
        trial_stats.loc[valid_mask, "last_step"] - offset
    ) / trial_stats.loc[valid_mask, "num_points"]
    scaling_factor = float(scaling_factors.mean())

    return scaling_factor if scaling_factor > 0.0 else 1.0

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict
from dataclasses import dataclass
from logging import Logger
from typing import Any

import numpy as np

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


@dataclass(frozen=True)
class ReplayPoint:
    step: float
    mean: float | None
    sem: float | None


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
        self.replay_data: dict[int, list[ReplayPoint]] = construct_replay_dict(
            map_data=map_data, metric_name=self.metric_name
        )
        self.offset: float = min(r[0].step for r in self.replay_data.values())
        self.scaling_factor: float = compute_scaling_factor(
            replay_data=self.replay_data,
            offset=self.offset,
        )
        self._trial_index_to_step: dict[int, int] = defaultdict(int)
        super().__init__(name=name, lower_is_better=lower_is_better)
        self._validate_replay_feasibility()

    @classmethod
    def is_available_while_running(cls) -> bool:
        return True

    def _validate_replay_feasibility(self) -> None:
        """Check that the offset and scaling factor results in a reasonable number
        of steps for all trials (i.e., we don't want an intractable number of trials
        if (trial_max_step - offset) / scaling_factor is too large).
        """
        if self.max_steps_validation is not None:
            max_steps = 0
            for trial_idx, replay_data in self.replay_data.items():
                max_steps_trial = (
                    replay_data[-1].step - self.offset
                ) / self.scaling_factor
                if max_steps_trial > self.max_steps_validation:  # pyre-ignore[58]
                    raise ValueError(
                        f"For trial {trial_idx}, the computed offset {self.offset} and "
                        f"scaling factor {self.scaling_factor} lead to "
                        f"{max_steps_trial} steps, which is larger than "
                        f"{self.max_steps_validation} steps to replay."
                    )
                max_steps = max(max_steps_trial, max_steps)
            logger.debug(
                f"Validated MapReplayMetric {self.name} with "
                f"{len(self.replay_data)} trials, scaling factor = "
                f"{self.scaling_factor:.2f}, and offset = {self.offset:.2f}, "
                f"resulting in maximum steps = {max_steps}."
            )

    def more_replay_available(self, trial_idx: int) -> bool:
        trial_max_step = self.replay_data[trial_idx][-1].step
        return (
            self.offset + self._trial_index_to_step[trial_idx] * self.scaling_factor
            < trial_max_step
        )

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MapMetricFetchResult:
        try:
            if not isinstance(trial, Trial):
                raise RuntimeError(
                    "Only (non-batch) Trials are supported by "
                    f"{self.__class__.__name__}."
                )
            trial_idx = trial.index
            # increment the step counter if we can
            if trial.status.is_running and self.more_replay_available(
                trial_idx=trial_idx
            ):
                self._trial_index_to_step[trial_idx] += 1
            trial_scaled_step = (
                self.offset + self._trial_index_to_step[trial_idx] * self.scaling_factor
            )
            logger.info(f"Trial {trial_idx} is at step {trial_scaled_step}.")
            datas = []
            for replay_point in self.replay_data[trial_idx]:
                if replay_point.step > trial_scaled_step:
                    break
                df = pd.DataFrame(
                    {
                        "arm_name": [none_throws(trial.arm).name],
                        "metric_name": [self.name],
                        "mean": [replay_point.mean],
                        "sem": [replay_point.sem],
                        "trial_index": [trial.index],
                        "metric_signature": [self.signature],
                        MAP_KEY: [replay_point.step],
                    }
                )
                datas.append(MapData(df=df))

            return Ok(value=MapData.from_multiple_data(data=datas))

        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )


def construct_replay_dict(
    map_data: MapData, metric_name: str
) -> dict[int, list[ReplayPoint]]:
    """Construct a dictionary of replay data mapping trials to
    lists of ReplayPoints.
    """
    map_data_metric = map_data.filter(metric_names=[metric_name])
    map_df = map_data_metric.map_df
    replay_data = defaultdict(list)
    for trial_idx, sub_df in map_df.groupby("trial_index"):
        sub_df = sub_df.sort_values(by=MAP_KEY, ascending=True)
        replay_data[trial_idx] = [
            ReplayPoint(step=step, mean=mean, sem=sem)
            for mean, sem, step in zip(
                sub_df["mean"].tolist(),
                sub_df["sem"].tolist(),
                sub_df[MAP_KEY].tolist(),
            )
        ]
    return replay_data


def compute_scaling_factor(
    replay_data: dict[int, list[ReplayPoint]], offset: float
) -> float:
    """Compute the scaling factor for replay data. The scaling factor is set to be:
    `mean_{trial in trials} (max_steps_trial - offset) / num_points_trial`.
    """
    scaling_factors = []
    for replay_points in replay_data.values():
        num_replay_points = len(replay_points)
        final_replay_step = float(replay_points[-1].step)
        if num_replay_points > 0 and final_replay_step > offset:
            scaling_factor_trial = (final_replay_step - offset) / num_replay_points
            scaling_factors.append(scaling_factor_trial)
    scaling_factor = (
        float(np.mean(scaling_factors)) if len(scaling_factors) > 0 else 1.0
    )
    return scaling_factor if scaling_factor > 0.0 else 1.0

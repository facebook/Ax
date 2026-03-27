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
from ax.core.data import Data, MAP_KEY
from ax.core.map_metric import MapMetric
from ax.core.metric import MetricFetchE, MetricFetchResult
from ax.core.trial import Trial
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Err, Ok
from pyre_extensions import none_throws

logger: Logger = get_logger(__name__)


class MapDataReplayState:
    """Shared state coordinator for replaying historical map data.

    Manages normalized cursor-based progression across multiple metrics
    and trials. The cursor model uses a global min/max MAP_KEY across
    all metrics to preserve cross-metric timing alignment.

    This class serves original MAP_KEY values (not normalized). Downstream
    early stopping strategies apply normalization independently via
    ``_maybe_normalize_map_key`` in ``ax.adapter.data_utils``.
    """

    def __init__(
        self,
        map_data: Data,
        metric_signatures: list[str],
        step_size: float = 0.01,
    ) -> None:
        """Initialize replay state from historical data.

        Args:
            map_data: Historical data containing progression data.
            metric_signatures: List of metric signatures to replay.
            step_size: Cursor increment per advancement step. Determines
                the granularity of replay (e.g. 0.01 = 100 steps).
        """
        self.step_size: float = step_size

        # Pre-index data by (trial_index, metric_signature) for O(1) lookups
        self._data: dict[tuple[int, str], pd.DataFrame] = {}
        all_trial_indices: set[int] = set()
        all_prog_values: list[float] = []
        per_trial_max_prog: dict[int, float] = {}

        for metric_signature in metric_signatures:
            replay_df = _prepare_replay_dataframe(
                map_data=map_data, metric_signature=metric_signature
            )
            for trial_index, group in replay_df.groupby("trial_index"):
                trial_index = int(trial_index)
                self._data[(trial_index, metric_signature)] = group.reset_index(
                    drop=True
                )
                all_trial_indices.add(trial_index)
                prog_values = group[MAP_KEY].values
                all_prog_values.extend(prog_values.tolist())
                trial_max = float(prog_values.max())
                if trial_index in per_trial_max_prog:
                    per_trial_max_prog[trial_index] = max(
                        per_trial_max_prog[trial_index], trial_max
                    )
                else:
                    per_trial_max_prog[trial_index] = trial_max

        if all_prog_values:
            self.min_prog: float = float(min(all_prog_values))
            self.max_prog: float = float(max(all_prog_values))
        else:
            self.min_prog = 0.0
            self.max_prog = 0.0

        self._per_trial_max_prog: dict[int, float] = per_trial_max_prog
        self._trial_cursors: defaultdict[int, float] = defaultdict(float)
        self._trial_indices: set[int] = all_trial_indices

    def advance_trial(self, trial_index: int) -> None:
        """Advance the cursor for a trial by one resolution step."""
        self._trial_cursors[trial_index] = min(
            self._trial_cursors[trial_index] + self.step_size, 1.0
        )

    def has_trial_data(self, trial_index: int) -> bool:
        """Check if any replay data exists for a given trial."""
        return trial_index in self._trial_indices

    def is_trial_complete(self, trial_index: int) -> bool:
        """Check if a trial's cursor has reached its maximum progression."""
        if self.min_prog == self.max_prog:
            return True
        curr_prog = self.min_prog + self._trial_cursors[trial_index] * (
            self.max_prog - self.min_prog
        )
        return curr_prog >= self._per_trial_max_prog.get(trial_index, 0.0)

    def get_data(self, trial_index: int, metric_signature: str) -> pd.DataFrame:
        """Get replay data for a trial up to the current cursor position.

        Returns a DataFrame filtered to rows where MAP_KEY <= current
        progression value, with original (non-normalized) MAP_KEY values.
        """
        df = self._data.get((trial_index, metric_signature))
        if df is None:
            return pd.DataFrame()
        if self.min_prog == self.max_prog:
            return df
        curr_prog = self.min_prog + self._trial_cursors[trial_index] * (
            self.max_prog - self.min_prog
        )
        return df[df[MAP_KEY] <= curr_prog]


class MapDataReplayMetric(MapMetric):
    """A metric for replaying historical map data.

    Delegates data storage and progression state to a shared
    ``MapDataReplayState`` instance, allowing multiple metrics
    to share the same progression timeline.
    """

    def __init__(
        self,
        name: str,
        replay_state: MapDataReplayState,
        metric_signature: str,
        lower_is_better: bool | None = None,
    ) -> None:
        """Initialize a replay metric.

        Args:
            name: The name of this metric in the replay experiment.
            replay_state: Shared state coordinator for replay progression.
            metric_signature: The metric signature to replay from the
                historical data.
            lower_is_better: If True, lower metric values are considered
                desirable.
        """
        self._replay_state: MapDataReplayState = replay_state
        self._metric_signature: str = metric_signature
        super().__init__(name=name, lower_is_better=lower_is_better)

    @classmethod
    def is_available_while_running(cls) -> bool:
        return True

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MetricFetchResult:
        try:
            if not isinstance(trial, Trial):
                raise RuntimeError(
                    "Only (non-batch) Trials are supported by "
                    f"{self.__class__.__name__}."
                )
            trial_data = self._replay_state.get_data(
                trial_index=trial.index,
                metric_signature=self._metric_signature,
            )

            if trial_data.empty:
                return Ok(value=Data())

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

            return Ok(value=Data(df=result_df))

        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )


def _prepare_replay_dataframe(map_data: Data, metric_signature: str) -> pd.DataFrame:
    """Prepare a pre-sorted DataFrame for efficient replay lookups.

    Filters the data to the specified metric signature and sorts by
    trial_index and step.
    """
    df = map_data.full_df
    df = df[df["metric_signature"] == metric_signature]
    # Sort once upfront for efficient lookups
    return df.sort_values(
        by=["trial_index", MAP_KEY], ascending=True, ignore_index=True
    )

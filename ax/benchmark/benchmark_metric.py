# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Metric classes for Ax benchmarking.

Metrics vary on two dimensions: Whether they are `MapMetric`s or not, and
whether they are available while running or not.

There are four Metric classes:
- `BenchmarkMetric`: A non-Map metric
    is not available while running.
- `BenchmarkMapMetric`: For when outputs should be `MapData` (not `Data`) and
    data is available while running.
- `BenchmarkTimeVaryingMetric`: For when outputs should be `Data` and the metric
  is available while running.
- `BenchmarkMapUnavailableWhileRunningMetric`: For when outputs should be
  `MapData` and the metric is not available while running.

Any of these can be used with or without a simulator. However,
`BenchmarkMetric.fetch_trial_data` cannot take in data with multiple time steps,
as they will not be used and this is assumed to be an error. The below table
enumerates use cases.

.. list-table:: Benchmark Metrics Table
   :widths: 5 25 5 5 5 50
   :header-rows: 1

   * -
     - Metric
     - Map
     - Available while running
     - Simulator
     - Reason/use case
   * - 1
     - BenchmarkMetric
     - No
     - No
     - No
     - Vanilla
   * - 2
     - BenchmarkMetric
     - No
     - No
     - Yes
     - Asynchronous, data read only at end
   * - 3
     - BenchmarkTimeVaryingMetric
     - No
     - Yes
     - No
     - Behaves like #1 because it will never be RUNNING
   * - 4
     - BenchmarkTimeVaryingMetric
     - No
     - Yes
     - Yes
     - Scalar data that changes over time
   * - 5
     - BenchmarkMapUnavailableWhileRunningMetric
     - Yes
     - No
     - No
     - MapData that returns immediately; could be used for getting baseline
   * - 6
     - BenchmarkMapUnavailableWhileRunningMetric
     - Yes
     - No
     - Yes
     - Asynchronicity with MapData read only at end
   * - 7
     - BenchmarkMapMetric
     - Yes
     - Yes
     - No
     - Behaves same as #5
   * - 8
     - BenchmarkMapMetric
     - Yes
     - Yes
     - Yes
     - Early stopping
"""

from abc import abstractmethod
from typing import Any

import numpy as np
from ax.benchmark.benchmark_trial_metadata import BenchmarkTrialMetadata
from ax.core.base_trial import BaseTrial
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.map_data import MapData, MapKeyInfo
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.utils.common.result import Err, Ok
from pandas import DataFrame
from pyre_extensions import none_throws


def _get_no_metadata_msg(trial_index: int) -> str:
    return f"No metadata available for trial {trial_index}."


class BenchmarkMetricBase(Metric):
    def __init__(
        self,
        name: str,
        # Needed to be boolean (not None) for validation of MOO opt configs
        lower_is_better: bool,
        observe_noise_sd: bool = True,
    ) -> None:
        """
        Args:
            name: Name of the metric.
            lower_is_better: If `True`, lower metric values are considered better.
            observe_noise_sd: If `True`, the standard deviation of the observation
                noise is included in the `sem` column of the the returned data.
                If `False`, `sem` is set to `None` (meaning that the model will
                have to infer the noise level).
        """
        super().__init__(name=name, lower_is_better=lower_is_better)
        # Declare `lower_is_better` as bool (rather than optional as in the base class)
        self.lower_is_better: bool = lower_is_better
        self.observe_noise_sd: bool = observe_noise_sd

    def _class_specific_metdata_validation(
        self, metadata: BenchmarkTrialMetadata | None
    ) -> None:
        return

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MetricFetchResult:
        """
        Args:
            trial: The trial from which to fetch data.
            kwargs: Unsupported and will raise an exception.

        Returns:
            A MetricFetchResult containing the data for the requested metric.
        """

        class_name = self.__class__.__name__
        if len(kwargs) > 0:
            raise NotImplementedError(
                f"Arguments {set(kwargs)} are not supported in "
                f"{class_name}.fetch_trial_data."
            )
        if isinstance(trial, BatchTrial) and len(trial.abandoned_arms) > 0:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support abandoned arms in "
                "batch trials."
            )
        if len(trial.run_metadata) == 0:
            return Err(
                MetricFetchE(
                    message=_get_no_metadata_msg(trial_index=trial.index),
                    exception=None,
                )
            )

        metadata = trial.run_metadata["benchmark_metadata"]
        self._class_specific_metdata_validation(metadata=metadata)
        backend_simulator = metadata.backend_simulator
        df = metadata.dfs[self.name]

        # Filter out the observable data
        if backend_simulator is None:
            # If there's no backend simulator then no filtering is needed; the
            # trial will complete immediately, with all data available.
            available_data = df
        else:
            sim_trial = none_throws(
                backend_simulator.get_sim_trial_by_index(trial.index)
            )
            # The BackendSimulator distinguishes between queued and running
            # trials "for testing particular initialization cases", but these
            # are all "running" to Scheduler.
            start_time = none_throws(sim_trial.sim_start_time)

            if sim_trial.sim_completed_time is None:  # Still running
                max_t = backend_simulator.time - start_time
            elif sim_trial.sim_completed_time > backend_simulator.time:
                raise RuntimeError(
                    "The trial's completion time is in the future! This is "
                    f"unexpected. {sim_trial.sim_completed_time=}, "
                    f"{backend_simulator.time=}"
                )
            else:
                # Completed, may have stopped early -- can't assume all data available
                completed_time = none_throws(sim_trial.sim_completed_time)
                max_t = completed_time - start_time

            available_data = df[df["virtual runtime"] <= max_t]

        if not self.observe_noise_sd:
            available_data.loc[:, "sem"] = np.nan
        return self._df_to_result(df=available_data.drop(columns=["virtual runtime"]))

    @abstractmethod
    def _df_to_result(self, df: DataFrame) -> MetricFetchResult:
        """
        Convert a DataFrame of observable data to Data or MapData, as
        appropriate for the class.
        """
        ...


class BenchmarkMetric(BenchmarkMetricBase):
    """
    Non-map Metric for benchmarking that is not available while running.

    It cannot process data with multiple time steps, as it would only return one
    value -- the value it has at completion time -- regardless.
    """

    def _class_specific_metdata_validation(
        self, metadata: BenchmarkTrialMetadata | None
    ) -> None:
        if metadata is not None:
            df = metadata.dfs[self.name]
            if df["step"].nunique() > 1:
                raise ValueError(
                    f"Trial has data from multiple time steps. This is"
                    f" not supported by `{self.__class__.__name__}`; use "
                    "`BenchmarkMapMetric`."
                )

    def _df_to_result(self, df: DataFrame) -> MetricFetchResult:
        return Ok(value=Data(df=df.drop(columns=["step"])))


class BenchmarkTimeVaryingMetric(BenchmarkMetricBase):
    """
    Non-Map Metric for benchmarking that is available while running.

    It can produce different values at different times depending on when it is
    called, using the `time` on a `BackendSimulator`.
    """

    @classmethod
    def is_available_while_running(cls) -> bool:
        return True

    def _df_to_result(self, df: DataFrame) -> MetricFetchResult:
        return Ok(
            value=Data(df=df[df["step"] == df["step"].max()].drop(columns=["step"]))
        )


class BenchmarkMapMetric(MapMetric, BenchmarkMetricBase):
    """MapMetric for benchmarking. It is available while running."""

    # pyre-fixme: Inconsistent override [15]: `map_key_info` overrides attribute
    # defined in `MapMetric` inconsistently. Type `MapKeyInfo[int]` is not a
    # subtype of the overridden attribute `MapKeyInfo[float]`
    map_key_info: MapKeyInfo[int] = MapKeyInfo(key="step", default_value=0)

    @classmethod
    def is_available_while_running(cls) -> bool:
        return True

    def _df_to_result(self, df: DataFrame) -> MetricFetchResult:
        # Just in case the key was renamed by a subclass
        df = df.rename(columns={"step": self.map_key_info.key})
        return Ok(value=MapData(df=df, map_key_infos=[self.map_key_info]))


class BenchmarkMapUnavailableWhileRunningMetric(MapMetric, BenchmarkMetricBase):
    # pyre-fixme: Inconsistent override [15]: `map_key_info` overrides attribute
    # defined in `MapMetric` inconsistently. Type `MapKeyInfo[int]` is not a
    # subtype of the overridden attribute `MapKeyInfo[float]`
    map_key_info: MapKeyInfo[int] = MapKeyInfo(key="step", default_value=0)

    def _df_to_result(self, df: DataFrame) -> MetricFetchResult:
        # Just in case the key was renamed by a subclass
        df = df.rename(columns={"step": self.map_key_info.key})
        return Ok(value=MapData(df=df, map_key_infos=[self.map_key_info]))

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Collection
from dataclasses import dataclass, field, InitVar
from pathlib import Path

import pandas as pd

import torch
from ax.benchmark.problems.data import AbstractParquetDataLoader
from ax.benchmark.problems.surrogate.lcbench.utils import (
    DEFAULT_METRIC_NAME,
    get_lcbench_log_scale_parameter_names,
    get_lcbench_parameter_names,
)

DATASET_NAMES = [
    "APSFailure",
    "Amazon_employee_access",
    "Australian",
    "Fashion-MNIST",
    "KDDCup09_appetency",
    "MiniBooNE",
    "adult",
    "airlines",
    "albert",
    "bank-marketing",
    "blood-transfusion-service-center",
    "car",
    "christine",
    "cnae-9",
    "connect-4",
    "covertype",
    "credit-g",
    "dionis",
    "fabert",
    "helena",
    "higgs",
    "jannis",
    "jasmine",
    "jungle_chess_2pcs_raw_endgame_complete",
    "kc1",
    "kr-vs-kp",
    "mfeat-factors",
    "nomao",
    "numerai28.6",
    "phoneme",
    "segment",
    "shuttle",
    "sylvine",
    "vehicle",
    "volkert",
]


class LCBenchDataLoader(AbstractParquetDataLoader):
    def __init__(
        self,
        dataset_name: str,
        stem: str,
        cache_dir: Path | None = None,
    ) -> None:
        super().__init__(
            benchmark_name="LCBenchLite",
            dataset_name=dataset_name,
            stem=stem,
            cache_dir=cache_dir,
        )

    @property
    def url(self) -> str:
        """
        URL to the GZIP compressed parquet files for the 35 datasets from LCBench.
        These files were created by splitting the massive JSON dump of LCBench into
        datasets, then further into config info, learning curve metrics, and final
        results, and subsequently saving them to an efficient Parquet format,
        compressed with GZIP, and finally uploading them to address.
        """

        return (
            "https://raw.githubusercontent.com/ltiao/"
            f"{self.benchmark_name}/main/{self.dataset_name}/{self.filename}"
        )


@dataclass(kw_only=True)
class LCBenchData:
    """
    Args:
        parameter_df: DataFrame with columns corresponding to the names of the
            parameters in get_lcbench_parameter_names().
        metric_series: Series of metric values with index names "trial" and "epoch".
        timestamp_series: Series of timestamps with index name "trial".
    """

    parameter_df: pd.DataFrame
    metric_series: pd.Series
    timestamp_series: pd.Series

    runtime_series: pd.Series = field(init=False)
    # pyre-ignore [16]: Pyre doesn't understand InitVars.
    runtime_fillna: InitVar[bool] = False
    # pyre-ignore [16]: Pyre doesn't understand InitVars.
    log_scale_parameter_names: InitVar[Collection[str] | None] = None
    dtype: torch.dtype = torch.double
    device: torch.device | None = None

    def __post_init__(
        self,
        runtime_fillna: bool,
        log_scale_parameter_names: Collection[str] | None,
    ) -> None:
        self.timestamp_series.name = "timestamp"

        self.runtime_series = self._get_runtime_series(fillna=runtime_fillna)
        self.runtime_series.name = "runtimes"

        parameter_names = get_lcbench_parameter_names()
        if log_scale_parameter_names is None:
            log_scale_parameter_names = get_lcbench_log_scale_parameter_names()

        if len(log_scale_parameter_names) > 0:
            if unrecognized_param_set := (
                set(log_scale_parameter_names) - set(parameter_names)
            ):
                raise ValueError(f"Unrecognized columns: {unrecognized_param_set}")
            self.parameter_df[log_scale_parameter_names] = self.parameter_df[
                log_scale_parameter_names
            ].transform("log")

        self.parameter_df = self.parameter_df[parameter_names]

    @staticmethod
    def _unstack_by_epoch(series: pd.Series) -> pd.DataFrame:
        # unstack by epoch and truncate 52 epochs [0, ..., 51]
        # to 50 epochs [1, ..., 50]
        return series.unstack(level="epoch").iloc[:, 1:-1]

    def _get_runtime_series(self, fillna: bool) -> pd.Series:
        # timestamp (in secs) at every epoch, grouped by trial
        timestamps_grouped = self.timestamp_series.groupby(level="trial")

        # runtime (in secs) of training each incremental epoch
        runtime_series = timestamps_grouped.diff(periods=1)  # first element is NaN
        if fillna:
            runtime_series.fillna(timestamps_grouped.head(n=1), inplace=True)

        return runtime_series

    def _to_tensor(
        self,
        x: pd.DataFrame | pd.Series,
    ) -> torch.Tensor:
        return torch.from_numpy(x.values).to(dtype=self.dtype, device=self.device)

    @property
    def metric_df(self) -> pd.DataFrame:
        return self._unstack_by_epoch(self.metric_series)

    @property
    def runtime_df(self) -> pd.DataFrame:
        return self._unstack_by_epoch(self.runtime_series)

    @property
    def average_runtime_series(self) -> pd.Series:
        # take average runtime over epochs (N6231489 shows runtime is
        # mostly constant across epochs, as one'd expect)
        return self.runtime_series.groupby(level="trial").mean()

    @property
    def parameters(self) -> torch.Tensor:
        return self._to_tensor(self.parameter_df)

    @property
    def metrics(self) -> torch.Tensor:
        return self._to_tensor(self.metric_df)

    @property
    def runtimes(self) -> torch.Tensor:
        return self._to_tensor(self.runtime_df)

    @property
    def average_runtimes(self) -> torch.Tensor:
        return self._to_tensor(self.average_runtime_series)


def load_lcbench_data(
    dataset_name: str,
    metric_name: str = DEFAULT_METRIC_NAME,
    log_scale_parameter_names: Collection[str] | None = None,
    dtype: torch.dtype = torch.double,
    device: torch.device | None = None,
) -> LCBenchData:
    if dataset_name not in DATASET_NAMES:
        raise ValueError(
            f"Invalid dataset {dataset_name}. Valid datasets: {DATASET_NAMES}"
        )

    parameter_df = LCBenchDataLoader(dataset_name, stem="config").load()
    metrics_df = LCBenchDataLoader(dataset_name, stem="metrics").load()

    return LCBenchData(
        parameter_df=parameter_df,
        metric_series=metrics_df[metric_name],
        timestamp_series=metrics_df["time"],
        log_scale_parameter_names=log_scale_parameter_names,
        dtype=dtype,
        device=device,
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterable

import numpy as np
import pandas as pd
from ax.core.experiment import Experiment
from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker


@dataclass(frozen=True)
class BenchmarkResult(Base):
    """The result of a single optimization loop from one
    (BenchmarkProblem, BenchmarkMethod) pair. More information will be added to the
    BenchmarkResult as the suite develops.
    """

    name: str
    experiment: Experiment

    # Tracks best point if single-objective problem, max hypervolume if MOO
    optimization_trace: np.ndarray
    fit_time: float
    gen_time: float

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        if not isinstance(other, BenchmarkResult):
            return False

        return (
            self.name == other.name
            and self.experiment == other.experiment
            and (self.optimization_trace == other.optimization_trace).all()
            and self.fit_time == other.fit_time
            and self.gen_time == other.gen_time
        )


@dataclass(frozen=True)
class AggregatedBenchmarkResult(Base):
    """The result of a benchmark test, or series of replications. Scalar data present
    in the BenchmarkResult is here represented as (mean, sem) pairs. More information
    will be added to the AggregatedBenchmarkResult as the suite develops.
    """

    name: str
    experiments: Iterable[Experiment]

    # mean, sem columns
    optimization_trace: pd.DataFrame

    # (mean, sem) pairs
    fit_time: Tuple[float, float]
    gen_time: Tuple[float, float]

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        if not isinstance(other, AggregatedBenchmarkResult):
            return False

        return (
            self.name == other.name
            and self.experiments == other.experiments
            and self.optimization_trace.eq(other.optimization_trace).all().all()
            and self.fit_time == other.fit_time
            and self.gen_time == other.gen_time
        )

    @classmethod
    def from_benchmark_results(
        cls,
        results: List[BenchmarkResult],
    ) -> AggregatedBenchmarkResult:

        return cls(
            name=results[0].name,
            experiments=[result.experiment for result in results],
            optimization_trace=pd.DataFrame(
                {
                    "mean": [
                        np.mean(
                            [
                                results[j].optimization_trace[i]
                                for j in range(len(results))
                            ]
                        )
                        for i in range(len(results[0].optimization_trace))
                    ],
                    "sem": [
                        cls._series_to_sem(
                            series=[
                                results[j].optimization_trace[i]
                                for j in range(len(results))
                            ]
                        )
                        for i in range(len(results[0].optimization_trace))
                    ],
                }
            ),
            fit_time=cls._series_to_mean_sem(
                series=[result.fit_time for result in results]
            ),
            gen_time=cls._series_to_mean_sem(
                series=[result.gen_time for result in results]
            ),
        )

    @staticmethod
    def _series_to_mean_sem(series: List[float]) -> Tuple[float, float]:
        return (
            np.mean(series),
            AggregatedBenchmarkResult._series_to_sem(series=series),
        )

    @staticmethod
    def _series_to_sem(series: List[float]) -> float:
        return np.std(series, ddof=1) / np.sqrt(len(series))

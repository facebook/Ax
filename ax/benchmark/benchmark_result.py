# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from ax.core.experiment import Experiment
from ax.utils.common.base import Base

# NOTE: Do not add `from __future__ import annotatations` to this file. Adding
# `annotations` postpones evaluation of types and will break FBLearner's usage of
# `BenchmarkResult` as return type annotation, used for serialization and rendering
# in the UI.


@dataclass(frozen=True, eq=False)
class BenchmarkResult(Base):
    """The result of a single optimization loop from one
    (BenchmarkProblem, BenchmarkMethod) pair. More information will be added to the
    BenchmarkResult as the suite develops.
    """

    name: str
    experiment: Experiment

    # Tracks best point if single-objective problem, max hypervolume if MOO
    optimization_trace: np.ndarray
    score_trace: np.ndarray

    fit_time: float
    gen_time: float


@dataclass(frozen=True, eq=False)
class AggregatedBenchmarkResult(Base):
    """The result of a benchmark test, or series of replications. Scalar data present
    in the BenchmarkResult is here represented as (mean, sem) pairs. More information
    will be added to the AggregatedBenchmarkResult as the suite develops.
    """

    name: str
    results: List[BenchmarkResult]

    # mean, sem, and quartile columns
    optimization_trace: pd.DataFrame
    score_trace: pd.DataFrame

    # (mean, sem) pairs
    fit_time: Tuple[float, float]
    gen_time: Tuple[float, float]

    def __str__(self) -> str:
        return f"{self.__class__}(name={self.name})"

    @classmethod
    def from_benchmark_results(
        cls,
        results: List[BenchmarkResult],
    ) -> "AggregatedBenchmarkResult":
        optimization_traces = pd.DataFrame([res.optimization_trace for res in results])
        score_traces = pd.DataFrame([res.score_trace for res in results])
        fit_times = pd.Series([result.fit_time for result in results])
        gen_times = pd.Series([result.gen_time for result in results])

        return cls(
            name=results[0].name,
            results=results,
            optimization_trace=pd.DataFrame(
                {
                    "mean": optimization_traces.mean(),
                    "sem": optimization_traces.sem(),
                    "P10": optimization_traces.quantile(q=0.10),
                    "P25": optimization_traces.quantile(q=0.25),
                    "P50": optimization_traces.quantile(q=0.5),
                    "P75": optimization_traces.quantile(q=0.75),
                    "P90": optimization_traces.quantile(q=0.90),
                }
            ),
            score_trace=pd.DataFrame(
                {
                    "mean": score_traces.mean(),
                    "sem": score_traces.sem(),
                    "P10": score_traces.quantile(q=0.10),
                    "P25": score_traces.quantile(q=0.25),
                    "P50": score_traces.quantile(q=0.5),
                    "P75": score_traces.quantile(q=0.75),
                    "P90": score_traces.quantile(q=0.90),
                }
            ),
            fit_time=(fit_times.mean().item(), fit_times.sem().item()),
            gen_time=(gen_times.mean().item(), gen_times.sem().item()),
        )

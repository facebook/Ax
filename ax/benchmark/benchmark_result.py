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
from ax.utils.common.equality import equality_typechecker

# NOTE: Do not add `from __future__ import annotatations` to this file. Adding
# `annotations` postpones evaluation of types and will break FBLearner's usage of
# `BenchmarkResult` as return type annotation, used for serialization and rendering
# in the UI.


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
    experiments: List[Experiment]

    # mean, sem columns
    optimization_trace: pd.DataFrame

    # (mean, sem) pairs
    fit_time: Tuple[float, float]
    gen_time: Tuple[float, float]

    def __str__(self) -> str:
        return f"AggregatedBenchmarkResult(name={self.name})"

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
    ) -> "AggregatedBenchmarkResult":
        optimization_traces = pd.DataFrame([res.optimization_trace for res in results])
        fit_times = pd.Series([result.fit_time for result in results])
        gen_times = pd.Series([result.gen_time for result in results])

        return cls(
            name=results[0].name,
            experiments=[result.experiment for result in results],
            optimization_trace=pd.DataFrame(
                {
                    "mean": optimization_traces.mean(),
                    "median": optimization_traces.median(),
                    "sem": optimization_traces.sem(),
                }
            ),
            fit_time=(fit_times.mean().item(), fit_times.sem().item()),
            gen_time=(gen_times.mean().item(), gen_times.sem().item()),
        )


@dataclass(frozen=True)
class ScoredBenchmarkResult(AggregatedBenchmarkResult):
    """An AggregatedBenchmarkResult normalized against some baseline method (for the
    same problem), typically Sobol. The score is calculated in such a way that 0
    corresponds to performance equivalent with the baseline and 100 indicates the true
    optimum was found.
    """

    baseline_result: AggregatedBenchmarkResult
    score: np.ndarray

    def __str__(self) -> str:
        return "ScoredBenchmarkResult("
        f"name={self.name}, baseline_result{self.baseline_result}"
        ")"

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        if not isinstance(other, ScoredBenchmarkResult):
            return False

        return (
            super().__eq__(other)
            and self.baseline_result == other.baseline_result
            and (self.score == other.score).all()
        )

    @classmethod
    def from_result_and_baseline(
        cls,
        aggregated_result: AggregatedBenchmarkResult,
        baseline_result: AggregatedBenchmarkResult,
        optimum: float,
    ) -> "ScoredBenchmarkResult":
        baseline = baseline_result.optimization_trace["mean"][
            : len(aggregated_result.optimization_trace["mean"])
        ]

        score = (
            100
            * (
                1
                - (aggregated_result.optimization_trace["mean"] - optimum)
                / (baseline - optimum)
            )
        ).to_numpy()

        return cls(
            name=aggregated_result.name,
            experiments=aggregated_result.experiments,
            optimization_trace=aggregated_result.optimization_trace,
            fit_time=aggregated_result.fit_time,
            gen_time=aggregated_result.gen_time,
            baseline_result=baseline_result,
            score=score,
        )

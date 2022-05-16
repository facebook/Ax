# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import cast, List, Tuple

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
    fit_time: float
    gen_time: float


@dataclass(frozen=True, eq=False)
class ScoredBenchmarkResult(BenchmarkResult):
    """A BenchmarkResult normalized against some baseline method (for the same
    problem), typically Sobol. The score is calculated in such a way that 0 corresponds
    to performance equivalent with the baseline and 100 indicates the true optimum was
    found.
    """

    baseline_result: BenchmarkResult
    score_trace: np.ndarray

    @classmethod
    def from_result_and_baseline(
        cls,
        result: BenchmarkResult,
        baseline_result: BenchmarkResult,
        optimum: float,
    ) -> "ScoredBenchmarkResult":
        """Combine a result and baseline into a ScoredBenchmarkResult. This entails
        computing a score trace from the baseline result and the problem's known (or
        estimated to best ability) optimum.
        """

        baseline = baseline_result.optimization_trace[: len(result.optimization_trace)]

        score_trace = 100 * (
            1 - (result.optimization_trace - optimum) / (baseline - optimum)
        )

        return cls(
            name=result.name,
            experiment=result.experiment,
            optimization_trace=result.optimization_trace,
            fit_time=result.fit_time,
            gen_time=result.gen_time,
            baseline_result=baseline_result,
            score_trace=score_trace,
        )


@dataclass(frozen=True, eq=False)
class AggregatedBenchmarkResult(Base):
    """The result of a benchmark test, or series of replications. Scalar data present
    in the BenchmarkResult is here represented as (mean, sem) pairs. More information
    will be added to the AggregatedBenchmarkResult as the suite develops.
    """

    name: str
    results: List[BenchmarkResult]

    # median, mean, sem columns
    optimization_trace: pd.DataFrame

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
        fit_times = pd.Series([result.fit_time for result in results])
        gen_times = pd.Series([result.gen_time for result in results])

        return cls(
            name=results[0].name,
            results=results,
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


@dataclass(frozen=True, eq=False)
class AggregatedScoredBenchmarkResult(AggregatedBenchmarkResult):
    """The result of a scored benchmark test, or series of scored replications. Scalar
    data present in the BenchmarkResult is here represented as (mean, sem) pairs, or
    as (median, mean, sem) traces.
    """

    # median, mean, sem columns
    score_trace: pd.DataFrame

    @classmethod
    def from_scored_results(
        cls,
        scored_results: List[ScoredBenchmarkResult],
    ) -> "AggregatedScoredBenchmarkResult":
        aggregated_result = AggregatedBenchmarkResult.from_benchmark_results(
            results=[
                cast(BenchmarkResult, result) for result in scored_results
            ]  # downcast from ScoredResult to BenchmarkResult
        )

        score_traces = pd.DataFrame([res.score_trace for res in scored_results])

        return cls(
            score_trace=pd.DataFrame(
                {
                    "mean": score_traces.mean(),
                    "median": score_traces.median(),
                    "sem": score_traces.sem(),
                }
            ),
            **aggregated_result.__dict__,
        )

    @classmethod
    def from_aggregated_result_and_aggregated_baseline_result(
        cls,
        aggregated_result: AggregatedBenchmarkResult,
        aggregated_baseline_result: AggregatedBenchmarkResult,
        optimum: float,
    ) -> "AggregatedScoredBenchmarkResult":
        return cls.from_scored_results(
            scored_results=[
                ScoredBenchmarkResult.from_result_and_baseline(
                    result=result, baseline_result=baseline_result, optimum=optimum
                )
                for result, baseline_result in zip(
                    aggregated_result.results, aggregated_baseline_result.results
                )
            ]
        )

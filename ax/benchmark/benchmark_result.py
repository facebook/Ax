# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List

from ax.core.experiment import Experiment
from ax.utils.common.base import Base
from numpy import nanmean, nanquantile, ndarray
from pandas import DataFrame
from scipy.stats import sem

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
    seed: int
    experiment: Experiment

    # Tracks best point if single-objective problem, max hypervolume if MOO
    optimization_trace: ndarray
    score_trace: ndarray

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
    optimization_trace: DataFrame
    score_trace: DataFrame

    # (mean, sem) pairs
    fit_time: List[float]
    gen_time: List[float]

    def __str__(self) -> str:
        return f"{self.__class__}(name={self.name})"

    @classmethod
    def from_benchmark_results(
        cls,
        results: List[BenchmarkResult],
    ) -> "AggregatedBenchmarkResult":
        """Aggregrates a list of BenchmarkResults. For various reasons (timeout, errors,
        etc.) each BenchmarkResult may have a different number of trials; aggregated
        traces and statistics are computed with and truncated to the minimum trial count
        to ensure each replication is included.
        """
        # Extract average wall times and standard errors thereof
        fit_time, gen_time = map(
            lambda Ts: [nanmean(Ts), float(sem(Ts, ddof=1, nan_policy="omit"))],
            zip(*((res.fit_time, res.gen_time) for res in results)),
        )

        # Compute some statistics for each trace
        trace_stats = {}
        percentiles = 0.1, 0.25, 0.5, 0.75, 0.9
        for name in ("optimization_trace", "score_trace"):
            stats = trace_stats[name] = {"mean": [], "sem": []}
            quantiles = []
            for step_vals in zip(
                *(getattr(res, name) for res in results),
            ):
                stats["mean"].append(nanmean(step_vals))
                stats["sem"].append(sem(step_vals, ddof=1, nan_policy="omit"))
                quantiles.append(nanquantile(step_vals, q=percentiles))

            stats.update(
                {f"P{100 * p:.0f}": q for p, q in zip(percentiles, zip(*quantiles))}
            )

        # Return aggregated results
        return cls(
            name=results[0].name,
            results=results,
            fit_time=fit_time,
            gen_time=gen_time,
            **{name: DataFrame(stats) for name, stats in trace_stats.items()},
        )

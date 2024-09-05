# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# NOTE: Do not add `from __future__ import annotatations` to this file. Adding
# `annotations` postpones evaluation of types and will break FBLearner's usage of
# `BenchmarkResult` as return type annotation, used for serialization and rendering
# in the UI.

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import numpy as np
from ax.core.experiment import Experiment
from ax.utils.common.base import Base
from numpy import nanmean, nanquantile, ndarray
from pandas import DataFrame
from scipy.stats import sem

PERCENTILES = [0.25, 0.5, 0.75]


@dataclass(eq=False)
class BenchmarkResult(Base):
    """The result of a single optimization loop from one
    (BenchmarkProblem, BenchmarkMethod) pair.

    Args:
        name: Name of the benchmark. Should make it possible to determine the
            problem and the method.
        seed: Seed used for determinism.
        optimization_trace: For single-objective problems, element i of the
            optimization trace is the oracle value of the "best" point, computed
            after the first i trials have been run. For multi-objective
            problems, element i of the optimization trace is the hypervolume of
            oracle values at a set of points, also computed after the first i
            trials (even if these were ``BatchTrials``).  Oracle values are
            typically ground-truth (rather than noisy) and evaluated at the
            target task and fidelity.

        score_trace: The scores associated with the problem, typically either
            the optimization_trace or inference_value_trace normalized to a
            0-100 scale for comparability between problems.
        fit_time: Total time spent fitting models.
        gen_time: Total time spent generating candidates.
        experiment: If not ``None``, the Ax experiment associated with the
            optimization that generated this data. Either ``experiment`` or
            ``experiment_storage_id`` must be provided.
        experiment_storage_id: Pointer to location where experiment data can be read.
    """

    name: str
    seed: int

    optimization_trace: ndarray
    score_trace: ndarray

    fit_time: float
    gen_time: float

    experiment: Optional[Experiment] = None
    experiment_storage_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.experiment is not None and self.experiment_storage_id is not None:
            raise ValueError(
                "Cannot specify both an `experiment` and an "
                "`experiment_storage_id` for the experiment."
            )
        if self.experiment is None and self.experiment_storage_id is None:
            raise ValueError(
                "Must provide an `experiment` or `experiment_storage_id` "
                "to construct a BenchmarkResult."
            )


@dataclass(frozen=True, eq=False)
class AggregatedBenchmarkResult(Base):
    """The result of a benchmark test, or series of replications. Scalar data present
    in the BenchmarkResult is here represented as (mean, sem) pairs.
    """

    name: str
    results: list[BenchmarkResult]

    # mean, sem, and quartile columns
    optimization_trace: DataFrame
    score_trace: DataFrame

    # (mean, sem) pairs
    fit_time: list[float]
    gen_time: list[float]

    def __str__(self) -> str:
        return f"{self.__class__}(name={self.name})"

    @classmethod
    def from_benchmark_results(
        cls,
        results: list[BenchmarkResult],
    ) -> "AggregatedBenchmarkResult":
        """Aggregrates a list of BenchmarkResults. For various reasons (timeout, errors,
        etc.) each BenchmarkResult may have a different number of trials; aggregated
        traces and statistics are computed with and truncated to the minimum trial count
        to ensure each replication is included.
        """
        # Extract average wall times and standard errors thereof
        fit_time, gen_time = (
            [nanmean(Ts), float(sem(Ts, ddof=1, nan_policy="propagate"))]
            for Ts in zip(*((res.fit_time, res.gen_time) for res in results))
        )

        # Compute some statistics for each trace
        trace_stats = {}
        for name in ("optimization_trace", "score_trace"):
            step_data = zip(*(getattr(res, name) for res in results))
            stats = _get_stats(step_data=step_data, percentiles=PERCENTILES)
            trace_stats[name] = stats

        # Return aggregated results
        return cls(
            name=results[0].name,
            results=results,
            fit_time=fit_time,
            gen_time=gen_time,
            **{name: DataFrame(stats) for name, stats in trace_stats.items()},
        )


def _get_stats(
    step_data: Iterable[np.ndarray],
    percentiles: list[float],
) -> dict[str, list[float]]:
    quantiles = []
    stats = {"mean": [], "sem": []}
    for step_vals in step_data:
        stats["mean"].append(nanmean(step_vals))
        stats["sem"].append(sem(step_vals, ddof=1, nan_policy="propagate"))
        quantiles.append(nanquantile(step_vals, q=percentiles))
    stats.update({f"P{100 * p:.0f}": q for p, q in zip(percentiles, zip(*quantiles))})
    return stats

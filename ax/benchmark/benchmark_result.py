# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.service.utils.best_point_mixin import BestPointMixin
from ax.utils.common.base import Base
from ax.utils.common.typeutils import not_none
from numpy import nanmean, nanquantile, ndarray
from pandas import DataFrame
from scipy.stats import sem

# NOTE: Do not add `from __future__ import annotatations` to this file. Adding
# `annotations` postpones evaluation of types and will break FBLearner's usage of
# `BenchmarkResult` as return type annotation, used for serialization and rendering
# in the UI.

PERCENTILES = 0.1, 0.25, 0.5, 0.75, 0.9


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

    def optimization_trace_by_progression(
        self, final_progression_only: bool = False
    ) -> Tuple[ndarray, ndarray]:  # (y-values, x-values)
        if isinstance(self.experiment.lookup_data(), MapData):
            by_progression_result = BestPointMixin._get_trace_by_progression(
                experiment=self.experiment,
                final_progression_only=final_progression_only,
            )
            # tuple of y-values, x-values
            optimization_trace_by_progression = (
                np.array(by_progression_result[0]),
                np.array(by_progression_result[1]),
            )
        else:
            # if not MapData, set this to standard optimization_trace
            # with a default x-values
            optimization_trace = np.array(
                BestPointMixin._get_trace(
                    experiment=self.experiment,
                )
            )
            optimization_trace_by_progression = (
                optimization_trace,
                np.arange(optimization_trace.shape[0]),
            )
        return optimization_trace_by_progression

    def progression_trace(self) -> ndarray:
        """Computes progressions used as a function of trials and
        also the total progression across all trials."""
        experiment = self.experiment
        optimization_config = not_none(experiment.optimization_config)
        objective = optimization_config.objective.metric.name
        map_data = experiment.lookup_data()
        if not isinstance(map_data, MapData):
            raise ValueError("`get_trace_by_progression` requires MapData.")
        map_df = map_data.map_df

        # assume the first map_key is progression
        map_key = map_data.map_keys[0]

        map_df = map_df[map_df["metric_name"] == objective]
        map_df = map_df.sort_values(by=["trial_index", map_key])
        map_df = map_df.drop_duplicates(MapData.DEDUPLICATE_BY_COLUMNS, keep="last")
        return map_df[map_key].to_numpy()

    def total_progression(self) -> float:
        return self.progression_trace().sum()


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
            lambda Ts: [nanmean(Ts), float(sem(Ts, ddof=1, nan_policy="propagate"))],
            zip(*((res.fit_time, res.gen_time) for res in results)),
        )

        # Compute some statistics for each trace
        trace_stats = {}
        for name in ("optimization_trace", "score_trace"):
            step_data = zip(
                *(getattr(res, name) for res in results),
            )
            stats = _get_stats(
                step_data=step_data, percentiles=PERCENTILES, progressions=None
            )
            trace_stats[name] = stats

        # Return aggregated results
        return cls(
            name=results[0].name,
            results=results,
            fit_time=fit_time,
            gen_time=gen_time,
            **{name: DataFrame(stats) for name, stats in trace_stats.items()},
        )

    def optimization_trace_by_progression(
        self, final_progression_only: bool = False
    ) -> DataFrame:
        trace_by_progression_results = [
            res.optimization_trace_by_progression(
                final_progression_only=final_progression_only
            )
            for res in self.results
        ]
        step_data = zip(
            *(res[0] for res in trace_by_progression_results),
        )
        progressions = trace_by_progression_results[0][1]
        stats = _get_stats(
            step_data=step_data,
            percentiles=PERCENTILES,
            progressions=progressions,
        )
        return DataFrame(stats)

    def progression_trace(self) -> DataFrame:
        progression_traces = zip(*(res.progression_trace() for res in self.results))
        stats = _get_stats(
            step_data=progression_traces, percentiles=PERCENTILES, progressions=None
        )
        return DataFrame(stats)

    def total_progression(self) -> List[float]:
        total_progressions = [res.total_progression() for res in self.results]
        return [
            nanmean(total_progressions),
            sem(total_progressions, ddof=1, nan_policy="propagate"),
        ]


def _get_stats(
    # pyre-fixme[24]: Generic type `Iterable` expects 1 type parameter.
    step_data: Iterable,
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    percentiles: Tuple,
    progressions: Optional[np.ndarray],
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use `typing.Dict`
    #  to avoid runtime subscripting errors.
) -> Dict:
    quantiles = []
    stats = {"mean": [], "sem": []}
    if progressions is not None:
        stats.update({"progression": []})
    for i, step_vals in enumerate(step_data):
        stats["mean"].append(nanmean(step_vals))
        stats["sem"].append(sem(step_vals, ddof=1, nan_policy="propagate"))
        quantiles.append(nanquantile(step_vals, q=percentiles))
        if progressions is not None:
            stats["progression"].append(progressions[i])
    stats.update({f"P{100 * p:.0f}": q for p, q in zip(percentiles, zip(*quantiles))})
    return stats

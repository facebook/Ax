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
from ax.core.utils import get_model_times
from ax.service.scheduler import Scheduler
from ax.utils.common.typeutils import not_none


@dataclass(frozen=True)
class BenchmarkResult:
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

    @classmethod
    def from_scheduler(cls, scheduler: Scheduler) -> BenchmarkResult:
        fit_time, gen_time = get_model_times(experiment=scheduler.experiment)

        return cls(
            name=scheduler.experiment.name,
            experiment=scheduler.experiment,
            optimization_trace=cls._get_trace(scheduler=scheduler),
            fit_time=fit_time,
            gen_time=gen_time,
        )

    @staticmethod
    def _get_trace(scheduler: Scheduler) -> np.ndarray:
        if scheduler.experiment.is_moo_problem:
            return np.array(
                [
                    scheduler.get_hypervolume(
                        trial_indices=[*range(i + 1)], use_model_predictions=False
                    )
                    if i != 0
                    else 0
                    # TODO[mpolson64] on i=0 we get an error with SearchspaceToChoice
                    for i in range(len(scheduler.experiment.trials))
                ],
            )

        best_trials = [
            scheduler.get_best_trial(
                trial_indices=[*range(i + 1)], use_model_predictions=False
            )
            for i in range(len(scheduler.experiment.trials))
        ]

        return np.array(
            [
                not_none(not_none(trial)[2])[0][
                    not_none(
                        scheduler.experiment.optimization_config
                    ).objective.metric.name
                ]
                for trial in best_trials
                if trial is not None and not_none(trial)[2] is not None
            ]
        )


@dataclass(frozen=True)
class AggregatedBenchmarkResult:
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

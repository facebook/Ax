# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Module for benchmarking Ax algorithms.

Key terms used:

* Replication: 1 run of an optimization loop; (BenchmarkProblem, BenchmarkMethod) pair.
* Test: multiple replications, ran for statistical significance.
* Full run: multiple tests on many (BenchmarkProblem, BenchmarkMethod) pairs.
* Method: (one of) the algorithm(s) being benchmarked.
* Problem: a synthetic function, a surrogate surface, or an ML model, on which
  to assess the performance of algorithms.

"""
from time import time
from typing import Tuple, List, Iterable

import numpy as np
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import (
    MultiObjectiveBenchmarkProblem,
    SingleObjectiveBenchmarkProblem,
    BenchmarkProblem,
)
from ax.benchmark.benchmark_result import (
    ScoredBenchmarkResult,
    BenchmarkResult,
    AggregatedBenchmarkResult,
)
from ax.core.experiment import Experiment
from ax.core.utils import get_model_times
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.scheduler import SchedulerOptions, Scheduler
from ax.utils.common.typeutils import not_none


def benchmark_replication(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
) -> BenchmarkResult:
    """Runs one benchmarking replication (equivalent to one optimization loop).

    Args:
        problem: The BenchmarkProblem to test against (can be synthetic or real)
        method: The BenchmarkMethod to test
    """

    experiment = Experiment(
        name=f"{problem.name}|{method.name}_{int(time())}",
        search_space=problem.search_space,
        optimization_config=problem.optimization_config,
        runner=problem.runner,
    )

    scheduler = Scheduler(
        experiment=experiment,
        generation_strategy=method.generation_strategy.clone_reset(),
        options=method.scheduler_options,
    )

    scheduler.run_all_trials()

    return _result_from_scheduler(scheduler=scheduler)


def benchmark_test(
    problem: BenchmarkProblem, method: BenchmarkMethod, num_replications: int = 10
) -> AggregatedBenchmarkResult:

    return AggregatedBenchmarkResult.from_benchmark_results(
        results=[
            benchmark_replication(problem=problem, method=method)
            for _ in range(num_replications)
        ]
    )


def benchmark_full_run(
    problems: Iterable[BenchmarkProblem],
    methods: Iterable[BenchmarkMethod],
    num_replications: int = 10,
) -> List[AggregatedBenchmarkResult]:

    return [
        benchmark_test(
            problem=problem, method=method, num_replications=num_replications
        )
        for problem in problems
        for method in methods
    ]


def benchmark_scored_test(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    baseline_result: AggregatedBenchmarkResult,
    num_replications: int = 10,
) -> ScoredBenchmarkResult:
    if isinstance(problem, SingleObjectiveBenchmarkProblem):
        optimum = problem.optimal_value
    elif isinstance(problem, MultiObjectiveBenchmarkProblem):
        optimum = problem.maximum_hypervolume
    else:
        raise UnsupportedError(
            "Scored benchmarking is not supported for BenchmarkProblems with no known "
            "optimum."
        )

    aggregated_result = benchmark_test(
        problem=problem, method=method, num_replications=num_replications
    )

    return ScoredBenchmarkResult.from_result_and_baseline(
        aggregated_result=aggregated_result,
        baseline_result=baseline_result,
        optimum=optimum,
    )


def benchmark_scored_full_run(
    problems_baseline_results: Iterable[
        Tuple[BenchmarkProblem, AggregatedBenchmarkResult]
    ],
    methods: Iterable[BenchmarkMethod],
    num_replications: int = 10,
) -> List[ScoredBenchmarkResult]:

    return [
        benchmark_scored_test(
            problem=problem,
            method=method,
            baseline_result=baseline_result,
            num_replications=num_replications,
        )
        for problem, baseline_result in problems_baseline_results
        for method in methods
    ]


def get_sobol_baseline(
    problem: BenchmarkProblem, num_replications: int = 100, total_trials: int = 100
) -> AggregatedBenchmarkResult:
    return benchmark_test(
        problem=problem,
        method=BenchmarkMethod(
            name="SOBOL_BASELINE",
            generation_strategy=GenerationStrategy(
                steps=[GenerationStep(model=Models.SOBOL, num_trials=-1)],
                name="SOBOL",
            ),
            scheduler_options=SchedulerOptions(
                total_trials=total_trials, init_seconds_between_polls=0
            ),
        ),
        num_replications=num_replications,
    )


def _result_from_scheduler(scheduler: Scheduler) -> BenchmarkResult:
    fit_time, gen_time = get_model_times(experiment=scheduler.experiment)

    return BenchmarkResult(
        name=scheduler.experiment.name,
        experiment=scheduler.experiment,
        optimization_trace=_get_trace(scheduler=scheduler),
        fit_time=fit_time,
        gen_time=gen_time,
    )


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
                not_none(scheduler.experiment.optimization_config).objective.metric.name
            ]
            for trial in best_trials
            if trial is not None and not_none(trial)[2] is not None
        ]
    )

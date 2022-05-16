# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, List, Optional, Tuple

from ax.benchmark.benchmark import benchmark_replication, benchmark_test
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import (
    BenchmarkProblem,
    MultiObjectiveBenchmarkProblem,
    SingleObjectiveBenchmarkProblem,
)
from ax.benchmark.benchmark_result import (
    AggregatedBenchmarkResult,
    AggregatedScoredBenchmarkResult,
    BenchmarkResult,
    ScoredBenchmarkResult,
)
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.scheduler import SchedulerOptions
from botorch.exceptions.errors import UnsupportedError


def scored_benchmark_replication(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    replication_seed: Optional[int] = None,
    baseline_result: Optional[BenchmarkResult] = None,
) -> ScoredBenchmarkResult:
    """Runs one scored benchmarking replication (equivalent to one optimization
    loop). The score is calculated such that 0 represents equivalent performance to
    the baseline and 100 indicates the known optimum was found.

    Args:
        problem: The BenchmarkProblem to test against (can be synthetic or real)
        method: The BenchmarkMethod to test
        replication_seed: The seed to use for this replication, set using `manual_seed`
            from `botorch.utils.sampling`.
        baseline_result: The result of some baseline method on the same problem. If
            no baseline_result is provided a Sobol replication will be run on the
            problem and used as the baseline for comparison.
    """

    if isinstance(problem, SingleObjectiveBenchmarkProblem):
        optimum = problem.optimal_value
    elif isinstance(problem, MultiObjectiveBenchmarkProblem):
        optimum = problem.maximum_hypervolume
    else:
        raise UnsupportedError(
            "Scored benchmarking is not supported for BenchmarkProblems with no known "
            "optimum."
        )
    if baseline_result is None:
        baseline_result = benchmark_replication(
            problem=problem,
            method=BenchmarkMethod(
                name="SOBOL_BASELINE",
                generation_strategy=GenerationStrategy(
                    steps=[GenerationStep(model=Models.SOBOL, num_trials=-1)],
                    name="SOBOL",
                ),
                scheduler_options=SchedulerOptions(
                    total_trials=method.scheduler_options.total_trials,
                    init_seconds_between_polls=0,
                ),
            ),
            replication_seed=replication_seed,
        )

    result = benchmark_replication(
        problem=problem, method=method, replication_seed=replication_seed
    )

    return ScoredBenchmarkResult.from_result_and_baseline(
        result=result, baseline_result=baseline_result, optimum=optimum
    )


def scored_benchmark_test(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    num_replications: int = 10,
    seed: Optional[int] = None,
    aggregated_baseline_result: Optional[AggregatedBenchmarkResult] = None,
) -> AggregatedScoredBenchmarkResult:
    """Runs n scored benchmarking replications (equivalent to n optimization loops).
    The score is calculated such that 0 represents equivalent performance to
    the baseline and 100 indicates the known optimum was found.

    Args:
        problem: The BenchmarkProblem to test against (can be synthetic or real)
        method: The BenchmarkMethod to test
        seed: The seed to use for the first replication (and incremented by 1 for
            subsequent replications), set using `manual_seed` from
            `botorch.utils.sampling`.
        aggregated_baseline_result: The result of some baseline method on the same
            problem. If no baseline_result is provided a Sobol replication will be
            run on the problem and used as the baseline for comparison.
    """
    if isinstance(problem, SingleObjectiveBenchmarkProblem):
        optimum = problem.optimal_value
    elif isinstance(problem, MultiObjectiveBenchmarkProblem):
        optimum = problem.maximum_hypervolume
    else:
        raise UnsupportedError(
            "Scored benchmarking is not supported for BenchmarkProblems with no known "
            "optimum."
        )

    if aggregated_baseline_result is None:
        aggregated_baseline_result = benchmark_test(
            problem=problem,
            method=BenchmarkMethod(
                name="SOBOL_BASELINE",
                generation_strategy=GenerationStrategy(
                    steps=[GenerationStep(model=Models.SOBOL, num_trials=-1)],
                    name="SOBOL",
                ),
                scheduler_options=SchedulerOptions(
                    total_trials=method.scheduler_options.total_trials,
                    init_seconds_between_polls=0,
                ),
            ),
            num_replications=num_replications,
            seed=seed,
        )

    aggregated_result = benchmark_test(
        problem=problem, method=method, num_replications=num_replications, seed=seed
    )

    return AggregatedScoredBenchmarkResult.from_aggregated_result_and_aggregated_baseline_result(  # noqa
        aggregated_result=aggregated_result,
        aggregated_baseline_result=aggregated_baseline_result,
        optimum=optimum,
    )


def scored_benchmark_full_run(
    problems_baseline_results: Iterable[
        Tuple[BenchmarkProblem, AggregatedBenchmarkResult]
    ],
    methods: Iterable[BenchmarkMethod],
    num_replications: int = 10,
    seed: Optional[int] = None,
) -> List[AggregatedScoredBenchmarkResult]:

    return [
        scored_benchmark_test(
            problem=problem,
            method=method,
            num_replications=num_replications,
            seed=seed,
            aggregated_baseline_result=aggregated_baseline_result,
        )
        for problem, aggregated_baseline_result in problems_baseline_results
        for method in methods
    ]

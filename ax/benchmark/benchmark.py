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
from typing import Iterable, List, Optional

import numpy as np
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult, BenchmarkResult
from ax.core.experiment import Experiment
from ax.core.utils import get_model_times
from ax.service.scheduler import Scheduler
from ax.service.utils.best_point_mixin import BestPointMixin
from botorch.utils.sampling import manual_seed


def benchmark_replication(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    replication_seed: Optional[int] = None,
) -> BenchmarkResult:
    """Runs one benchmarking replication (equivalent to one optimization loop).

    Args:
        problem: The BenchmarkProblem to test against (can be synthetic or real)
        method: The BenchmarkMethod to test
        replication_seed: The seed to use for this replication, set using `manual_seed`
            from `botorch.utils.sampling`.
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
    with manual_seed(seed=replication_seed):
        scheduler.run_all_trials()

    return _result_from_scheduler(scheduler=scheduler)


def benchmark_test(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    num_replications: int = 10,
    seed: Optional[int] = None,
) -> AggregatedBenchmarkResult:

    return AggregatedBenchmarkResult.from_benchmark_results(
        results=[
            benchmark_replication(
                problem=problem,
                method=method,
                replication_seed=seed + i if seed is not None else None,
            )
            for i in range(num_replications)
        ]
    )


def benchmark_full_run(
    problems: Iterable[BenchmarkProblem],
    methods: Iterable[BenchmarkMethod],
    num_replications: int = 10,
    seed: Optional[int] = None,
) -> List[AggregatedBenchmarkResult]:

    return [
        benchmark_test(
            problem=problem, method=method, num_replications=num_replications, seed=seed
        )
        for problem in problems
        for method in methods
    ]


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
    return np.array(BestPointMixin.get_trace(experiment=scheduler.experiment))

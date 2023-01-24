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

from functools import partial
from itertools import product
from time import time
from typing import Any, Iterable, List

import numpy as np

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import (
    BenchmarkProblem,
    MultiObjectiveBenchmarkProblem,
    SingleObjectiveBenchmarkProblem,
)
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult, BenchmarkResult
from ax.core.experiment import Experiment
from ax.core.utils import get_model_times
from ax.service.scheduler import Scheduler
from botorch.utils.sampling import manual_seed


def benchmark_replication(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    seed: int,
) -> BenchmarkResult:
    """Runs one benchmarking replication (equivalent to one optimization loop).

    Args:
        problem: The BenchmarkProblem to test against (can be synthetic or real)
        method: The BenchmarkMethod to test
        seed: The seed to use for this replication, set using `manual_seed`
            from `botorch.utils.sampling`.
    """

    experiment = Experiment(
        name=f"{problem.name}|{method.name}_{int(time())}",
        search_space=problem.search_space,
        optimization_config=problem.optimization_config,
        tracking_metrics=problem.tracking_metrics,
        runner=problem.runner,
    )

    scheduler = Scheduler(
        experiment=experiment,
        generation_strategy=method.generation_strategy.clone_reset(),
        options=method.scheduler_options,
    )

    with manual_seed(seed=seed):
        scheduler.run_n_trials(max_trials=problem.num_trials)

    optimization_trace = np.array(scheduler.get_trace())

    # Use the first GenerationStep's best found point as baseline. Sometimes (ex. in
    # a timeout) the first GenerationStep will not have not completed and we will not
    # have enough trials; in this case we do not score.
    num_baseline_trials = scheduler.generation_strategy._steps[0].num_trials
    baseline = (
        optimization_trace[num_baseline_trials - 1]
        if len(optimization_trace) > num_baseline_trials
        else None
    )

    if isinstance(problem, SingleObjectiveBenchmarkProblem):
        optimum = problem.optimal_value
    elif isinstance(problem, MultiObjectiveBenchmarkProblem):
        optimum = problem.maximum_hypervolume
    else:
        # If no known optimum exists scoring cannot take place in a meaningful way
        optimum = None

    score_trace = (
        (100 * (1 - (optimization_trace - optimum) / (baseline - optimum))).clip(min=0)
        if optimum is not None and baseline is not None
        else np.full(len(optimization_trace), np.nan)
    )

    fit_time, gen_time = get_model_times(experiment=scheduler.experiment)

    return BenchmarkResult(
        name=scheduler.experiment.name,
        seed=seed,
        experiment=scheduler.experiment,
        optimization_trace=optimization_trace,
        score_trace=score_trace,
        fit_time=fit_time,
        gen_time=gen_time,
    )


def benchmark_test(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    seeds: Iterable[int],
    **kwargs: Any,
) -> AggregatedBenchmarkResult:
    base_case = partial(benchmark_replication, problem=problem, method=method, **kwargs)
    return AggregatedBenchmarkResult.from_benchmark_results(
        results=[base_case(seed=seed) for seed in seeds]
    )


def benchmark_full_run(
    problems: Iterable[BenchmarkProblem],
    methods: Iterable[BenchmarkMethod],
    seeds: Iterable[int],
    **kwargs: Any,
) -> List[AggregatedBenchmarkResult]:
    test_env = partial(benchmark_test, seeds=seeds, **kwargs)
    return [test_env(problem=p, method=m) for p, m in product(problems, methods)]

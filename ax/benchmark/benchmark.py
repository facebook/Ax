# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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

from collections.abc import Iterable
from itertools import product
from logging import Logger
from time import time

import numpy as np

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult, BenchmarkResult
from ax.core.experiment import Experiment
from ax.core.utils import get_model_times
from ax.service.scheduler import Scheduler
from ax.service.utils.best_point_mixin import BestPointMixin
from ax.utils.common.logger import get_logger
from ax.utils.common.random import with_rng_seed

logger: Logger = get_logger(__name__)


def compute_score_trace(
    optimization_trace: np.ndarray,
    num_baseline_trials: int,
    problem: BenchmarkProblem,
) -> np.ndarray:
    """Computes a score trace from the optimization trace."""

    # Use the first GenerationStep's best found point as baseline. Sometimes (ex. in
    # a timeout) the first GenerationStep will not have not completed and we will not
    # have enough trials; in this case we do not score.
    if len(optimization_trace) <= num_baseline_trials:
        return np.full(len(optimization_trace), np.nan)
    optimum = problem.optimal_value
    baseline = optimization_trace[num_baseline_trials - 1]

    score_trace = 100 * (1 - (optimization_trace - optimum) / (baseline - optimum))
    if score_trace.max() > 100:
        logger.info(
            "Observed scores above 100. This indicates that we found a trial that is "
            "better than the optimum. Clipping scores to 100 for now."
        )
    return score_trace.clip(min=0, max=100)


def _create_benchmark_experiment(
    problem: BenchmarkProblem, method_name: str
) -> Experiment:
    """Creates an empty experiment for the given problem and method.

    If the problem is "noiseless" (i.e. evaluations prior to adding artificial
    noise to the outcomes are determinstic), also adds the respective ground
    truth metrics as tracking metrics for each metric defined on the problem
    (including tracking metrics and the metrics in the OptimizationConfig).

    Args:
        problem: The BenchmarkProblem to test against (can be synthetic or real).
        method_name: Name of the method being tested.

    Returns:
        The Experiment object to be used for benchmarking.
    """
    return Experiment(
        name=f"{problem.name}|{method_name}_{int(time())}",
        search_space=problem.search_space,
        optimization_config=problem.optimization_config,
        runner=problem.runner,
    )


def benchmark_replication(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    seed: int,
) -> BenchmarkResult:
    """Runs one benchmarking replication (equivalent to one optimization loop).

    Args:
        problem: The BenchmarkProblem to test against (can be synthetic or real)
        method: The BenchmarkMethod to test
        seed: The seed to use for this replication.
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

    with with_rng_seed(seed=seed):
        scheduler.run_n_trials(max_trials=problem.num_trials)

    oracle_experiment = problem.get_oracle_experiment_from_experiment(
        experiment=experiment
    )
    optimization_trace = np.array(
        BestPointMixin._get_trace(
            experiment=oracle_experiment,
            optimization_config=problem.optimization_config,
        )
    )

    try:
        # Catch any errors that may occur during score computation, such as errors
        # while accessing "steps" in node based generation strategies. The error
        # handling here is intentionally broad. The score computations is not
        # an essential step of the benchmark runs. We do not want to throw away
        # valuable results after the benchmark run completes just because a
        # non-essential step failed.
        num_baseline_trials = scheduler.standard_generation_strategy._steps[
            0
        ].num_trials

        score_trace = compute_score_trace(
            optimization_trace=optimization_trace,
            num_baseline_trials=num_baseline_trials,
            problem=problem,
        )
    except Exception as e:
        logger.warning(
            f"Failed to compute score trace. Returning NaN. Original error message: {e}"
        )
        score_trace = np.full(len(optimization_trace), np.nan)

    fit_time, gen_time = get_model_times(experiment=experiment)

    return BenchmarkResult(
        name=scheduler.experiment.name,
        seed=seed,
        experiment=scheduler.experiment,
        optimization_trace=optimization_trace,
        score_trace=score_trace,
        fit_time=fit_time,
        gen_time=gen_time,
    )


def benchmark_one_method_problem(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    seeds: Iterable[int],
) -> AggregatedBenchmarkResult:
    return AggregatedBenchmarkResult.from_benchmark_results(
        results=[
            benchmark_replication(problem=problem, method=method, seed=seed)
            for seed in seeds
        ]
    )


def benchmark_multiple_problems_methods(
    problems: Iterable[BenchmarkProblem],
    methods: Iterable[BenchmarkMethod],
    seeds: Iterable[int],
) -> list[AggregatedBenchmarkResult]:
    """
    For each `problem` and `method` in the Cartesian product of `problems` and
    `methods`, run the replication on each seed in `seeds` and get the results
    as an `AggregatedBenchmarkResult`, then return a list of each
    `AggregatedBenchmarkResult`.
    """
    return [
        benchmark_one_method_problem(problem=p, method=m, seeds=seeds)
        for p, m in product(problems, methods)
    ]

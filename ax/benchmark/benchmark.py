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

from collections.abc import Iterable, Mapping
from itertools import product
from logging import Logger
from time import monotonic, time
from typing import Set

import numpy as np
import numpy.typing as npt

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult, BenchmarkResult
from ax.benchmark.benchmark_runner import BenchmarkRunner
from ax.core.experiment import Experiment
from ax.core.types import TParameterization, TParamValue
from ax.core.utils import get_model_times
from ax.service.scheduler import Scheduler
from ax.service.utils.best_point_mixin import BestPointMixin
from ax.utils.common.logger import get_logger
from ax.utils.common.random import with_rng_seed

logger: Logger = get_logger(__name__)


def compute_score_trace(
    optimization_trace: npt.NDArray,
    num_baseline_trials: int,
    problem: BenchmarkProblem,
) -> npt.NDArray:
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


def get_benchmark_runner(
    problem: BenchmarkProblem,
    max_concurrency: int = 1,
) -> BenchmarkRunner:
    """
    Construct a ``BenchmarkRunner`` for the given problem and concurrency.

    If ``max_concurrency > 1`` or if there is a ``sample_runtime_func`` is
    present on ``BenchmarkProblem``, construct a ``SimulatedBenchmarkRunner`` to
    track when trials start and stop.

    Args:
        problem: The ``BenchmarkProblem``; provides a ``BenchmarkTestFunction``
            (used to generate data) and ``trial_runtime_func`` (used to
            determine the length of trials for the simulator).
        max_concurrency: The maximum number of trials that can be run concurrently.
            Typically, ``max_pending_trials`` from ``SchedulerOptions``, which are
            stored on the ``BenchmarkMethod``.
    """
    return BenchmarkRunner(
        test_function=problem.test_function,
        noise_std=problem.noise_std,
        trial_runtime_func=problem.trial_runtime_func,
        max_concurrency=max_concurrency,
    )


def get_oracle_experiment_from_params(
    problem: BenchmarkProblem,
    dict_of_dict_of_params: Mapping[int, Mapping[str, Mapping[str, TParamValue]]],
) -> Experiment:
    """
    Get a new experiment with the same search space and optimization config
    as those belonging to this problem, but with parameterizations evaluated
    at oracle values (noiseless ground-truth values evaluated at the target task
    and fidelity).

    Args:
        problem: ``BenchmarkProblem`` from which to take a test function for
            generating metrics, as well as a search space and optimization
            config for generating an experiment.
        dict_of_dict_of_params: Keys are trial indices, values are Mappings
            (e.g. dicts) that map arm names to parameterizations.

    Example:
        >>> get_oracle_experiment_from_params(
        ...     problem=problem,
        ...     dict_of_dict_of_params={
        ...         0: {
        ...            "0_0": {"x0": 0.0, "x1": 0.0},
        ...            "0_1": {"x0": 0.3, "x1": 0.4},
        ...         },
        ...         1: {"1_0": {"x0": 0.0, "x1": 0.0}},
        ...     }
        ... )
    """

    experiment = Experiment(
        search_space=problem.search_space,
        optimization_config=problem.optimization_config,
    )

    runner = BenchmarkRunner(test_function=problem.test_function, noise_std=0.0)

    for trial_index, dict_of_params in dict_of_dict_of_params.items():
        if len(dict_of_params) == 0:
            raise ValueError(
                "Can't create a trial with no arms. Each sublist in "
                "list_of_list_of_params must have at least one element."
            )
        experiment.attach_trial(
            parameterizations=[
                {**parameters, **problem.target_fidelity_and_task}
                for parameters in dict_of_params.values()
            ],
            arm_names=list(dict_of_params.keys()),
        )
        trial = experiment.trials[trial_index]
        metadata = runner.run(trial=trial)
        trial.update_run_metadata(metadata=metadata)
        trial.mark_completed()

    experiment.fetch_data()
    return experiment


def get_oracle_experiment_from_experiment(
    problem: BenchmarkProblem, experiment: Experiment
) -> Experiment:
    """
    Get an ``Experiment`` that is the same as the original experiment but has
    metrics evaluated at oracle values (noiseless ground-truth values
    evaluated at the target task and fidelity)
    """
    return get_oracle_experiment_from_params(
        problem=problem,
        dict_of_dict_of_params={
            trial.index: {arm.name: arm.parameters for arm in trial.arms}
            for trial in experiment.trials.values()
        },
    )


def benchmark_replication(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    seed: int,
    strip_runner_before_saving: bool = True,
) -> BenchmarkResult:
    """
    Run one benchmarking replication (equivalent to one optimization loop).

    After each trial, the `method` gets the best parameter(s) found so far, as
    evaluated based on empirical data. After all trials are run, the `problem`
    gets the oracle values of each "best" parameter; this yields the ``inference
    trace``. The cumulative maximum of the oracle value of each parameterization
    tested is the ``oracle_trace``.

    Args:
        problem: The BenchmarkProblem to test against (can be synthetic or real)
        method: The BenchmarkMethod to test
        seed: The seed to use for this replication.
        strip_runner_before_saving: Whether to strip the runner from the
            experiment before saving it. This enables serialization.

    Return:
        ``BenchmarkResult`` object.
    """

    runner = get_benchmark_runner(
        problem=problem, max_concurrency=method.scheduler_options.max_pending_trials
    )

    experiment = Experiment(
        name=f"{problem.name}|{method.name}_{int(time())}",
        search_space=problem.search_space,
        optimization_config=problem.optimization_config,
        runner=runner,
    )

    scheduler = Scheduler(
        experiment=experiment,
        generation_strategy=method.generation_strategy.clone_reset(),
        options=method.scheduler_options,
    )

    # list of parameters for each trial
    best_params_by_trial: list[list[TParameterization]] = []

    is_mf_or_mt = len(problem.target_fidelity_and_task) > 0
    trials_used_for_best_point: Set[int] = set()

    # Run the optimization loop.
    timeout_hours = method.timeout_hours
    remaining_hours = timeout_hours
    with with_rng_seed(seed=seed):
        start = monotonic()
        # These next several lines do the same thing as `run_n_trials`, but
        # decrement the timeout with each step, so that the timeout refers to
        # the total time spent in the optimization loop, not time per trial.
        scheduler.poll_and_process_results()
        for _ in scheduler.run_trials_and_yield_results(
            max_trials=problem.num_trials,
            timeout_hours=remaining_hours,
        ):
            if timeout_hours is not None:
                elapsed_hours = (monotonic() - start) / 3600
                remaining_hours = timeout_hours - elapsed_hours
                if remaining_hours <= 0.0:
                    logger.warning("The optimization loop timed out.")
                    break

            if problem.is_moo or is_mf_or_mt:
                # Inference trace is not supported for MOO.
                # It's also not supported for multi-fidelity or multi-task
                # problems, because Ax's best-point functionality doesn't know
                # to predict at the target task or fidelity.
                continue
            currently_completed_trials = {t.index for t in experiment.completed_trials}
            newly_completed_trials = (
                currently_completed_trials - trials_used_for_best_point
            )
            if len(newly_completed_trials) == 0:
                continue
            for t in newly_completed_trials:
                trials_used_for_best_point.add(t)

            best_params = method.get_best_parameters(
                experiment=experiment,
                optimization_config=problem.optimization_config,
                n_points=problem.n_best_points,
            )
            # If multiple trials complete at the same time, add that number of
            # points to the inference trace so that the trace has length equal to
            # the number of trials.
            for _ in newly_completed_trials:
                best_params_by_trial.append(best_params)

        scheduler.summarize_final_result()

    # Construct inference trace from best parameters
    inference_trace = np.full(problem.num_trials, np.nan)
    for trial_index, best_params in enumerate(best_params_by_trial):
        if len(best_params) == 0:
            inference_trace[trial_index] = np.nan
            continue
        # Construct an experiment with one BatchTrial
        best_params_oracle_experiment = get_oracle_experiment_from_params(
            problem=problem,
            dict_of_dict_of_params={0: {str(i): p for i, p in enumerate(best_params)}},
        )
        # Get the optimization trace. It will have only one point.
        inference_trace[trial_index] = BestPointMixin._get_trace(
            experiment=best_params_oracle_experiment,
            optimization_config=problem.optimization_config,
        )[0]

    actual_params_oracle_experiment = get_oracle_experiment_from_experiment(
        problem=problem, experiment=experiment
    )
    oracle_trace = np.array(
        BestPointMixin._get_trace(
            experiment=actual_params_oracle_experiment,
            optimization_config=problem.optimization_config,
        )
    )
    optimization_trace = (
        inference_trace if problem.report_inference_value_as_trace else oracle_trace
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
    if strip_runner_before_saving:
        # Strip runner from experiment before returning, so that the experiment can
        # be serialized (the runner can't be)
        experiment.runner = None

    return BenchmarkResult(
        name=scheduler.experiment.name,
        seed=seed,
        experiment=scheduler.experiment,
        oracle_trace=oracle_trace,
        inference_trace=inference_trace,
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

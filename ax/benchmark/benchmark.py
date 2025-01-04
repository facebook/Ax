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

import warnings
from collections.abc import Iterable, Mapping
from itertools import product
from logging import Logger, WARNING
from time import monotonic, time
from typing import Set

import numpy as np
import numpy.typing as npt
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult, BenchmarkResult
from ax.benchmark.benchmark_runner import BenchmarkRunner
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.benchmark.methods.sobol import get_sobol_generation_strategy
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization, TParamValue
from ax.core.utils import get_model_times
from ax.service.scheduler import Scheduler
from ax.service.utils.best_point_mixin import BestPointMixin
from ax.service.utils.scheduler_options import SchedulerOptions, TrialType
from ax.utils.common.logger import DEFAULT_LOG_LEVEL, get_logger
from ax.utils.common.random import with_rng_seed

logger: Logger = get_logger(__name__)


def compute_score_trace(
    optimization_trace: npt.NDArray, baseline_value: float, optimal_value: float
) -> npt.NDArray:
    """
    Compute a score trace from the optimization trace.

    Score is expressed as a percentage of possible improvement over a baseline.
    A higher score is better.

    Element `i` of the score trace is `optimization_trace[i] - baseline`
    expressed as a percent of `optimal_value - baseline`, where `baseline` is
    `optimization_trace[num_baseline_trials - 1]`. It can be over 100 if values
    better than `optimal_value` are attained or below 0 if values worse than the
    baseline value are attained.

    Args:
        optimization_trace: Objective values. Can be either higher- or
            lower-is-better.
        baseline_value: Value to use as a baseline. Any values that are not
            better than the baseline will receive negative scores.
        optimal_value: The best possible value of the objective; when the
            optimization_trace equals the optimal_value, the score is 100.
    """
    return (
        100 * (optimization_trace - baseline_value) / (optimal_value - baseline_value)
    )


def get_benchmark_runner(
    problem: BenchmarkProblem, max_concurrency: int = 1
) -> BenchmarkRunner:
    """
    Construct a ``BenchmarkRunner`` for the given problem and concurrency.

    If ``max_concurrency > 1`` or if there is a ``sample_runtime_func`` is
    present on ``BenchmarkProblem``, construct a ``SimulatedBenchmarkRunner`` to
    track when trials start and stop.

    Args:
        problem: The ``BenchmarkProblem``; provides a ``BenchmarkTestFunction``
            (used to generate data) and ``step_runtime_function`` (used to
            determine timing for the simulator).
        max_concurrency: The maximum number of trials that can be run concurrently.
            Typically, ``max_pending_trials`` from ``SchedulerOptions``, which are
            stored on the ``BenchmarkMethod``.
    """

    return BenchmarkRunner(
        test_function=problem.test_function,
        noise_std=problem.noise_std,
        step_runtime_function=problem.step_runtime_function,
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

    # Silence INFO logs from ax.core.experiment that state "Attached custom
    # parameterizations"
    logger = get_logger("ax.core.experiment")
    original_log_level = logger.level
    logger.setLevel(level="WARNING")

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

    logger.setLevel(level=original_log_level)

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


def get_benchmark_scheduler_options(
    method: BenchmarkMethod,
    include_sq: bool = False,
    logging_level: int = DEFAULT_LOG_LEVEL,
) -> SchedulerOptions:
    """
    Get the ``SchedulerOptions`` for the given ``BenchmarkMethod``.

    Args:
        method: The ``BenchmarkMethod``.
        include_sq: Whether to include the status quo in each trial.

    Returns:
        ``SchedulerOptions``
    """
    if method.batch_size > 1 or include_sq:
        trial_type = TrialType.BATCH_TRIAL
    else:
        trial_type = TrialType.TRIAL
    return SchedulerOptions(
        # No new candidates can be generated while any are pending.
        # If batched, an entire batch must finish before the next can be
        # generated.
        max_pending_trials=method.max_pending_trials,
        # Do not throttle, as is often necessary when polling real endpoints
        init_seconds_between_polls=0,
        min_seconds_before_poll=0,
        trial_type=trial_type,
        batch_size=method.batch_size,
        run_trials_in_batches=method.run_trials_in_batches,
        early_stopping_strategy=method.early_stopping_strategy,
        status_quo_weight=1.0 if include_sq else 0.0,
        logging_level=logging_level,
    )


def benchmark_replication(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    seed: int,
    strip_runner_before_saving: bool = True,
    scheduler_logging_level: int = DEFAULT_LOG_LEVEL,
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
    sq_arm = (
        None
        if problem.status_quo_params is None
        else Arm(name="status_quo", parameters=problem.status_quo_params)
    )
    scheduler_options = get_benchmark_scheduler_options(
        method=method,
        include_sq=sq_arm is not None,
        logging_level=scheduler_logging_level,
    )
    runner = get_benchmark_runner(
        problem=problem,
        max_concurrency=scheduler_options.max_pending_trials,
    )
    experiment = Experiment(
        name=f"{problem.name}|{method.name}_{int(time())}",
        search_space=problem.search_space,
        optimization_config=problem.optimization_config,
        runner=runner,
        status_quo=sq_arm,
        auxiliary_experiments_by_purpose=problem.auxiliary_experiments_by_purpose,
    )

    scheduler = Scheduler(
        experiment=experiment,
        generation_strategy=method.generation_strategy.clone_reset(),
        options=scheduler_options,
    )

    # list of parameters for each trial
    best_params_by_trial: list[list[TParameterization]] = []

    is_mf_or_mt = len(problem.target_fidelity_and_task) > 0
    trials_used_for_best_point: Set[int] = set()

    # Run the optimization loop.
    timeout_hours = method.timeout_hours
    remaining_hours = timeout_hours

    with with_rng_seed(seed=seed), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Encountered exception in computing model fit quality",
            category=UserWarning,
            module="ax.modelbridge.cross_validation",
        )
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

            currently_completed_trials = {
                t.index
                for t in experiment.trials.values()
                if t.status
                in (
                    TrialStatus.COMPLETED,
                    TrialStatus.EARLY_STOPPED,
                )
            }
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

    score_trace = compute_score_trace(
        optimization_trace=optimization_trace,
        optimal_value=problem.optimal_value,
        baseline_value=problem.baseline_value,
    )

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


def compute_baseline_value_from_sobol(
    optimization_config: OptimizationConfig,
    search_space: SearchSpace,
    test_function: BenchmarkTestFunction,
    target_fidelity_and_task: Mapping[str, TParamValue] | None = None,
    n_repeats: int = 50,
) -> float:
    """
    Compute the `baseline_value` that will be assigned to
    a `BenchmarkProblem`.

    Computed by taking the best of five quasi-random Sobol trials and then
    repeating 50 times. The value is evaluated at the ground truth (noiseless
    and at the target task and fidelity).

    Args:
        optimization_config: Typically, the `optimization_config` of a
            `BenchmarkProblem` (or that will later be used to define a
            `BenchmarkProblem`).
        search_space: Similarly, the `search_space` of a `BenchmarkProblem`.
        test_function: Similarly, the `test_function` of a `BenchmarkProblem`.
        target_fidelity_and_task: Typically, the `target_fidelity_and_task` of a
            `BenchmarkProblem`.
        n_repeats: Number of times to repeat the five Sobol trials.
    """
    gs = get_sobol_generation_strategy()
    method = BenchmarkMethod(generation_strategy=gs)
    target_fidelity_and_task = {} if target_fidelity_and_task is None else {}

    # set up a dummy problem so we can use `benchmark_replication`
    # MOO problems are always higher-is-better because they use hypervolume
    higher_is_better = isinstance(optimization_config.objective, MultiObjective) or (
        not optimization_config.objective.minimize
    )
    dummy_optimal_value = float("inf") if higher_is_better else float("-inf")
    dummy_problem = BenchmarkProblem(
        name="dummy",
        optimization_config=optimization_config,
        search_space=search_space,
        num_trials=5,
        test_function=test_function,
        optimal_value=dummy_optimal_value,
        baseline_value=-dummy_optimal_value,
        target_fidelity_and_task=target_fidelity_and_task,
    )

    values = np.full(n_repeats, np.nan)
    for i in range(n_repeats):
        result = benchmark_replication(
            problem=dummy_problem,
            method=method,
            seed=i,
            scheduler_logging_level=WARNING,
        )
        values[i] = result.optimization_trace[-1]

    return values.mean()


def benchmark_one_method_problem(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    seeds: Iterable[int],
    scheduler_logging_level: int = DEFAULT_LOG_LEVEL,
) -> AggregatedBenchmarkResult:
    return AggregatedBenchmarkResult.from_benchmark_results(
        results=[
            benchmark_replication(
                problem=problem,
                method=method,
                seed=seed,
                scheduler_logging_level=scheduler_logging_level,
            )
            for seed in seeds
        ]
    )


def benchmark_multiple_problems_methods(
    problems: Iterable[BenchmarkProblem],
    methods: Iterable[BenchmarkMethod],
    seeds: Iterable[int],
    scheduler_logging_level: int = DEFAULT_LOG_LEVEL,
) -> list[AggregatedBenchmarkResult]:
    """
    For each `problem` and `method` in the Cartesian product of `problems` and
    `methods`, run the replication on each seed in `seeds` and get the results
    as an `AggregatedBenchmarkResult`, then return a list of each
    `AggregatedBenchmarkResult`.
    """
    return [
        benchmark_one_method_problem(
            problem=p,
            method=m,
            seeds=seeds,
            scheduler_logging_level=scheduler_logging_level,
        )
        for p, m in product(problems, methods)
    ]

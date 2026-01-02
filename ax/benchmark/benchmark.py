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
* Method: (one of) the algorithm(s) being benchmarked.
* Full run: multiple tests on many (BenchmarkProblem, BenchmarkMethod) pairs.
* Problem: a synthetic function, a surrogate surface, or an ML model, on which
  to assess the performance of algorithms.

"""

import math
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from itertools import product
from logging import Logger, WARNING
from time import time

import numpy as np
import numpy.typing as npt
import pandas as pd
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult, BenchmarkResult
from ax.benchmark.benchmark_runner import BenchmarkRunner
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.benchmark.methods.sobol import get_sobol_benchmark_method
from ax.core.arm import Arm
from ax.core.data import MAP_KEY
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.search_space import SearchSpace
from ax.core.trial import BaseTrial, Trial
from ax.core.types import TParameterization, TParamValue
from ax.core.utils import get_model_times
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.orchestration.orchestrator import Orchestrator
from ax.service.utils.best_point import (
    _prepare_data_for_trace,
    derelativize_opt_config,
    get_trace,
)
from ax.service.utils.best_point_mixin import BestPointMixin
from ax.service.utils.orchestrator_options import OrchestratorOptions, TrialType
from ax.utils.common.logger import DEFAULT_LOG_LEVEL, get_logger
from ax.utils.common.random import with_rng_seed
from ax.utils.testing.backend_simulator import BackendSimulator

from pyre_extensions import assert_is_instance, none_throws

logger: Logger = get_logger(__name__)


def update_trials_to_use_sim_time_in_place(
    trials: dict[int, BaseTrial], simulator: BackendSimulator
) -> None:
    """
    Update the start and end times of all trials to be in simulated time
    (represented as datetime objects -- seconds since the start of time).
    """
    fromtimestamp = datetime.fromtimestamp
    for trial_index, trial in trials.items():
        sim_trial = none_throws(
            simulator.get_sim_trial_by_index(trial_index=trial_index)
        )
        trial._time_created = fromtimestamp(
            timestamp=none_throws(sim_trial.sim_queued_time)
        )
        trial._time_staged = fromtimestamp(
            timestamp=none_throws(sim_trial.sim_queued_time)
        )
        trial._time_completed = fromtimestamp(
            timestamp=none_throws(sim_trial.sim_completed_time)
        )
        trial._time_run_started = fromtimestamp(none_throws(sim_trial.sim_start_time))


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
    problem: BenchmarkProblem,
    max_concurrency: int = 1,
    force_use_simulated_backend: bool = False,
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
            Typically, ``max_pending_trials`` from ``OrchestratorOptions``, which are
            stored on the ``BenchmarkMethod``.
        force_use_simulated_backend: Whether to use a simulated backend even if
            ``max_concurrency`` is 1 and ``problem.step_runtime_function`` is
            None. Recommended for use with a ``BenchmarkMethod`` that uses early
            stopping.
    """

    return BenchmarkRunner(
        test_function=problem.test_function,
        noise_std=problem.noise_std,
        step_runtime_function=problem.step_runtime_function,
        max_concurrency=max_concurrency,
        force_use_simulated_backend=force_use_simulated_backend,
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


def get_benchmark_orchestrator_options(
    batch_size: int | None,
    run_trials_in_batches: bool,
    max_pending_trials: int,
    early_stopping_strategy: BaseEarlyStoppingStrategy | None,
    include_status_quo: bool = False,
    logging_level: int = DEFAULT_LOG_LEVEL,
) -> OrchestratorOptions:
    """
    Get the ``OrchestratorOptions`` for the given ``BenchmarkMethod``.

    Args:
        batch_size: The batch size to use for the optimiation.
        run_trials_in_batches: Whether to run trials in batches. This is used
            for high-throughput settings where there are many trials and
            generating them in bulk reduces overhead (not to be confused with
            `BatchTrial`s, which are different).
        max_pending_trials: The maximum number of pending trials allowed.
        early_stopping_strategy: The early stopping strategy to use (if any).
        include_status_quo: Whether to include the status quo in each trial.
        logging_level: The logging level to use for the Orchestrator.

    Returns:
        ``OrchestratorOptions``
    """
    if batch_size is None or batch_size > 1 or include_status_quo:
        trial_type = TrialType.BATCH_TRIAL
    else:
        trial_type = TrialType.TRIAL
    return OrchestratorOptions(
        # No new candidates can be generated while any are pending.
        # If batched, an entire batch must finish before the next can be generated.
        max_pending_trials=max_pending_trials,
        # Do not throttle, as is often necessary when polling real endpoints
        init_seconds_between_polls=0,
        min_seconds_before_poll=0,
        trial_type=trial_type,
        batch_size=batch_size,
        run_trials_in_batches=run_trials_in_batches,
        early_stopping_strategy=early_stopping_strategy,
        status_quo_weight=1.0 if include_status_quo else 0.0,
        logging_level=logging_level,
    )


def _get_oracle_value_of_params(
    params: Mapping[str, TParamValue], problem: BenchmarkProblem
) -> float:
    """
    A roundabout way of getting the value of a parameterization:
    1. Construct an experiment with the parameterization as its only trial,
        using the BenchmarkProblem to get the oracle value of that
        parameterization.
    2. Get the optimization trace of that experiment.
    """
    dummy_experiment = get_oracle_experiment_from_params(
        problem=problem, dict_of_dict_of_params={0: {"0_0": params}}
    )
    (inference_value,) = get_trace(
        experiment=dummy_experiment, optimization_config=problem.optimization_config
    )
    return inference_value


def get_inference_trace(
    trial_completion_order: Sequence[set[int]],
    experiment: Experiment,
    generation_strategy: GenerationStrategy,
    problem: BenchmarkProblem,
) -> npt.NDArray:
    """
    Get the inference trace from a completed experiment.

    The inference trace is the value of the parameterization that would have
    been predicted to be best at each time when a trial completes, using only
    information that would have been available at the time.

    Args:
        trial_completion_order: A list of sets of trial indices, where the first
            element includes the trials that finished first (all at the same
            time), the second element is trials that finished next after that,
            etc.
        experiment: Passed to ``get_trace``.
        generation_strategy: Passed to ``get_trace``.
        problem: Used to get the oracle value of each parameterization.

    """
    completed_trial_idcs: set[int] = set()
    inference_trace = np.full(
        shape=(len(trial_completion_order),), fill_value=float("NaN")
    )

    # Inference trace is not supported for MOO.
    if isinstance(experiment.optimization_config, MultiObjectiveOptimizationConfig):
        return inference_trace

    for i, newly_completed_trials in enumerate(trial_completion_order):
        completed_trial_idcs |= newly_completed_trials
        # Note: Ax's best-point functionality doesn't know to predict at the
        # target task or fidelity, so this won't produce good recommendations in
        # MF/MT settings.
        best_params = get_best_parameters(
            experiment=experiment,
            generation_strategy=generation_strategy,
            trial_indices=completed_trial_idcs,
        )
        if best_params is not None:
            inference_trace[i] = _get_oracle_value_of_params(
                params=best_params, problem=problem
            )
    return inference_trace


def get_is_feasible_trace(
    experiment: Experiment, optimization_config: OptimizationConfig
) -> list[float]:
    """Get a trace of feasibility for the experiment.

    For batch trials we return True if any arm in a given batch is feasible.
    """
    df = experiment.lookup_data().df.copy()  # Let's not modify the original df
    if len(df) == 0:
        return []
    # Derelativize the optimization config if needed.
    optimization_config = derelativize_opt_config(
        optimization_config=optimization_config,
        experiment=experiment,
    )
    # Compute feasibility and return feasibility per group
    df = _prepare_data_for_trace(df=df, optimization_config=optimization_config)
    trial_grouped = df.groupby("trial_index")["feasible"]
    return trial_grouped.any().tolist()


def get_best_parameters(
    experiment: Experiment,
    generation_strategy: GenerationStrategy,
    trial_indices: Iterable[int] | None = None,
) -> TParameterization | None:
    """
    Get the most promising point.

    Only SOO is supported. It will return None if no best point can be found.

    Args:
        experiment: The experiment to get the data from. This should contain
            values that would be observed in a realistic setting and not
            contain oracle values.
        generation_strategy: The ``GenerationStrategy`` to use to predict the
            best point.
        trial_indices: Use data from only these trials. If None, use all data.
    """
    result = BestPointMixin._get_best_trial(
        experiment=experiment,
        generation_strategy=generation_strategy,
        trial_indices=trial_indices,
    )
    if result is None:
        # This can happen if no points are predicted to satisfy all outcome
        # constraints.
        return None
    _, params, _ = none_throws(result)
    return params


def get_benchmark_result_from_experiment_and_gs(
    experiment: Experiment,
    generation_strategy: GenerationStrategy,
    problem: BenchmarkProblem,
    seed: int,
    strip_runner_before_saving: bool = True,
) -> BenchmarkResult:
    """
    Parse the ``Experiment`` and ``GenerationStrategy`` into a ``BenchmarkResult``.

    All results are ordered according to ``trial_completion_order``.
    After all trials have been run, the `problem` gets the oracle values of each
    "best" parameter; this yields the ``inference trace``. The cumulative
    maximum of the oracle value of each parameterization tested is the
    ``oracle_trace``.

    Args:
        experiment: The completed ``Experiment`` to extract results from.
        generation_strategy: The ``GenerationStrategy`` used to generate
            ``experiment``; it will be ultimately passed to best-point utilities
            in order to generate the ``inference_trace`` on ``BenchmarkResult``.
        problem: The ``BenchmarkProblem`` used to generate ``experiment``. It
            will be used to extract the oracle values of parameterizations, and
            its ``OptimizationConfig`` is used for identifying best points.
        seed: The seed used to generate ``experiment``.
        strip_runner_before_saving: Whether to write the experiment's runner to
            the returned ``BenchmarkResult``.
    """

    runner = assert_is_instance(experiment.runner, BenchmarkRunner)
    sim_runner = runner.simulated_backend_runner
    if sim_runner is not None:
        trial_indices_by_completion_time: dict[datetime, set[int]] = defaultdict(set)
        for trial_index, trial in experiment.trials.items():
            trial_indices_by_completion_time[none_throws(trial._time_completed)].add(
                trial_index
            )
        trial_completion_order = [
            trial_indices_by_completion_time[k]
            for k in sorted(trial_indices_by_completion_time.keys())
        ]
        cost_trace = np.array(
            [
                completion_time.timestamp()
                for completion_time in sorted(trial_indices_by_completion_time.keys())
            ]
        )
    else:
        trial_completion_order = [{i} for i in range(len(experiment.trials))]
        cost_trace = 1.0 + np.arange(len(experiment.trials), dtype=float)

    # {trial_index: {arm_name: params}}
    dict_of_dict_of_params = {
        new_trial_index: {
            arm.name: arm.parameters
            for old_trial_index in trials
            for arm in experiment.trials[old_trial_index].arms
        }
        for new_trial_index, trials in enumerate(trial_completion_order)
    }

    actual_params_oracle_dummy_experiment = get_oracle_experiment_from_params(
        problem=problem, dict_of_dict_of_params=dict_of_dict_of_params
    )
    oracle_trace = np.array(
        get_trace(
            experiment=actual_params_oracle_dummy_experiment,
            optimization_config=problem.optimization_config,
        )
    )
    is_feasible_trace = np.array(
        get_is_feasible_trace(
            experiment=actual_params_oracle_dummy_experiment,
            optimization_config=problem.optimization_config,
        )
    )
    if problem.report_inference_value_as_trace:
        inference_trace = get_inference_trace(
            trial_completion_order=trial_completion_order,
            experiment=experiment,
            problem=problem,
            generation_strategy=generation_strategy,
        )
        optimization_trace = inference_trace
    else:
        inference_trace = np.full_like(oracle_trace, fill_value=np.nan)
        optimization_trace = oracle_trace

    # Need to modify the optimization trace for constrained problems
    if len(problem.optimization_config.outcome_constraints) > 0:
        inds_is_feas = np.where(is_feasible_trace)[0]
        infeasible_inds = (
            np.arange(len(optimization_trace))
            if len(inds_is_feas) == 0
            else np.arange(inds_is_feas[0])
        )
        oracle_trace[infeasible_inds] = problem.worst_feasible_value
        if problem.report_inference_value_as_trace:
            # Note: The inference trace isn't cumulative.
            inference_trace[~is_feasible_trace] = problem.worst_feasible_value
            optimization_trace[~is_feasible_trace] = problem.worst_feasible_value
        else:
            optimization_trace[infeasible_inds] = problem.worst_feasible_value

        baseline_value = (
            none_throws(problem.worst_feasible_value)
            if not math.isfinite(problem.baseline_value)
            else problem.baseline_value
        )
        score_trace = compute_score_trace(
            optimization_trace=optimization_trace,
            optimal_value=problem.optimal_value,
            baseline_value=baseline_value,
        )
    else:
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
        name=experiment.name,
        seed=seed,
        experiment=experiment,
        oracle_trace=oracle_trace.tolist(),
        inference_trace=inference_trace.tolist(),
        optimization_trace=optimization_trace.tolist(),
        is_feasible_trace=is_feasible_trace.tolist(),
        score_trace=score_trace.tolist(),
        cost_trace=cost_trace.tolist(),
        fit_time=fit_time,
        gen_time=gen_time,
    )


def run_optimization_with_orchestrator(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    seed: int,
    run_trials_in_batches: bool = False,
    timeout_hours: float | None = None,
    orchestrator_logging_level: int = DEFAULT_LOG_LEVEL,
) -> Experiment:
    """
    Optimize the ``problem`` using the ``method`` and ``Orchestrator``, seeding
    the optimization with ``seed``.

    Args:
        problem: The BenchmarkProblem to test against (can be synthetic or real)
        method: The BenchmarkMethod to test
        seed: The seed to use for this replication.
        run_trials_in_batches: Whether to run trials in batches. This is used
            for high-throughput settings where there are many trials and
            generating them in bulk reduces overhead (not to be confused with
            `BatchTrial`s, which are different).
        timeout_hours: The maximum number of hours for which to run the
            optimization loop before timing out.
        orchestrator_logging_level: If >INFO, logs will only appear when unexpected
            things happen. If INFO, logs will update when a trial is completed
            and when an early stopping strategy, if present, decides whether or
            not to continue a trial. If DEBUG, logs additionally include
            information from a `BackendSimulator`, if present.

    Return:
        ``Experiment`` object.
    """
    sq_arm = (
        None
        if problem.status_quo_params is None
        else Arm(name="status_quo", parameters=problem.status_quo_params)
    )
    orchestrator_options = get_benchmark_orchestrator_options(
        batch_size=method.batch_size,
        run_trials_in_batches=run_trials_in_batches,
        max_pending_trials=method.max_pending_trials,
        early_stopping_strategy=method.early_stopping_strategy,
        include_status_quo=sq_arm is not None,
        logging_level=orchestrator_logging_level,
    )
    runner = get_benchmark_runner(
        problem=problem,
        max_concurrency=orchestrator_options.max_pending_trials,
        force_use_simulated_backend=method.early_stopping_strategy is not None,
    )
    experiment = Experiment(
        name=f"{problem.name}|{method.name}_{int(time())}",
        search_space=problem.search_space,
        optimization_config=problem.optimization_config,
        runner=runner,
        status_quo=sq_arm,
        tracking_metrics=problem.tracking_metrics,
        auxiliary_experiments_by_purpose=problem.auxiliary_experiments_by_purpose,
    )

    orchestrator = Orchestrator(
        experiment=experiment,
        generation_strategy=method.generation_strategy.clone_reset(),
        options=orchestrator_options,
    )

    with with_rng_seed(seed=seed), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Encountered exception in computing model fit quality",
            category=UserWarning,
            module="ax.adapter.cross_validation",
        )
        orchestrator.run_n_trials(
            max_trials=problem.num_trials, timeout_hours=timeout_hours
        )

    sim_runner = runner.simulated_backend_runner
    if sim_runner is not None:
        simulator = sim_runner.simulator
        update_trials_to_use_sim_time_in_place(
            trials=experiment.trials, simulator=simulator
        )

    return experiment


def benchmark_replication(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    seed: int,
    run_trials_in_batches: bool = False,
    timeout_hours: float = 4.0,
    orchestrator_logging_level: int = DEFAULT_LOG_LEVEL,
    strip_runner_before_saving: bool = True,
) -> BenchmarkResult:
    """
    Run one benchmarking replication (equivalent to one optimization loop).

    Optimize the ``problem`` using the ``method`` and ``Orchestrator``, seeding
    the optimization with ``seed``. This produces an ``Experiment``. Then parse
    the ``Experiment`` into a ``BenchmarkResult``, extracting traces.


    Args:
        problem: The BenchmarkProblem to test against (can be synthetic or real)
        method: The BenchmarkMethod to test
        seed: The seed to use for this replication.
        run_trials_in_batches: Whether to run trials in batches. This is used
            for high-throughput settings where there are many trials and
            generating them in bulk reduces overhead (not to be confused with
            `BatchTrial`s, which are different).
        timeout_hours: The maximum number of hours for which to run the
            optimization loop before timing out.
        orchestrator_logging_level: If >INFO, logs will only appear when unexpected
            things happen. If INFO, logs will update when a trial is completed
            and when an early stopping strategy, if present, decides whether or
            not to continue a trial. If DEBUG, logs additionally include
            information from a ``BackendSimulator``, if present.
        strip_runner_before_saving: Whether to strip the runner from the
            experiment before saving it. This enables serialization.

    Return:
        ``BenchmarkResult`` object.
    """
    experiment = run_optimization_with_orchestrator(
        problem=problem,
        method=method,
        seed=seed,
        run_trials_in_batches=run_trials_in_batches,
        timeout_hours=timeout_hours,
        orchestrator_logging_level=orchestrator_logging_level,
    )

    benchmark_result = get_benchmark_result_from_experiment_and_gs(
        experiment=experiment,
        generation_strategy=method.generation_strategy,
        problem=problem,
        seed=seed,
        strip_runner_before_saving=strip_runner_before_saving,
    )
    return benchmark_result


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
    method = get_sobol_benchmark_method()
    target_fidelity_and_task = {} if target_fidelity_and_task is None else {}

    # set up a dummy problem so we can use `benchmark_replication`
    # MOO problems are always higher-is-better because they use hypervolume
    higher_is_better = isinstance(optimization_config.objective, MultiObjective) or (
        not optimization_config.objective.minimize
    )
    dummy_problem = BenchmarkProblem(
        name="dummy",
        optimization_config=optimization_config,
        num_trials=5,
        test_function=test_function,
        # Optimal value and baseline value are only used to compute the score_trace,
        # which we don't use here. The order of baseline and optimal value needs to
        # be correct, though, as a ValueError is raised otherwise.
        optimal_value=1.0 if higher_is_better else -1.0,
        baseline_value=0.0,
        search_space=search_space,
        target_fidelity_and_task=target_fidelity_and_task,
    )

    values = np.full(n_repeats, np.nan)
    for i in range(n_repeats):
        result = benchmark_replication(
            problem=dummy_problem,
            method=method,
            seed=i,
            run_trials_in_batches=False,
            timeout_hours=0.1,
            orchestrator_logging_level=WARNING,
        )
        values[i] = result.optimization_trace[-1]

    return values.mean().item()


def benchmark_one_method_problem(
    problem: BenchmarkProblem,
    method: BenchmarkMethod,
    seeds: Iterable[int],
    run_trials_in_batches: bool = False,
    timeout_hours: float = 4.0,
    orchestrator_logging_level: int = DEFAULT_LOG_LEVEL,
) -> AggregatedBenchmarkResult:
    return AggregatedBenchmarkResult.from_benchmark_results(
        results=[
            benchmark_replication(
                problem=problem,
                method=method,
                seed=seed,
                run_trials_in_batches=run_trials_in_batches,
                timeout_hours=timeout_hours,
                orchestrator_logging_level=orchestrator_logging_level,
            )
            for seed in seeds
        ]
    )


def benchmark_multiple_problems_methods(
    problems: Iterable[BenchmarkProblem],
    methods: Iterable[BenchmarkMethod],
    seeds: Iterable[int],
    run_trials_in_batches: bool = False,
    timeout_hours: float = 4.0,
    orchestrator_logging_level: int = DEFAULT_LOG_LEVEL,
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
            run_trials_in_batches=run_trials_in_batches,
            timeout_hours=timeout_hours,
            orchestrator_logging_level=orchestrator_logging_level,
        )
        for p, m in product(problems, methods)
    ]


def get_opt_trace_by_steps(experiment: Experiment) -> npt.NDArray:
    """
    Transform an optimization trace in the standard format produced by
    `benchmark_replication`, with one element per trial completion, into a trace
    that is in terms of steps, with one element added each time a step
    completes.

    Args:
        experiment: An experiment produced by `benchmark_replication`; it must
            have `BenchmarkTrialMetadata` (as produced by `BenchmarkRunner`) for
            each trial, and its data must have a "step" column.
    """
    optimization_config = none_throws(experiment.optimization_config)

    if optimization_config.is_moo_problem:
        raise NotImplementedError(
            "Cumulative epochs only supported for single objective problems."
        )
    if len(optimization_config.outcome_constraints) > 0:
        raise NotImplementedError(
            "Cumulative epochs not supported for problems with outcome constraints."
        )

    objective_name = optimization_config.objective.metric.name
    data = experiment.lookup_data()
    full_df = data.full_df

    # Has timestamps; needs to be merged with full_df because it contains
    # data on epochs that didn't actually run due to early stopping, and we need
    # to know which actually ran
    def _get_df(trial: Trial) -> pd.DataFrame:
        """
        Get the (virtual) time each epoch finished at.
        """
        metadata = trial.run_metadata["benchmark_metadata"]
        backend_simulator = none_throws(metadata.backend_simulator)
        # Data for the first metric, which is the only metric
        df = next(iter(metadata.dfs.values()))
        start_time = backend_simulator.get_sim_trial_by_index(
            trial.index
        ).sim_start_time
        df["time"] = df["virtual runtime"] + start_time
        return df

    with_timestamps = pd.concat(
        (
            _get_df(trial=assert_is_instance(trial, Trial))
            for trial in experiment.trials.values()
        ),
        axis=0,
        ignore_index=True,
    )[["trial_index", MAP_KEY, "time"]]

    df = (
        full_df.loc[
            full_df["metric_name"] == objective_name,
            ["trial_index", "arm_name", "mean", MAP_KEY],
        ]
        .merge(with_timestamps, how="left")
        .sort_values("time", ignore_index=True)
    )
    return (
        df["mean"].cummin()
        if optimization_config.objective.minimize
        else df["mean"].cummax()
    ).to_numpy()


def get_benchmark_result_with_cumulative_steps(
    result: BenchmarkResult,
    optimal_value: float,
    baseline_value: float,
) -> BenchmarkResult:
    """
    Replaces the cost trace with the cumulative number of steps run and
    recomputes the optimization trace accordingly, using
    `get_opt_trace_by_steps`.
    """

    experiment = none_throws(result.experiment)
    opt_trace = get_opt_trace_by_steps(experiment=experiment)
    return replace(
        result,
        optimization_trace=opt_trace,
        cost_trace=np.arange(1, len(opt_trace) + 1, dtype=int),
        # Empty
        oracle_trace=np.full(len(opt_trace), np.nan),
        inference_trace=np.full(len(opt_trace), np.nan),
        score_trace=compute_score_trace(
            optimization_trace=opt_trace,
            baseline_value=baseline_value,
            optimal_value=optimal_value,
        ),
    )

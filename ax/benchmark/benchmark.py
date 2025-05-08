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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from itertools import product
from logging import Logger, WARNING
from time import monotonic, time

import numpy as np
import numpy.typing as npt
import pandas as pd
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult, BenchmarkResult
from ax.benchmark.benchmark_runner import BenchmarkRunner, get_total_runtime
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.benchmark.methods.sobol import get_sobol_generation_strategy
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.objective import MultiObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.trial import BaseTrial, Trial
from ax.core.trial_status import TrialStatus
from ax.core.types import TParamValue
from ax.core.utils import get_model_times
from ax.service.scheduler import Scheduler
from ax.service.utils.best_point import get_trace
from ax.service.utils.scheduler_options import SchedulerOptions, TrialType
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
            Typically, ``max_pending_trials`` from ``SchedulerOptions``, which are
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
    if method.batch_size is None or method.batch_size > 1 or include_sq:
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


def _get_cumulative_cost(
    previous_cost: float,
    new_trials: set[int],
    experiment: Experiment,
) -> float:
    """
    Get the total cost of running a benchmark where `new_trials` have just
    completed, and the cost up to that point was `previous_cost`.

    If a backend simulator is used to track runtime the cost is just the
    simulated time. If there is no backend simulator, it is still possible that
    trials have varying runtimes without that being simulated, so in that case,
    runtimes are computed.
    """
    runner = assert_is_instance(experiment.runner, BenchmarkRunner)
    if runner.simulated_backend_runner is not None:
        return runner.simulated_backend_runner.simulator.time

    per_trial_times = (
        get_total_runtime(
            trial=experiment.trials[i],
            step_runtime_function=runner.step_runtime_function,
            n_steps=runner.test_function.n_steps,
        )
        for i in new_trials
    )
    return previous_cost + sum(per_trial_times)


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


def _get_oracle_trace_from_arms(
    evaluated_arms_list: Iterable[set[Arm]], problem: BenchmarkProblem
) -> npt.NDArray:
    """
    Get the oracle trace from a list of arms.

    1. Construct a dummy experiment where trial ``i`` contains the arms in
        ``evaluated_arms_list[i]``; if there are multiple arms, it will be a
        ``BatchTrial``. Its data will be at oracle values.
    2. Get the optimization trace of that experiment.
    """
    dummy_experiment = get_oracle_experiment_from_params(
        problem=problem,
        dict_of_dict_of_params={
            i: {arm.name: arm.parameters for arm in arms}
            for i, arms in enumerate(evaluated_arms_list)
        },
    )
    oracle_trace = get_trace(
        experiment=dummy_experiment,
        optimization_config=problem.optimization_config,
    )
    return np.array(oracle_trace)


def _get_inference_trace_from_params(
    best_params_list: Sequence[Mapping[str, TParamValue]],
    problem: BenchmarkProblem,
    n_elements: int,
) -> npt.NDArray:
    """
    Get the inference value of each parameterization in ``best_params_list``.

    ``best_params_list`` can be empty, indicating that inference value is not
    supported for this benchmark, in which case the returned array will be all
    NaNs with length ``n_elements``. If it is not empty, it must have length
    ``n_elements``.
    """
    if len(best_params_list) == 0:
        return np.full(n_elements, np.nan)
    if len(best_params_list) != n_elements:
        raise RuntimeError(
            f"Expected {n_elements} elements in `best_params_list`, got "
            f"{len(best_params_list)}."
        )
    return np.array(
        [
            _get_oracle_value_of_params(params=params, problem=problem)
            for params in best_params_list
        ]
    )


def _update_benchmark_tracking_vars_in_place(
    experiment: Experiment,
    method: BenchmarkMethod,
    problem: BenchmarkProblem,
    cost_trace: list[float],
    evaluated_arms_list: list[set[Arm]],
    best_params_list: list[Mapping[str, TParamValue]],
    completed_trial_idcs: set[int],
) -> None:
    """
    Update cost_trace, evaluated_arms_list, best_params_list, and
    completed_trial_idcs in place.
    """
    currently_completed_trial_idcs = {
        t.index
        for t in experiment.trials.values()
        if t.status
        in (
            TrialStatus.COMPLETED,
            TrialStatus.EARLY_STOPPED,
        )
    }
    newly_completed_trials = currently_completed_trial_idcs - completed_trial_idcs
    completed_trial_idcs |= newly_completed_trials

    is_mf_or_mt = len(problem.target_fidelity_and_task) > 0
    compute_best_params = not (problem.is_moo or is_mf_or_mt)

    if len(newly_completed_trials) > 0:
        previous_cost = cost_trace[-1] if len(cost_trace) > 0 else 0.0
        cost = _get_cumulative_cost(
            new_trials=newly_completed_trials,
            experiment=experiment,
            previous_cost=previous_cost,
        )
        cost_trace.append(cost)

        # Track what params are newly evaluated from those trials, for
        # the oracle trace
        params = {
            arm for i in newly_completed_trials for arm in experiment.trials[i].arms
        }
        evaluated_arms_list.append(params)

        # Inference trace: Not supported for MOO.
        # It's also not supported for multi-fidelity or multi-task
        # problems, because Ax's best-point functionality doesn't know
        # to predict at the target task or fidelity.
        if compute_best_params:
            (best_params,) = method.get_best_parameters(
                experiment=experiment,
                optimization_config=problem.optimization_config,
                n_points=problem.n_best_points,
            )
            best_params_list.append(best_params)


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
        scheduler_logging_level: If >INFO, logs will only appear when unexpected
            things happen. If INFO, logs will update when a trial is completed
            and when an early stopping strategy, if present, decides whether or
            not to continue a trial. If DEBUG, logs additionaly include
            information from a `BackendSimulator`, if present.

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
        force_use_simulated_backend=method.early_stopping_strategy is not None,
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

    # Each of these lists is added to when a trial completes or stops early.
    # Since multiple trials can complete at once, there may be fewer elements in
    # these traces than the number of trials run.
    cost_trace: list[float] = []
    best_params_list: list[Mapping[str, TParamValue]] = []  # For inference trace
    evaluated_arms_list: list[set[Arm]] = []  # For oracle trace
    completed_trial_idcs: set[int] = set()

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
        # These next several lines do the same thing as
        # `scheduler.run_n_trials`, but
        # decrement the timeout with each step, so that the timeout refers to
        # the total time spent in the optimization loop, not time per trial.
        scheduler.poll_and_process_results()
        for _ in scheduler.run_trials_and_yield_results(
            max_trials=problem.num_trials,
            timeout_hours=remaining_hours,
        ):
            _update_benchmark_tracking_vars_in_place(
                experiment=experiment,
                method=method,
                problem=problem,
                cost_trace=cost_trace,
                evaluated_arms_list=evaluated_arms_list,
                best_params_list=best_params_list,
                completed_trial_idcs=completed_trial_idcs,
            )

            if timeout_hours is not None:
                elapsed_hours = (monotonic() - start) / 3600
                remaining_hours = timeout_hours - elapsed_hours
                if remaining_hours <= 0.0:
                    logger.warning("The optimization loop timed out.")
                    break

        scheduler.summarize_final_result()

    sim_runner = runner.simulated_backend_runner
    if sim_runner is not None:
        simulator = sim_runner.simulator
        update_trials_to_use_sim_time_in_place(
            trials=experiment.trials, simulator=simulator
        )

    inference_trace = _get_inference_trace_from_params(
        best_params_list=best_params_list,
        problem=problem,
        n_elements=len(cost_trace),
    )
    oracle_trace = _get_oracle_trace_from_arms(
        evaluated_arms_list=evaluated_arms_list, problem=problem
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
        cost_trace=np.array(cost_trace),
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
    dummy_problem = BenchmarkProblem(
        name="dummy",
        optimization_config=optimization_config,
        search_space=search_space,
        num_trials=5,
        test_function=test_function,
        # Optimal value and baseline value are only used
        # to compute the score_trace, which we don't use here.
        # The order of baseline and optimal value needs to be correct,
        # though, as a ValueError is raised otherwise.
        optimal_value=1.0 if higher_is_better else -1.0,
        baseline_value=0.0,
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


def get_opt_trace_by_steps(experiment: Experiment) -> npt.NDArray:
    """
    Transform an optimization trace in the standard format produced by
    `benchmark_replication`, with one element per trial completion, into a trace
    that is in terms of steps, with one element added each time a step
    completes.

    Args:
        experiment: An experiment produced by `benchmark_replication`; it must
            have `BenchmarkTrialMetadata` (as produced by `BenchmarkRunner`) for
            each trial, and its data must be `MapData`.
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
    data = assert_is_instance(experiment.lookup_data(), MapData)
    map_key = data.map_key_infos[0].key
    map_df = data.map_df
    if len(data.map_key_infos) > 1:
        raise ValueError("Multiple map keys are not supported")

    # Has timestamps; needs to be merged with map_df because it contains
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
    )[["trial_index", map_key, "time"]]

    df = (
        map_df.loc[
            map_df["metric_name"] == objective_name,
            ["trial_index", "arm_name", "mean", map_key],
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

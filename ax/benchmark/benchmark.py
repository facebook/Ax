#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Module for benchmarking Ax algorithms.

Key terms used:

* Trial –– usual Ax `Trial` or `BatchTral`, one execution of a given arm or
  group of arms.
* Replication –– one run of an optimization loop; 1 method + problem combination.
* Test –– multiple replications, ran for statistical significance.
* Full run –– multiple tests: run all methods with all problems.
* Method –– (one of) the algorithm(s) being benchmarked.
* Problem –– a synthetic function, a surrogate surface, or an ML model, on which
  to assess the performance of algorithms.

"""

import time
from types import FunctionType
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from ax.benchmark import utils
from ax.benchmark.benchmark_problem import BenchmarkProblem, SimpleBenchmarkProblem
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.parameter import RangeParameter
from ax.modelbridge.base import gen_arms
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.runners.synthetic import SyntheticRunner
from ax.service.ax_client import AxClient
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none
from ax.utils.measurement.synthetic_functions import SyntheticFunction


logger = get_logger(__name__)


# To bypass catching of exceptions during benchmarking, since all other exceptions
# will be caught and recorded, but will not necessarily terminate the benchmarking
# run.
class NonRetryableBenchmarkingError(ValueError):
    """Error that indicates an issue with the benchmarking setup (e.g. unexpected
    problem setup, a benchmarking function called incorrectly, etc.) –– something
    that prevents the benchmarking suite itself from running, rather than an error
    that occurs during the runs of the benchmarking trials, replications, or tests.
    """

    pass


def benchmark_trial(
    parameterization: Optional[np.ndarray] = None,
    evaluation_function: Optional[Union[SyntheticFunction, FunctionType]] = None,
    experiment: Optional[Experiment] = None,
    trial_index: Optional[int] = None,
) -> Union[Tuple[float, float], Data]:  # Mean and SEM or a Data object.
    """Evaluates one trial from benchmarking replication (an Ax trial or batched
    trial). Evaluation requires either the `parameterization` and `evalution_
    function` parameters or the `experiment` and `trial_index` parameters.

    Note: evaluation function relies on the ordering of items in the
    parameterization nd-array.

    Args:
        parameterization: The parameterization to evaluate.
        evaluation_function: The evaluation function for the benchmark objective.
        experiment: Experiment, for a trial on which to fetch data.
        trial_index: Index of the trial, for which to fetch data.
    """
    use_Service_API = parameterization is not None and evaluation_function is not None
    use_Dev_API = experiment is not None and trial_index is not None
    if not use_Service_API ^ use_Dev_API:
        raise NonRetryableBenchmarkingError(  # TODO[T53975770]: test
            "A parameterization and an evaluation function required for Service-"
            "API-style trial evaluation and an experiment and trial index are "
            "required for Dev API trial evalution via fetching metric data."
        )
    if use_Service_API:
        sem = 0.0 if isinstance(evaluation_function, SyntheticFunction) else None
        # pyre-fixme[7]: Expected `Union[Tuple[float, float], Data]` but got
        #  `Tuple[typing.Any, Optional[float]]`.
        return evaluation_function(parameterization), sem  # pyre-ignore[29]: call err.
    else:
        trial_index = not_none(trial_index)
        return not_none(not_none(experiment).trials.get(trial_index)).fetch_data()


def benchmark_replication(  # One optimization loop.
    problem: BenchmarkProblem,
    method: GenerationStrategy,
    num_trials: int,
    replication_index: Optional[int] = None,
    batch_size: int = 1,
    raise_all_exceptions: bool = False,
    benchmark_trial: FunctionType = benchmark_trial,
    verbose_logging: bool = True,
    # Number of trials that need to fail for a replication to be considered failed.
    failed_trials_tolerated: int = 5,
) -> Experiment:
    """Runs one benchmarking replication (equivalent to one optimization loop).

    Args:
        problem: Problem to benchmark on.
        method: Method to benchmark, represented as generation strategies.
        num_trials: Number of trials in each test experiment.
        batch_size: Batch size for this replication, defaults to 1.
        raise_all_exceptions: If set to True, any encountered exception will be
            raised; alternatively, failure tolerance thresholds are used and a few
            number of trials `failed_trials_tolerated` can fail before a replication
            is considered failed.
        benchmark_trial: Function that runs a single trial. Defaults
            to `benchmark_trial` in this module and must have the same signature.
        verbose_logging: Whether logging level should be set to `INFO`.
        failed_trials_tolerated: How many trials can fail before a replication is
            considered failed and aborted. Defaults to 5.
    """
    trial_exceptions = []
    experiment_name = f"{method.name}_on_{problem.name}"
    if replication_index is not None:
        experiment_name += f"__v{replication_index}"
    # Make sure the generation strategy starts from the beginning.
    method = method.clone_reset()

    # Choose whether to run replication via Service or Developer API, based on
    # whether the problem was set up using Ax classes like `SearchSpace` and
    # `OptimizationConfig` or using "RESTful" Service API-like constructs like
    # dict parameter representations and `SyntheticFunction`-s or custom callables
    # for evaluation function.
    replication_runner = (
        _benchmark_replication_Service_API
        if isinstance(problem, SimpleBenchmarkProblem)
        else _benchmark_replication_Dev_API
    )
    experiment, exceptions = replication_runner(
        problem=problem,  # pyre-ignore[6]
        method=method,
        num_trials=num_trials,
        experiment_name=experiment_name,
        batch_size=batch_size,
        raise_all_exceptions=raise_all_exceptions,
        benchmark_trial=benchmark_trial,
        verbose_logging=verbose_logging,
        failed_trials_tolerated=failed_trials_tolerated,
    )

    trial_exceptions.extend(exceptions)
    return experiment


def benchmark_test(  # One test, multiple replications.
    problem: BenchmarkProblem,
    method: GenerationStrategy,
    num_trials: int,
    num_replications: int = 20,
    batch_size: int = 1,
    raise_all_exceptions: bool = False,
    benchmark_replication: FunctionType = benchmark_replication,
    benchmark_trial: FunctionType = benchmark_trial,
    verbose_logging: bool = True,
    # Number of trials that need to fail for a replication to be considered failed.
    failed_trials_tolerated: int = 5,
    # Number of replications that need to fail for a test to be considered failed.
    failed_replications_tolerated: int = 3,
) -> List[Experiment]:
    """Runs one benchmarking test (equivalent to one problem-method combination),
    translates into `num_replication` replications, ran for statistical
    significance of the results.

    Args:
        problem: Problem to benchmark on.
        method: Method to benchmark, represented as generation strategies.
        num_replications: Number of times to run each test (each problem-method
            combination), for an aggregated result.
        num_trials: Number of trials in each test experiment, defaults to 20.
        batch_size: Batch size for this test, defaults to 1.
        raise_all_exceptions: If set to True, any encountered exception will be
            raised; alternatively, failure tolerance thresholds are used and a few
            number of trials `failed_trials_tolerated` can fail before a replication
            is considered failed, as well some replications
            `failed_replications_tolerated` can fail before a benchmarking test
            is considered failed.
        benchmark_replication: Function that runs a single benchmarking replication.
            Defaults to `benchmark_replication` in this module and must have the
            same signature.
        benchmark_trial: Function that runs a single trial. Defaults
            to `benchmark_trial` in this module and must have the same signature.
        verbose_logging: Whether logging level should be set to `INFO`.
        failed_trials_tolerated: How many trials can fail before a replication is
            considered failed and aborted. Defaults to 5.
        failed_replications_tolerated: How many replications can fail before a
            test is considered failed and aborted. Defaults to 3.
    """
    replication_exceptions = []
    test_replications = []
    for replication_idx in range(num_replications):
        try:
            test_replications.append(
                benchmark_replication(
                    problem=problem,
                    method=method,
                    replication_index=replication_idx,
                    num_trials=num_trials,
                    batch_size=batch_size,
                    raise_all_exceptions=raise_all_exceptions,
                    verbose_logging=verbose_logging,
                    failed_trials_tolerated=failed_trials_tolerated,
                )
            )
        except Exception as err:
            if raise_all_exceptions:
                raise
            replication_exceptions.append(err)  # TODO[T53975770]: test
        if len(replication_exceptions) > failed_replications_tolerated:
            raise RuntimeError(  # TODO[T53975770]: test
                f"More than {failed_replications_tolerated} failed for "
                "{method.name}_on_{problem.name}."
            )
    return test_replications


def full_benchmark_run(  # Full run, multiple tests.
    problems: Optional[Union[List[BenchmarkProblem], List[str]]] = None,
    methods: Optional[Union[List[GenerationStrategy], List[str]]] = None,
    num_trials: Union[int, List[List[int]]] = 20,
    num_replications: int = 20,
    batch_size: Union[int, List[List[int]]] = 1,
    raise_all_exceptions: bool = False,
    benchmark_test: FunctionType = benchmark_test,
    benchmark_replication: FunctionType = benchmark_replication,
    benchmark_trial: FunctionType = benchmark_trial,
    verbose_logging: bool = True,
    # Number of trials that need to fail for a replication to be considered failed.
    failed_trials_tolerated: int = 5,
    # Number of replications that need to fail for a test to be considered failed.
    failed_replications_tolerated: int = 3,
) -> Dict[str, Dict[str, List[Experiment]]]:
    """Full run of the benchmarking suite. To make benchmarking distrubuted at
    a level of a test, a replication, or a trial (or any combination of those),
    by passing in a wrapped (in some scheduling logic) version of a corresponding
    function from this module.

    Args:
        problems: Problems to benchmark on, represented as BenchmarkProblem-s
            or string keys (must be in standard BOProblems). Defaults to all
            standard BOProblems.
        methods: Methods to benchmark, represented as generation strategies or
            or string keys (must be in standard BOMethods). Defaults to all
            standard BOMethods.
        num_replications: Number of times to run each test (each problem-method
            combination), for an aggregated result.
        num_trials: Number of trials in each test experiment.
        raise_all_exceptions: If set to True, any encountered exception will be
            raised; alternatively, failure tolerance thresholds are used and a few
            number of trials `failed_trials_tolerated` can fail before a replication
            is considered failed, as well some replications
            `failed_replications_tolerated` can fail before a benchmarking test
            is considered failed.
        benchmark_test: Function that runs a single benchmarking test. Defaults
            to `benchmark_test` in this module and must have the same signature.
        benchmark_replication: Function that runs a single benchmarking replication.
            Defaults to `benchmark_replication` in this module and must have the
            same signature.
        benchmark_trial: Function that runs a single trial. Defaults
            to `benchmark_trial` in this module and must have the same signature.
        verbose_logging: Whether logging level should be set to `INFO`.
        failed_trials_tolerated: How many trials can fail before a replication is
            considered failed and aborted. Defaults to 5.
        failed_replications_tolerated: How many replications can fail before a
            test is considered failed and aborted. Defaults to 3.
    """
    exceptions = []
    tests: Dict[str, Dict[str, List[Experiment]]] = {}
    problems, methods = utils.get_problems_and_methods(
        problems=problems, methods=methods
    )
    for problem_idx, problem in enumerate(problems):
        tests[problem.name] = {}
        for method_idx, method in enumerate(methods):
            tests[problem.name][method.name] = []
            try:
                tests[problem.name][method.name] = benchmark_test(
                    problem=problem,
                    method=method,
                    num_replications=num_replications,
                    # For arguments passed as either numbers, or matrices,
                    # xtract corresponding values for the given combination.
                    num_trials=utils.get_corresponding(
                        num_trials, problem_idx, method_idx
                    ),
                    batch_size=utils.get_corresponding(
                        batch_size, problem_idx, method_idx
                    ),
                    benchmark_replication=benchmark_replication,
                    benchmark_trial=benchmark_trial,
                    raise_all_exceptions=raise_all_exceptions,
                    verbose_logging=verbose_logging,
                    failed_replications_tolerated=failed_replications_tolerated,
                    failed_trials_tolerated=failed_trials_tolerated,
                )
            except Exception as err:
                if raise_all_exceptions:
                    raise
                exceptions.append(err)  # TODO[T53975770]: test
    logger.info(f"Obtained benchmarking test experiments: {tests}")
    return tests


def _benchmark_replication_Service_API(
    problem: SimpleBenchmarkProblem,
    method: GenerationStrategy,
    num_trials: int,
    experiment_name: str,
    batch_size: int = 1,
    raise_all_exceptions: bool = False,
    benchmark_trial: FunctionType = benchmark_trial,
    verbose_logging: bool = True,
    # Number of trials that need to fail for a replication to be considered failed.
    failed_trials_tolerated: int = 5,
) -> Tuple[Experiment, List[Exception]]:
    """Run a benchmark replication via the Service API because the problem was
    set up in a simplified way, without the use of Ax classes like `OptimizationConfig`
    or `SearchSpace`.
    """
    exceptions = []
    if batch_size == 1:
        ax_client = AxClient(
            generation_strategy=method, verbose_logging=verbose_logging
        )
    else:  # pragma: no cover, TODO[T53975770]
        assert batch_size > 1, "Batch size of 1 or greater is expected."
        raise NotImplementedError(
            "Batched benchmarking on `SimpleBenchmarkProblem`-s not yet implemented."
        )
    ax_client.create_experiment(
        name=experiment_name,
        parameters=problem.domain_as_ax_client_parameters(),
        minimize=problem.minimize,
        objective_name=problem.name,
    )
    parameter_names = list(ax_client.experiment.search_space.parameters.keys())
    assert num_trials > 0
    for _ in range(num_trials):
        parameterization, idx = ax_client.get_next_trial()
        param_values = np.array([parameterization.get(x) for x in parameter_names])
        try:
            mean, sem = benchmark_trial(
                parameterization=param_values, evaluation_function=problem.f
            )
            # If problem indicates a noise level and is using a synthetic callable,
            # add normal noise to the measurement of the mean.
            if problem.uses_synthetic_function and problem.noise_sd != 0.0:
                noise = np.random.randn() * problem.noise_sd
                sem = (sem or 0.0) + problem.noise_sd
                logger.info(
                    f"Adding noise of {noise} to the measurement mean ({mean})."
                    f"Problem noise SD setting: {problem.noise_sd}."
                )
                mean = mean + noise
            ax_client.complete_trial(trial_index=idx, raw_data=(mean, sem))
        except Exception as err:  # TODO[T53975770]: test
            if raise_all_exceptions:
                raise
            exceptions.append(err)
        if len(exceptions) > failed_trials_tolerated:
            raise RuntimeError(  # TODO[T53975770]: test
                f"More than {failed_trials_tolerated} failed for {experiment_name}."
            )
    return ax_client.experiment, exceptions


def _benchmark_replication_Dev_API(
    problem: BenchmarkProblem,
    method: GenerationStrategy,
    num_trials: int,
    experiment_name: str,
    batch_size: int = 1,
    raise_all_exceptions: bool = False,
    benchmark_trial: FunctionType = benchmark_trial,
    verbose_logging: bool = True,
    # Number of trials that need to fail for a replication to be considered failed.
    failed_trials_tolerated: int = 5,
) -> Tuple[Experiment, List[Exception]]:
    """Run a benchmark replication via the Developer API because the problem was
    set up with Ax classes (likely to allow for additional complexity like
    adding constraints or non-range parameters).
    """
    exceptions = []
    experiment = Experiment(
        name=experiment_name,
        search_space=problem.search_space,
        optimization_config=problem.optimization_config,
        runner=SyntheticRunner(),
    )
    for _ in range(num_trials):
        try:
            gr = method.gen(experiment=experiment, n=batch_size)
            if batch_size == 1:
                experiment.new_trial(generator_run=gr).run()
            else:
                assert batch_size > 1
                experiment.new_batch_trial(generator_run=gr).run()
        except Exception as err:  # TODO[T53975770]: test
            if raise_all_exceptions:
                raise
            exceptions.append(err)
        if len(exceptions) > failed_trials_tolerated:
            raise RuntimeError(  # TODO[T53975770]: test
                f"More than {failed_trials_tolerated} failed for {experiment_name}."
            )
    return experiment, exceptions


def benchmark_minimize_callable(
    problem: BenchmarkProblem,
    num_trials: int,
    method_name: str,
    replication_index: Optional[int] = None,
) -> Tuple[Experiment, Callable[[List[float]], float]]:
    """
    An interface for evaluating external methods on Ax benchmark problems. The
    arms run and performance will be tracked by Ax, so the external method can
    be evaluated alongside Ax methods.

    It is designed around methods that implement an interface like
    scipy.optimize.minimize. This function will return a callable evaluation
    function that takes in an array of parameter values and returns a float
    objective value. The evaluation function should always be minimized: if the
    benchmark problem is a maximization problem, then the value returned by
    the evaluation function will be negated so it can be used directly by
    methods that minimize. This callable can be given to an external
    minimization function, and Ax will track all of the calls made to it and
    the arms that were evaluated.

    This will also return an Experiment object that will track the arms
    evaluated by the external method in the same way as done for Ax
    internal benchmarks. This function should thus be used for each benchmark
    replication.

    Args:
        problem: The Ax benchmark problem to be used to construct the
            evalutaion function.
        num_trials: The maximum number of trials for a benchmark run.
        method_name: Name of the method being tested.
        replication_index: Replicate number, if multiple replicates are being
            run.
    """
    # Some validation
    if isinstance(problem, SimpleBenchmarkProblem):
        raise NonRetryableBenchmarkingError("`SimpleBenchmarkProblem` not supported.")
    if not all(
        isinstance(p, RangeParameter) for p in problem.search_space.parameters.values()
    ):
        raise NonRetryableBenchmarkingError("Only continuous search spaces supported.")
    if any(
        p.log_scale for p in problem.search_space.parameters.values()  # pyre-ignore
    ):
        raise NonRetryableBenchmarkingError("Log-scale parameters not supported.")

    # Create Ax experiment
    experiment_name = f"{method_name}_on_{problem.name}"
    if replication_index is not None:
        experiment_name += f"__v{replication_index}"
    experiment = Experiment(
        name=experiment_name,
        search_space=problem.search_space,
        optimization_config=problem.optimization_config,
        runner=SyntheticRunner(),
    )
    max_trials = num_trials  # to be used below

    # Construct the evaluation function
    def evaluation_function(x: List[float]) -> float:
        # Check if we have exhuasted the evaluation budget
        if len(experiment.trials) >= max_trials:
            raise ValueError(f"Evaluation budget ({max_trials} trials) exhuasted.")

        # Create an ObservationFeatures
        param_dict = {
            pname: x[i]
            for i, pname in enumerate(problem.search_space.parameters.keys())
        }
        obsf = ObservationFeatures(parameters=param_dict)  # pyre-ignore
        # Get the time since last call
        num_trials = len(experiment.trials)
        if num_trials == 0:
            gen_time = None
        else:
            previous_ts = experiment.trials[num_trials - 1].time_created.timestamp()
            gen_time = time.time() - previous_ts
        # Create a GR
        arms, candidate_metadata_by_arm_signature = gen_arms(
            observation_features=[obsf], arms_by_signature=experiment.arms_by_signature
        )
        gr = GeneratorRun(
            arms=arms,
            gen_time=gen_time,
            candidate_metadata_by_arm_signature=candidate_metadata_by_arm_signature,
        )
        # Add it as a trial
        trial = experiment.new_trial().add_generator_run(gr).run()
        # Evaluate function
        df = trial.fetch_data().df
        if len(df) > 1:
            raise Exception("Does not support multiple outcomes")  # pragma: no cover
        obj = float(df["mean"].values[0])
        if not problem.optimization_config.objective.minimize:
            obj = -obj
        return obj

    return experiment, evaluation_function

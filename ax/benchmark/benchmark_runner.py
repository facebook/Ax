#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.optimization_config import OptimizationConfig
from ax.core.trial import Trial
from ax.core.types import ComparisonOp
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.logger import get_logger


logger: logging.Logger = get_logger(__name__)

ALLOWED_RUN_RETRIES = 5
PROBLEM_METHOD_DELIMETER = "_on_"
RUN_DELIMETER = "_run_"


class BenchmarkResult(NamedTuple):
    # {method_name -> [[best objective per trial] per benchmark run]}
    objective_at_true_best: Dict[str, np.ndarray]
    # {method_name -> trials where generation strategy changed}
    generator_changes: Dict[str, Optional[List[int]]]
    optimum: float
    # {method_name -> [total fit time per run]}
    fit_times: Dict[str, List[float]]
    # {method_name -> [total gen time per run]}
    gen_times: Dict[str, List[float]]


class BenchmarkSetup(Experiment):
    """An extension of `Experiment`, specific to benchmarking. Contains
    additional data, such as the benchmarking problem, iterations to run per
    benchmarking method and problem combination, etc.

    Args:
        problem: description of the benchmarking problem for
            this setup
        total_iterations: how many optimization iterations to run
        batch_size: if this benchmark requires batch trials,
            batch size for those. Defaults to None
    """

    problem: BenchmarkProblem
    total_iterations: int
    batch_size: int

    def __init__(
        self, problem: BenchmarkProblem, total_iterations: int = 20, batch_size: int = 1
    ) -> None:
        super().__init__(
            name=problem.name,
            search_space=problem.search_space,
            runner=SyntheticRunner(),
            optimization_config=problem.optimization_config,
        )
        self.problem = problem
        self.total_iterations = total_iterations
        self.batch_size = batch_size

    def clone_reset(self) -> "BenchmarkSetup":
        """Create a clean copy of this benchmarking setup, with no run data
        attached to it."""
        return BenchmarkSetup(self.problem, self.total_iterations, self.batch_size)


class BenchmarkRunner:
    """Runner that keeps track of benchmark runs and failures encountered
    during benchmarking.
    """

    _failed_runs: List[Tuple[str, str]]
    _runs: Dict[Tuple[str, str, int], BenchmarkSetup]
    _error_messages: List[str]
    _generator_changes: Dict[Tuple[str, str, int], Optional[List[int]]]

    def __init__(self) -> None:
        self._runs = {}
        self._failed_runs = []
        self._error_messages = []
        self._generator_changes = {}

    @abstractmethod
    def run_benchmark_run(
        self, setup: BenchmarkSetup, generation_strategy: GenerationStrategy
    ) -> BenchmarkSetup:
        """Run a single full benchmark run of the given problem and method
        combination.
        """
        pass  # pragma: no cover

    def run_benchmark_test(
        self,
        setup: BenchmarkSetup,
        generation_strategy: GenerationStrategy,
        num_runs: int = 20,
        raise_all_errors: bool = False,
    ) -> Dict[Tuple[str, str, int], BenchmarkSetup]:
        """Run full benchmark test for the given method and problem combination.
        A benchmark test consists of repeated full benchmark runs.

        Args:
            setup: setup, runs on which to execute; includes
                a benchmarking problem, total number of iterations, etc.
            generation strategy: generation strategy that defines which
                generation methods should be used in this benchmarking test
            num_runs: how many benchmark runs of given problem and method
                combination to run with the given setup for one benchmark test
        """
        num_failures = 0
        benchmark_runs: Dict[Tuple[str, str, int], BenchmarkSetup] = {}
        logger.info(f"Testing {generation_strategy.name} on {setup.name}:")
        for run_idx in range(num_runs):
            logger.info(f"Run {run_idx}")
            run_key = (setup.name, generation_strategy.name, run_idx)
            # If this run has already been executed, log and skip it.
            if run_key in self._runs:
                self._error_messages.append(  # pragma: no cover
                    f"Run {run_idx} of {generation_strategy.name} on {setup.name} "
                    "has already been executed in this benchmarking suite."
                    "Check that this method + problem combination is not "
                    "included in the benchmarking suite twice. Only the first "
                    "run will be recorded."
                )
                continue

            # When number of failures in this test exceeds the allowed max,
            # we consider the whole run failed.
            while num_failures < ALLOWED_RUN_RETRIES:
                try:
                    benchmark_runs[run_key] = self.run_benchmark_run(
                        setup.clone_reset(), generation_strategy.clone_reset()
                    )
                    self._generator_changes[
                        run_key
                    ] = generation_strategy.generator_changes
                    break
                except Exception as err:  # pragma: no cover
                    if raise_all_errors:
                        raise err
                    logger.exception(err)
                    num_failures += 1
                    self._error_messages.append(f"Error in {run_key}: {err}")

        if num_failures >= 5:
            self._error_messages.append(
                f"Considering {generation_strategy.name} on {setup.name} failed"
            )
            self._failed_runs.append((setup.name, generation_strategy.name))
        else:
            self._runs.update(benchmark_runs)
        return self._runs

    def aggregate_results(self) -> Dict[str, BenchmarkResult]:
        """Pull results from each of the runs (BenchmarkSetups aka Experiments)
        and aggregate them into a BenchmarkResult for each problem.
        """
        n_iters: Dict[Tuple[str, str], int] = {}
        optima: Dict[str, float] = {}
        # Results will be put in nested dictionaries problem -> method -> results
        objective_at_true_best: Dict[str, Dict[str, List[np.ndarray]]] = {}
        fit_times: Dict[str, Dict[str, List[float]]] = {}
        gen_times: Dict[str, Dict[str, List[float]]] = {}
        generator_changes: Dict[str, Dict[str, Optional[List[int]]]] = {}
        for (p, m, r), setup in self._runs.items():
            for res_dict in [
                objective_at_true_best,
                fit_times,
                gen_times,
                generator_changes,
            ]:
                if p not in res_dict:
                    res_dict[p] = defaultdict(list)
            optima[p] = setup.problem.fbest
            generator_changes[p][m] = self._generator_changes[(p, m, r)]
            # Extract iterations for this pmr
            names = []
            for trial in setup.trials.values():
                for i, arm in enumerate(trial.arms):
                    if isinstance(trial, BatchTrial):
                        reps = int(trial.weights[i])
                    else:
                        reps = 1
                    names.extend([arm.name] * reps)
            # Make sure every run has the same number of iterations, so we can safely
            # stack them in a matrix.
            if (p, m) in n_iters:
                if len(names) != n_iters[(p, m)]:
                    raise ValueError(  # pragma: no cover
                        f"Expected {n_iters[(p, m)]} iterations, got {len(names)}"
                    )
            else:
                n_iters[(p, m)] = len(names)
            # Get true values for every outcome for each iteration
            iters_df = pd.DataFrame({"arm_name": names})
            data_df = setup.fetch_data(noisy=False).df
            metrics = data_df["metric_name"].unique()
            true_values = {}
            for metric in metrics:
                df_m = data_df[data_df["metric_name"] == metric]
                # Get one row per arm
                df_m = df_m.groupby("arm_name").first().reset_index()
                df_b = pd.merge(iters_df, df_m, how="left", on="arm_name")
                true_values[metric] = df_b["mean"].values
            # Compute the things we care about
            # 1. True best objective value.
            objective_at_true_best[p][m].append(
                true_best_objective(
                    optimization_config=setup.problem.optimization_config,
                    true_values=true_values,
                )
            )
            # 2. Time
            fit_time, gen_time = get_model_times(setup)
            fit_times[p][m].append(fit_time)
            gen_times[p][m].append(gen_time)
            # 3. True objective value of model-predicted best (TODO)
            # 4. True feasiblity of model-predicted best (TODO)
            # 5. Model prediction MSE for each gen run (TODO)

        # Combine methods for each problem for the BenchmarkResult
        res: Dict[str, BenchmarkResult] = {}
        for p in objective_at_true_best:
            res[p] = BenchmarkResult(
                objective_at_true_best={
                    m: np.array(v) for m, v in objective_at_true_best[p].items()
                },
                generator_changes=generator_changes[p],
                optimum=optima[p],
                fit_times=fit_times[p],
                gen_times=gen_times[p],
            )
        return res

    @property
    def errors(self) -> List[str]:
        """Messages from errors encoutered while running benchmark test."""
        return self._error_messages


class BanditBenchmarkRunner(BenchmarkRunner):
    def run_benchmark_run(
        self, setup: BenchmarkSetup, generation_strategy: GenerationStrategy
    ) -> BenchmarkSetup:
        pass  # pragma: no cover  TODO[drfreund]


class BOBenchmarkRunner(BenchmarkRunner):
    def run_benchmark_run(
        self, setup: BenchmarkSetup, generation_strategy: GenerationStrategy
    ) -> BenchmarkSetup:
        remaining_iterations = setup.total_iterations
        updated_trials = []
        while remaining_iterations > 0:
            num_suggestions = min(remaining_iterations, setup.batch_size)
            generator_run = generation_strategy.gen(
                experiment=setup,
                new_data=Data.from_multiple_data(
                    [setup._fetch_trial_data(idx) for idx in updated_trials]
                ),
                n=setup.batch_size,
            )
            updated_trials = []
            if setup.batch_size > 1:  # pragma: no cover
                trial = setup.new_batch_trial().add_generator_run(generator_run).run()
            else:
                trial = setup.new_trial(generator_run=generator_run).run()
            updated_trials.append(trial.index)
            remaining_iterations -= num_suggestions
        return setup


def true_best_objective(
    optimization_config: OptimizationConfig, true_values: Dict[str, np.ndarray]
) -> np.ndarray:
    """Compute the true best objective value found by each iteration.

    Args:
        optimization_config: Optimization config
        true_values: Dictionary from metric name to array of value at each
            iteration.

    Returns: Array of cumulative best feasible value.
    """
    # Get objective at each iteration
    objective = optimization_config.objective
    f = true_values[objective.metric.name]
    # Set infeasible points to have inf bad values
    if objective.minimize:
        infeas_val = np.Inf
    else:
        infeas_val = -np.Inf
    for oc in optimization_config.outcome_constraints:
        if oc.relative:
            raise ValueError(
                "Benchmark aggregation does not support relative constraints"
            )
        g = true_values[oc.metric.name]
        if oc.op == ComparisonOp.LEQ:
            feas = g <= oc.bound
        else:
            feas = g >= oc.bound
        f[~feas] = infeas_val
    # Get cumulative best
    if objective.minimize:
        return np.minimum.accumulate(f)
    else:
        return np.maximum.accumulate(f)


def get_model_times(setup: BenchmarkSetup) -> Tuple[float, float]:
    fit_time = 0.0
    gen_time = 0.0
    for trial in setup.trials.values():
        if isinstance(trial, BatchTrial):  # pragma: no cover
            gr = trial._generator_run_structs[0].generator_run
        elif isinstance(trial, Trial):
            gr = trial.generator_run
        else:
            raise ValueError("Unexpected trial type")  # pragma: no cover
        if gr is None:  # for typing
            raise ValueError(
                "Unexpected trial with no generator run"
            )  # pragma: no cover
        if gr.fit_time is not None:
            fit_time += gr.fit_time
        if gr.gen_time is not None:
            gen_time += gr.gen_time
    return fit_time, gen_time

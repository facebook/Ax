#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from ax.benchmark.benchmark_problem import BenchmarkProblem, SimpleBenchmarkProblem
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.trial import Trial
from ax.core.utils import best_feasible_objective, feasible_hypervolume, get_model_times
from ax.plot.base import AxPlotConfig
from ax.plot.pareto_frontier import plot_multiple_pareto_frontiers
from ax.plot.pareto_utils import (
    get_observed_pareto_frontiers,
    ParetoFrontierResults,
)
from ax.plot.render import plot_config_to_html
from ax.plot.trace import (
    optimization_times,
    optimization_trace_all_methods,
    optimization_trace_single_method,
)
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.report.render import h2_html, h3_html, p_html, render_report_elements

logger: logging.Logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    # {method_name -> [[best objective per trial] per benchmark run]}
    true_performance: Dict[str, np.ndarray]
    # {method_name -> [total fit time per run]}
    fit_times: Dict[str, List[float]]
    # {method_name -> [total gen time per run]}
    gen_times: Dict[str, List[float]]
    # {method_name -> trials where generation strategy changed}
    optimum: Optional[float] = None
    model_transitions: Optional[Dict[str, Optional[List[int]]]] = None
    is_multi_objective: bool = False
    pareto_frontiers: Optional[Dict[str, ParetoFrontierResults]] = None


def aggregate_problem_results(
    runs: Dict[str, List[Experiment]],
    problem: BenchmarkProblem,
    # Model transitions, can be obtained as `generation_strategy.model_transitions`
    model_transitions: Optional[Dict[str, List[int]]] = None,
) -> BenchmarkResult:
    # Results will be put in {method -> results} dictionaries.
    true_performances: Dict[str, List[np.ndarray]] = {}
    fit_times: Dict[str, List[float]] = {}
    gen_times: Dict[str, List[float]] = {}
    exp = list(runs.values())[0][0]
    is_moo = isinstance(exp.optimization_config, MultiObjectiveOptimizationConfig)
    plot_pfs = is_moo and len(not_none(exp.optimization_config).objective.metrics) == 2
    pareto_frontiers = {} if plot_pfs else None
    for method, experiments in runs.items():
        true_performances[method] = []
        fit_times[method] = []
        gen_times[method] = []
        for experiment in experiments:
            assert (
                problem.name in experiment.name
            ), "Problem and experiment name do not match."
            fit_time, gen_time = get_model_times(experiment=experiment)
            true_performance = extract_optimization_trace(
                experiment=experiment, problem=problem
            )

            # Compute the things we care about
            # 1. True best objective value.
            true_performances[method].append(true_performance)
            # 2. Time
            fit_times[method].append(fit_time)
            gen_times[method].append(gen_time)
            # TODO: If `evaluate_suggested` is True on the problem
            # 3. True obj. value of model-predicted best
            # 4. True feasiblity of model-predicted best
            # 5. Model prediction MSE for each gen run
        # only include pareto frontier for one experiment per method
        if plot_pfs:
            # pyre-ignore [16]
            pareto_frontiers[method] = get_observed_pareto_frontiers(
                experiment=experiment,
                # pyre-ignore [6]
                data=experiment.fetch_data(),
            )[0]

        # TODO: remove rows from <values>[method] of length different
        # from the length of other rows, log warning when removing
    return BenchmarkResult(
        true_performance={m: np.array(v) for m, v in true_performances.items()},
        # pyre-fixme[6]: [6]: Expected `Optional[Dict[str, Optional[List[int]]]]`
        # but got `Optional[Dict[str, List[int]]]`
        model_transitions=model_transitions,
        optimum=problem.optimal_value,
        fit_times=fit_times,
        gen_times=gen_times,
        is_multi_objective=is_moo,
        pareto_frontiers=pareto_frontiers,
    )


def make_plots(
    benchmark_result: BenchmarkResult, problem_name: str, include_individual: bool
) -> List[AxPlotConfig]:
    plots: List[AxPlotConfig] = []
    # Plot objective at true best
    ylabel = (
        "Feasible Hypervolume"
        if benchmark_result.is_multi_objective
        else "Objective at best-feasible point observed so far"
    )
    plots.append(
        optimization_trace_all_methods(
            y_dict=benchmark_result.true_performance,
            optimum=benchmark_result.optimum,
            title=f"{problem_name}: Optimization Performance",
            ylabel=ylabel,
        )
    )
    if include_individual:
        # Plot individual plots of a single method on a single problem.
        for m, y in benchmark_result.true_performance.items():
            plots.append(
                optimization_trace_single_method(
                    y=y,
                    optimum=benchmark_result.optimum,
                    # model_transitions=benchmark_result.model_transitions[m],
                    title=f"{problem_name}, {m}: cumulative best objective",
                    ylabel=ylabel,
                )
            )
    # Plot time
    plots.append(
        optimization_times(
            fit_times=benchmark_result.fit_times,
            gen_times=benchmark_result.gen_times,
            title=f"{problem_name}: cumulative optimization times",
        )
    )
    if benchmark_result.pareto_frontiers is not None:
        plots.append(
            plot_multiple_pareto_frontiers(
                frontiers=not_none(benchmark_result.pareto_frontiers),
                CI_level=0.0,
            )
        )
    return plots


def generate_report(
    benchmark_results: Dict[str, BenchmarkResult],
    errors_encountered: Optional[List[str]] = None,
    include_individual_method_plots: bool = False,
    notebook_env: bool = False,
) -> str:
    html_elements = [h2_html("Bayesian Optimization benchmarking suite report")]
    for p, benchmark_result in benchmark_results.items():
        html_elements.append(h3_html(f"{p}:"))
        plots = make_plots(
            benchmark_result,
            problem_name=p,
            include_individual=include_individual_method_plots,
        )
        html_elements.extend(plot_config_to_html(plt) for plt in plots)
    if errors_encountered:
        html_elements.append(h3_html("Errors encountered:"))
        html_elements.extend(p_html(err) for err in errors_encountered)
    else:
        html_elements.append(h3_html("No errors encountered!"))
    # Experiment name is used in header, which is disabled in this case.
    return render_report_elements(
        experiment_name="",
        html_elements=html_elements,
        header=False,
        notebook_env=notebook_env,
    )


def extract_optimization_trace(  # pragma: no cover
    experiment: Experiment, problem: BenchmarkProblem
) -> np.ndarray:
    """Extract outcomes of an experiment: best cumulative objective as numpy ND-
    array, and total model-fitting time and candidate generation time as floats.
    """
    # Get true values by evaluting the synthetic function noiselessly
    if isinstance(problem, SimpleBenchmarkProblem) and problem.uses_synthetic_function:
        return _extract_optimization_trace_from_synthetic_function(
            experiment=experiment, problem=problem
        )

    # True values are not available, so just use the known values
    elif isinstance(problem, SimpleBenchmarkProblem):
        logger.info(
            "Cannot obtain true best objectives since an ad-hoc function was used."
        )
        # pyre-fixme[16]: `Optional` has no attribute `outcome_constraints`.
        assert len(experiment.optimization_config.outcome_constraints) == 0
        values = np.array(
            [checked_cast(Trial, trial).objective_mean for trial in experiment.trials]
        )
        return best_feasible_objective(
            # pyre-fixme[6]: Expected `OptimizationConfig` for 1st param but got
            #  `Optional[ax.core.optimization_config.OptimizationConfig]`.
            optimization_config=experiment.optimization_config,
            values={problem.name: values},
        )

    else:  # Get true values for every outcome for each iteration
        return _extract_optimization_trace_from_metrics(experiment=experiment)


def _extract_optimization_trace_from_metrics(experiment: Experiment) -> np.ndarray:
    names = []
    for trial in experiment.trials.values():
        for i, arm in enumerate(trial.arms):
            reps = int(trial.weights[i]) if isinstance(trial, BatchTrial) else 1
            names.extend([arm.name] * reps)
    iters_df = pd.DataFrame({"arm_name": names})
    data_df = experiment.fetch_data(noisy=False).df
    metrics = data_df["metric_name"].unique()
    true_values = {}
    for metric in metrics:
        df_m = data_df[data_df["metric_name"] == metric]
        # Get one row per arm
        df_m = df_m.groupby("arm_name").first().reset_index()
        df_b = pd.merge(iters_df, df_m, how="left", on="arm_name")
        true_values[metric] = df_b["mean"].values
    if isinstance(experiment.optimization_config, MultiObjectiveOptimizationConfig):
        return feasible_hypervolume(
            # pyre-fixme[6]: Expected `OptimizationConfig` for 1st param but got
            #  `Optional[ax.core.optimization_config.OptimizationConfig]`.
            optimization_config=experiment.optimization_config,
            values=true_values,
        )
    return best_feasible_objective(
        # pyre-fixme[6]: Expected `OptimizationConfig` for 1st param but got
        #  `Optional[ax.core.optimization_config.OptimizationConfig]`.
        optimization_config=experiment.optimization_config,
        values=true_values,
    )


def _extract_optimization_trace_from_synthetic_function(
    experiment: Experiment, problem: SimpleBenchmarkProblem
) -> np.ndarray:
    if any(isinstance(trial, BatchTrial) for trial in experiment.trials.values()):
        raise NotImplementedError("Batched trials are not yet supported.")
    true_values = []
    for trial in experiment.trials.values():
        parameters = not_none(checked_cast(Trial, trial).arm).parameters
        # Expecting numerical parameters only.
        value = problem.f(*[float(x) for x in parameters.values()])  # pyre-ignore[6]
        true_values.append(value)
    return best_feasible_objective(
        # pyre-fixme[6]: Expected `OptimizationConfig` for 1st param but got
        #  `Optional[ax.core.optimization_config.OptimizationConfig]`.
        optimization_config=experiment.optimization_config,
        values={problem.name: true_values},
    )

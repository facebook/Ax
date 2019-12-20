#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd
from ax.benchmark.benchmark_problem import BenchmarkProblem, SimpleBenchmarkProblem
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.trial import Trial
from ax.core.utils import best_feasible_objective, get_model_times
from ax.plot.base import AxPlotConfig
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


class BenchmarkResult(NamedTuple):
    # {method_name -> [[best objective per trial] per benchmark run]}
    objective_at_true_best: Dict[str, np.ndarray]
    # {method_name -> [total fit time per run]}
    fit_times: Dict[str, List[float]]
    # {method_name -> [total gen time per run]}
    gen_times: Dict[str, List[float]]
    # {method_name -> trials where generation strategy changed}
    optimum: Optional[float] = None
    model_transitions: Optional[Dict[str, Optional[List[int]]]] = None


def aggregate_problem_results(
    runs: Dict[str, List[Experiment]],
    problem: BenchmarkProblem,
    # Model transitions, can be obtained as `generation_strategy.model_transitions`
    model_transitions: Optional[Dict[str, List[int]]] = None,
) -> BenchmarkResult:
    # Results will be put in {method -> results} dictionaries.
    objective_at_true_best: Dict[str, List[np.ndarray]] = {}
    fit_times: Dict[str, List[float]] = {}
    gen_times: Dict[str, List[float]] = {}

    for method, experiments in runs.items():
        objective_at_true_best[method] = []
        fit_times[method] = []
        gen_times[method] = []
        for experiment in experiments:
            assert (
                problem.name in experiment.name
            ), "Problem and experiment name do not match."
            fit_time, gen_time = get_model_times(experiment=experiment)
            true_best_objective = extract_optimization_trace(
                experiment=experiment, problem=problem
            )

            # Compute the things we care about
            # 1. True best objective value.
            objective_at_true_best[method].append(true_best_objective)
            # 2. Time
            fit_times[method].append(fit_time)
            gen_times[method].append(gen_time)
            # TODO: If `evaluate_suggested` is True on the problem
            # 3. True obj. value of model-predicted best
            # 4. True feasiblity of model-predicted best
            # 5. Model prediction MSE for each gen run

        # TODO: remove rows from <values>[method] of length different
        # from the length of other rows, log warning when removing
    return BenchmarkResult(
        objective_at_true_best={
            m: np.array(v) for m, v in objective_at_true_best.items()
        },
        # pyre-fixme[6]: [6]: Expected `Optional[Dict[str, Optional[List[int]]]]`
        # but got `Optional[Dict[str, List[int]]]`
        model_transitions=model_transitions,
        optimum=problem.optimal_value,
        fit_times=fit_times,
        gen_times=gen_times,
    )


def make_plots(
    benchmark_result: BenchmarkResult, problem_name: str, include_individual: bool
) -> List[AxPlotConfig]:
    plots: List[AxPlotConfig] = []
    # Plot objective at true best
    plots.append(
        optimization_trace_all_methods(
            y_dict=benchmark_result.objective_at_true_best,
            optimum=benchmark_result.optimum,
            title=f"{problem_name}: cumulative best objective",
            ylabel="Objective at best-feasible point observed so far",
        )
    )
    if include_individual:
        # Plot individual plots of a single method on a single problem.
        for m, y in benchmark_result.objective_at_true_best.items():
            plots.append(
                optimization_trace_single_method(
                    y=y,
                    optimum=benchmark_result.optimum,
                    # model_transitions=benchmark_result.model_transitions[m],
                    title=f"{problem_name}, {m}: cumulative best objective",
                    ylabel="Objective at best-feasible point observed so far",
                )
            )
    # Plot time
    plots.append(
        optimization_times(
            fit_times=benchmark_result.fit_times,
            gen_times=benchmark_result.gen_times,
            title=f"{problem_name}: optimization times",
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
        assert len(experiment.optimization_config.outcome_constraints) == 0
        values = np.array(
            [checked_cast(Trial, trial).objective_mean for trial in experiment.trials]
        )
        return best_feasible_objective(
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
    return best_feasible_objective(
        optimization_config=experiment.optimization_config, values=true_values
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
        optimization_config=experiment.optimization_config,
        values={problem.name: true_values},
    )

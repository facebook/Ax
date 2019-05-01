#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# pyre-strict

from itertools import product
from typing import List

from ax.benchmark.benchmark_problem import (
    BenchmarkProblem,
    branin_max,
    hartmann6_constrained,
)
from ax.benchmark.benchmark_runner import (
    BenchmarkResult,
    BenchmarkSetup,
    BOBenchmarkRunner,
)
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.plot.base import AxPlotConfig
from ax.plot.render import plot_config_to_html
from ax.plot.trace import (
    optimization_times,
    optimization_trace_all_methods,
    optimization_trace_single_method,
)
from ax.utils.report.render import h2_html, h3_html, p_html, render_report_elements


BOStrategies: List[GenerationStrategy] = [
    GenerationStrategy(
        name="Sobol", steps=[GenerationStep(model=Models.SOBOL, num_arms=50)]
    ),
    # Generation strategy to use Sobol for first 5 arms and GP+EI for next 45:
    GenerationStrategy(
        name="Sobol+GPEI",
        steps=[
            GenerationStep(model=Models.SOBOL, num_arms=5, min_arms_observed=5),
            GenerationStep(model=Models.GPEI, num_arms=-1),
        ],
    ),
]

BOProblems: List[BenchmarkProblem] = [hartmann6_constrained, branin_max]


class BOBenchmarkingSuite:
    """Suite that runs all standard Bayesian optimization benchmarks."""

    def __init__(self) -> None:
        self._runner: BOBenchmarkRunner = BOBenchmarkRunner()

    def run(
        self,
        num_runs: int,
        total_iterations: int,
        bo_strategies: List[GenerationStrategy],
        bo_problems: List[BenchmarkProblem],
        batch_size: int = 1,
        raise_all_errors: bool = False,
    ) -> BOBenchmarkRunner:
        """Run all standard BayesOpt benchmarks.

        Args:
            num_runs: How many time to run each test.
            total_iterations: How many iterations to run each optimization for.
            bo_strategies: GenerationStrategies representing each method to
                benchmark.
            bo_problems: Problems to benchmark the methods on.
            batch_size: Number of arms to be generated and evaluated in optimization
                at once.
            raise_all_errors: Debugging setting; set to true if all encountered
                errors should be raised right away (and interrupt the benchm arking)
                rather than logged and recorded.
        """

        setups = (
            BenchmarkSetup(problem, total_iterations, batch_size)
            for problem in bo_problems
        )
        for setup, gs in product(setups, bo_strategies):
            self._runner.run_benchmark_test(
                setup=setup,
                generation_strategy=gs,
                num_runs=num_runs,
                raise_all_errors=raise_all_errors,
            )

        return self._runner

    @classmethod
    def _make_plots(
        cls,
        benchmark_result: BenchmarkResult,
        problem_name: str,
        include_individual: bool,
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
                        generator_changes=benchmark_result.generator_changes[m],
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

    def generate_report(self, include_individual: bool = False) -> str:
        benchmark_result_dict = self._runner.aggregate_results()
        html_elements = [h2_html("Bayesian Optimization benchmarking suite report")]
        for p, benchmark_result in benchmark_result_dict.items():
            html_elements.append(h3_html(f"{p}:"))
            plots = self._make_plots(
                benchmark_result, problem_name=p, include_individual=include_individual
            )
            html_elements.extend(plot_config_to_html(plt) for plt in plots)
        if len(self._runner._error_messages) > 0:
            html_elements.append(h3_html("Errors encountered"))
            html_elements.extend(p_html(err) for err in self._runner._error_messages)
        else:
            html_elements.append(h3_html("No errors encountered"))
        return render_report_elements("bo_benchmark_suite_test", html_elements)

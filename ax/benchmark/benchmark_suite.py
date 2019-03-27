#!/usr/bin/env python3
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
from ax.modelbridge.factory import get_GPEI, get_sobol
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.plot.base import AEPlotConfig
from ax.plot.render import plot_config_to_html
from ax.plot.trace import (
    optimization_times,
    optimization_trace_all_methods,
    optimization_trace_single_method,
)
from ax.utils.render.render import h2_html, h3_html, p_html, render_report_elements


BOStrategies: List[GenerationStrategy] = [
    GenerationStrategy(model_factories=[get_sobol], arms_per_model=[50]),  # pyre-ignore
    # Generation strategy to use Sobol for first 5 arms and GP+EI for next 45:
    GenerationStrategy(model_factories=[get_sobol, get_GPEI], arms_per_model=[5, 45]),
]

BOProblems: List[BenchmarkProblem] = [hartmann6_constrained, branin_max]


class BOBenchmarkingSuite:
    """Suite that runs all standard Bayesian optimization benchmarks."""

    def __init__(self) -> None:
        self._runner: BOBenchmarkRunner = BOBenchmarkRunner()

    def run(
        self,
        num_trials: int,
        total_iterations: int,
        bo_strategies: List[GenerationStrategy],
        bo_problems: List[BenchmarkProblem],
        batch_size: int = 1,
    ) -> BOBenchmarkRunner:
        setups = (
            BenchmarkSetup(problem, total_iterations, batch_size)
            for problem in bo_problems
        )
        for setup, gs in product(setups, bo_strategies):
            self._runner.run_benchmark_test(
                setup=setup,
                model_factory=gs.get_model,  # pyre-ignore
                num_runs=num_trials,
            )

        return self._runner

    @classmethod
    def _make_plots(
        cls,
        benchmark_result: BenchmarkResult,
        problem_name: str,
        include_individual: bool,
    ) -> List[AEPlotConfig]:
        plots: List[AEPlotConfig] = []
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

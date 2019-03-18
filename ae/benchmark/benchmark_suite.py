#!/usr/bin/env python3
# pyre-strict

from itertools import product
from typing import List, Optional

from ae.lazarus.ae.benchmark.benchmark_problem import (
    BenchmarkProblem,
    branin_max,
    constrained_branin,
)
from ae.lazarus.ae.benchmark.benchmark_runner import (
    BenchmarkResult,
    BenchmarkSetup,
    BOBenchmarkRunner,
)
from ae.lazarus.ae.modelbridge.factory import get_GPEI, get_sobol
from ae.lazarus.ae.modelbridge.generation_strategy import (
    GenerationStrategy,
    TModelFactory,
)
from ae.lazarus.ae.plot.base import AEPlotConfig
from ae.lazarus.ae.plot.render import plot_config_to_html
from ae.lazarus.ae.plot.trace import (
    optimization_times,
    optimization_trace_all_methods,
    optimization_trace_single_method,
)
from ae.lazarus.ae.utils.render.render import (
    h2_html,
    h3_html,
    p_html,
    render_report_elements,
)


# pyre-fixme[9]: BOMethods has type `List[Callable[..., ModelBridge]]`; used as `List...
BOMethods: List[TModelFactory] = [
    get_sobol,
    # Generation strategy to use Sobol for first 5 arms and GP+EI for next 30:
    GenerationStrategy([get_sobol, get_GPEI], [5, 30]).get_model,
]

BOProblems: List[BenchmarkProblem] = [constrained_branin, branin_max]


class BOBenchmarkingSuite:
    """Suite that runs all standard Bayesian optimization benchmarks."""

    def __init__(self) -> None:
        self._runner: BOBenchmarkRunner = BOBenchmarkRunner()

    def run(
        self,
        num_trials: int = 10,
        total_iterations: int = 20,
        batch_size: int = 1,
        bo_methods: Optional[List[TModelFactory]] = None,
        bo_problems: Optional[List[BenchmarkProblem]] = None,
    ) -> BOBenchmarkRunner:
        if bo_methods is None:
            bo_methods = BOMethods
        if bo_problems is None:
            bo_problems = BOProblems
        setups = (
            BenchmarkSetup(problem, total_iterations, batch_size)
            for problem in bo_problems
        )
        for setup, model_factory in product(setups, bo_methods):
            self._runner.run_benchmark_test(setup, model_factory, num_trials)

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
        # TODO[drfreund]: plot other metrics
        return plots

    def generate_report(self, include_individual: bool = True) -> str:
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

#!/usr/bin/env python3
from typing import Any

from ax.benchmark.benchmark_problem import BenchmarkProblem, branin
from ax.benchmark.benchmark_suite import (
    BOBenchmarkingSuite,
    BOMethods,
    BOProblems,
)
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types.types import ComparisonOp
from ax.metrics.branin import BraninConstraintMetric, BraninMetric
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.tests.fake import get_branin_search_space
from ax.utils.common.testutils import TestCase


def fail(*args: Any, **kwargs: Any) -> None:
    raise ValueError


class TestBOBenchmarkingSuite(TestCase):
    def test_basic(self):
        num_trials = 3
        total_iterations = 2
        suite = BOBenchmarkingSuite()
        runner = suite.run(num_trials=num_trials, total_iterations=total_iterations)
        self.assertEqual(
            len(runner._runs.items()), len(BOMethods) * len(BOProblems) * num_trials
        )
        # If run_benchmarking_trial fails, corresponding trial in '_runs' is None.
        self.assertTrue(all(x is not None for x in runner._runs.values()))
        # Make sure no errors came up in running trials.
        runner.aggregate_results()
        # Also test that the suite works with batch trials.
        BOBenchmarkingSuite().run(
            num_trials=num_trials, total_iterations=total_iterations, batch_size=2
        )
        report = suite.generate_report()
        self.assertIsInstance(report, str)

    def test_repeat_problem_method_combo(self):
        suite = BOBenchmarkingSuite()
        runner = suite.run(
            num_trials=1, total_iterations=1, bo_methods=[get_sobol, get_sobol]
        )
        self.assertRegex(runner.errors[0], r"^Run [0-9]* of .* on")
        self.assertGreater(len(runner._runs), 0)
        report = suite.generate_report()
        self.assertIsInstance(report, str)

    def test_run_should_fail(self):
        suite = BOBenchmarkingSuite()
        runner = suite.run(num_trials=1, total_iterations=1, bo_methods=[fail])
        self.assertEqual(len(runner._runs), 0)
        self.assertRegex(runner.errors[5], r"^Considering")

    def test_sobol(self):
        suite = BOBenchmarkingSuite()
        runner = suite.run(
            num_trials=1,
            total_iterations=5,
            batch_size=2,
            bo_methods=[get_sobol],
            bo_problems=[branin],
        )
        # If run_benvhmarking_trial fails, corresponding trial in '_runs' is None.
        self.assertTrue(all(x is not None for x in runner._runs.values()))
        # Make sure no errors came up in running trials.
        self.assertEqual(len(runner.errors), 0)
        report = suite.generate_report()
        self.assertIsInstance(report, str)

    def test_generation_strategy(self):
        bo_methods = [
            get_sobol,
            GenerationStrategy([get_sobol, get_sobol], [5, 30]).get_model,
        ]
        suite = BOBenchmarkingSuite()
        suite.run(
            num_trials=1,
            total_iterations=5,
            bo_methods=bo_methods,
            bo_problems=[branin],
        )
        self.assertIsInstance(suite.generate_report(), str)

    def testRelativeConstraint(self):
        branin_rel = BenchmarkProblem(
            name="constrained_branin",
            fbest=0.397887,
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=BraninMetric(
                        name="branin_objective", param_names=["x1", "x2"], noise_sd=5.0
                    ),
                    minimize=True,
                ),
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=BraninConstraintMetric(
                            name="branin_constraint",
                            param_names=["x1", "x2"],
                            noise_sd=5.0,
                        ),
                        op=ComparisonOp.LEQ,
                        bound=0.0,
                        relative=True,
                    )
                ],
            ),
            search_space=get_branin_search_space(),
        )
        suite = BOBenchmarkingSuite()
        suite.run(
            num_trials=1,
            total_iterations=5,
            bo_methods=[get_sobol],
            bo_problems=[branin_rel],
        )
        with self.assertRaises(ValueError):
            suite.generate_report()

    def testLowerBound(self):
        branin_lb = BenchmarkProblem(
            name="constrained_branin",
            fbest=0.397887,
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=BraninMetric(
                        name="branin_objective", param_names=["x1", "x2"], noise_sd=5.0
                    ),
                    minimize=True,
                ),
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=BraninConstraintMetric(
                            name="branin_constraint",
                            param_names=["x1", "x2"],
                            noise_sd=5.0,
                        ),
                        op=ComparisonOp.GEQ,
                        bound=0.0,
                        relative=False,
                    )
                ],
            ),
            search_space=get_branin_search_space(),
        )
        suite = BOBenchmarkingSuite()
        suite.run(
            num_trials=1,
            total_iterations=5,
            bo_methods=[get_sobol],
            bo_problems=[branin_lb],
        )
        suite.generate_report()

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Any

from ax.benchmark.benchmark_problem import BenchmarkProblem, branin
from ax.benchmark.benchmark_suite import BOBenchmarkingSuite, BOProblems, BOStrategies
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.metrics.branin import BraninMetric
from ax.metrics.l2norm import L2NormMetric
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.utils.common.testutils import TestCase
from ax.utils.testing.fake import get_branin_search_space


def fail(*args: Any, **kwargs: Any) -> None:
    raise ValueError("This is failing intentionally for testing purposes.")


class TestBOBenchmarkingSuite(TestCase):
    def test_basic(self):
        num_runs = 3
        total_iterations = 2
        suite = BOBenchmarkingSuite()
        runner = suite.run(
            num_runs=num_runs,
            total_iterations=total_iterations,
            bo_strategies=BOStrategies,
            bo_problems=BOProblems,
        )
        self.assertEqual(
            len(runner._runs.items()), len(BOStrategies) * len(BOProblems) * num_runs
        )
        # If run_benchmarking_trial fails, corresponding trial in '_runs' is None.
        self.assertTrue(all(x is not None for x in runner._runs.values()))
        # Make sure no errors came up in running trials.
        runner.aggregate_results()
        # Also test that the suite works with batch trials.
        BOBenchmarkingSuite().run(
            num_runs=num_runs,
            total_iterations=total_iterations,
            bo_strategies=BOStrategies,
            bo_problems=BOProblems,
            batch_size=2,
        )
        report = suite.generate_report()
        self.assertIsInstance(report, str)

    def test_repeat_problem_method_combo(self):
        suite = BOBenchmarkingSuite()
        runner = suite.run(
            num_runs=1,
            total_iterations=1,
            bo_strategies=[
                GenerationStrategy([GenerationStep(model=Models.SOBOL, num_arms=5)])
            ]
            * 2,
            bo_problems=BOProblems,
        )
        self.assertRegex(runner.errors[0], r"^Run [0-9]* of .* on")
        self.assertGreater(len(runner._runs), 0)
        report = suite.generate_report()
        self.assertIsInstance(report, str)

    def test_run_should_fail(self):
        suite = BOBenchmarkingSuite()
        runner = suite.run(
            num_runs=1,
            total_iterations=1,
            bo_strategies=[
                GenerationStrategy([GenerationStep(model=fail, num_arms=30)])
            ],
            bo_problems=BOProblems,
        )
        self.assertEqual(len(runner._runs), 0)
        self.assertRegex(runner.errors[5], r"^Considering")

    def test_sobol(self):
        suite = BOBenchmarkingSuite()
        runner = suite.run(
            num_runs=1,
            total_iterations=5,
            batch_size=2,
            bo_strategies=[
                GenerationStrategy([GenerationStep(model=Models.SOBOL, num_arms=10)])
            ],
            bo_problems=[branin],
        )
        # If run_benchmarking_trial fails, corresponding trial in '_runs' is None.
        self.assertTrue(all(x is not None for x in runner._runs.values()))
        # Make sure no errors came up in running trials.
        self.assertEqual(len(runner.errors), 0)
        report = suite.generate_report()
        self.assertIsInstance(report, str)

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
                        metric=L2NormMetric(
                            name="branin_constraint",
                            param_names=["x1", "x2"],
                            noise_sd=5.0,
                        ),
                        op=ComparisonOp.LEQ,
                        bound=5.0,
                        relative=True,
                    )
                ],
            ),
            search_space=get_branin_search_space(),
        )
        suite = BOBenchmarkingSuite()
        suite.run(
            num_runs=1,
            total_iterations=5,
            bo_strategies=[
                GenerationStrategy([GenerationStep(model=Models.SOBOL, num_arms=5)])
            ],
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
                        metric=L2NormMetric(
                            name="branin_constraint",
                            param_names=["x1", "x2"],
                            noise_sd=5.0,
                        ),
                        op=ComparisonOp.GEQ,
                        bound=5.0,
                        relative=False,
                    )
                ],
            ),
            search_space=get_branin_search_space(),
        )
        suite = BOBenchmarkingSuite()
        suite.run(
            num_runs=1,
            batch_size=2,
            total_iterations=4,
            bo_strategies=[
                GenerationStrategy([GenerationStep(model=Models.SOBOL, num_arms=5)])
            ],
            bo_problems=[branin_lb],
        )
        suite.generate_report(include_individual=True)

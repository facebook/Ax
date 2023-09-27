# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.benchmark_problem import (
    MultiObjectiveBenchmarkProblem,
    SingleObjectiveBenchmarkProblem,
)
from ax.core.types import ComparisonOp
from ax.utils.common.testutils import TestCase
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.test_functions.synthetic import Ackley, ConstrainedHartmann


class TestBenchmarkProblem(TestCase):
    def test_single_objective_from_botorch(self) -> None:
        for botorch_test_problem in [Ackley(), ConstrainedHartmann(dim=6)]:
            test_problem = SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
                test_problem_class=botorch_test_problem.__class__,
                test_problem_kwargs={},
                num_trials=1,
            )

            # Test search space
            self.assertEqual(
                len(test_problem.search_space.parameters), botorch_test_problem.dim
            )
            self.assertEqual(
                len(test_problem.search_space.parameters),
                len(test_problem.search_space.range_parameters),
            )
            self.assertTrue(
                all(
                    # pyre-fixme[16]: `Parameter` has no attribute `lower`.
                    test_problem.search_space.range_parameters[f"x{i}"].lower
                    == botorch_test_problem._bounds[i][0]
                    for i in range(botorch_test_problem.dim)
                ),
                "Parameters' lower bounds must all match Botorch problem's bounds.",
            )
            self.assertTrue(
                all(
                    # pyre-fixme[16]: `Parameter` has no attribute `upper`.
                    test_problem.search_space.range_parameters[f"x{i}"].upper
                    == botorch_test_problem._bounds[i][1]
                    for i in range(botorch_test_problem.dim)
                ),
                "Parameters' upper bounds must all match Botorch problem's bounds.",
            )

            # Test optimum
            self.assertEqual(
                test_problem.optimal_value, botorch_test_problem._optimal_value
            )
            # test optimization config
            self.assertEqual(
                test_problem.optimization_config.objective.metric.name,
                test_problem.name,
            )
            self.assertTrue(test_problem.optimization_config.objective.minimize)
            # test repr method
            if isinstance(botorch_test_problem, Ackley):
                self.assertEqual(
                    test_problem.optimization_config.outcome_constraints, []
                )
                expected_repr = (
                    "SingleObjectiveBenchmarkProblem(name=Ackley, "
                    "optimization_config=OptimizationConfig(objective=Objective("
                    'metric_name="Ackley", '
                    "minimize=True), outcome_constraints=[]), "
                    "num_trials=1, "
                    "infer_noise=True, "
                    "tracking_metrics=[])"
                )
            else:
                outcome_constraint = (
                    test_problem.optimization_config.outcome_constraints[0]
                )
                self.assertEqual(outcome_constraint.metric.name, "constraint_slack_0")
                self.assertEqual(outcome_constraint.op, ComparisonOp.GEQ)
                self.assertFalse(outcome_constraint.relative)
                self.assertEqual(outcome_constraint.bound, 0.0)
                expected_repr = (
                    "SingleObjectiveBenchmarkProblem(name=ConstrainedHartmann, "
                    "optimization_config=OptimizationConfig(objective=Objective("
                    'metric_name="ConstrainedHartmann", minimize=True), '
                    "outcome_constraints=[OutcomeConstraint(constraint_slack_0"
                    " >= 0.0)]), num_trials=1, infer_noise=True, "
                    "tracking_metrics=[])"
                )

            self.assertEqual(repr(test_problem), expected_repr)

    def test_moo_from_botorch(self) -> None:
        test_problem = BraninCurrin()
        branin_currin_problem = (
            MultiObjectiveBenchmarkProblem.from_botorch_multi_objective(
                test_problem_class=test_problem.__class__,
                test_problem_kwargs={},
                num_trials=1,
            )
        )

        # Test search space
        self.assertEqual(
            len(branin_currin_problem.search_space.parameters), test_problem.dim
        )
        self.assertEqual(
            len(branin_currin_problem.search_space.parameters),
            len(branin_currin_problem.search_space.range_parameters),
        )
        self.assertTrue(
            all(
                # pyre-fixme[16]: `Parameter` has no attribute `lower`.
                branin_currin_problem.search_space.range_parameters[f"x{i}"].lower
                == test_problem._bounds[i][0]
                for i in range(test_problem.dim)
            ),
            "Parameters' lower bounds must all match Botorch problem's bounds.",
        )
        self.assertTrue(
            all(
                # pyre-fixme[16]: `Parameter` has no attribute `upper`.
                branin_currin_problem.search_space.range_parameters[f"x{i}"].upper
                == test_problem._bounds[i][1]
                for i in range(test_problem.dim)
            ),
            "Parameters' upper bounds must all match Botorch problem's bounds.",
        )

        # Test hypervolume
        self.assertEqual(
            branin_currin_problem.maximum_hypervolume, test_problem._max_hv
        )
        self.assertEqual(branin_currin_problem.reference_point, test_problem._ref_point)

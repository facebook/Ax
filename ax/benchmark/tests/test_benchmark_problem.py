# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.benchmark_problem import (
    MultiObjectiveBenchmarkProblem,
    SingleObjectiveBenchmarkProblem,
)
from ax.utils.common.testutils import TestCase
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.test_functions.synthetic import Ackley


class TestBenchmarkProblem(TestCase):
    def test_single_objective_from_botorch(self) -> None:
        test_problem = Ackley()
        ackley_problem = SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
            test_problem_class=test_problem.__class__,
            test_problem_kwargs={},
            num_trials=1,
        )

        # Test search space
        self.assertEqual(len(ackley_problem.search_space.parameters), test_problem.dim)
        self.assertEqual(
            len(ackley_problem.search_space.parameters),
            len(ackley_problem.search_space.range_parameters),
        )
        self.assertTrue(
            all(
                # pyre-fixme[16]: `Parameter` has no attribute `lower`.
                ackley_problem.search_space.range_parameters[f"x{i}"].lower
                == test_problem._bounds[i][0]
                for i in range(test_problem.dim)
            ),
            "Parameters' lower bounds must all match Botorch problem's bounds.",
        )
        self.assertTrue(
            all(
                # pyre-fixme[16]: `Parameter` has no attribute `upper`.
                ackley_problem.search_space.range_parameters[f"x{i}"].upper
                == test_problem._bounds[i][1]
                for i in range(test_problem.dim)
            ),
            "Parameters' upper bounds must all match Botorch problem's bounds.",
        )

        # Test optimum
        self.assertEqual(ackley_problem.optimal_value, test_problem._optimal_value)

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

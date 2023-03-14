# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.problems.synthetic.discretized.mixed_integer import (
    get_discrete_ackley,
    get_discrete_hartmann,
    get_discrete_rosenbrock,
)
from ax.core.parameter import ParameterType
from ax.runners.botorch_test_problem import BotorchTestProblemRunner
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast


class TestMixedIntegerProblems(TestCase):
    def test_problems(self) -> None:
        for name, constructor, dim, dim_int in (
            ("Hartmann", get_discrete_hartmann, 6, 4),
            ("Ackley", get_discrete_ackley, 13, 10),
            ("Rosenbrock", get_discrete_rosenbrock, 10, 6),
        ):
            problem = constructor()
            self.assertEqual(f"Discrete {name}", problem.name)
            self.assertEqual(
                checked_cast(
                    BotorchTestProblemRunner, problem.runner
                )._test_problem_class.__name__,
                name,
            )
            self.assertEqual(len(problem.search_space.parameters), dim)
            self.assertEqual(
                sum(
                    p.parameter_type == ParameterType.INT
                    for p in problem.search_space.parameters.values()
                ),
                dim_int,
            )
            # Check that the underlying problem has the correct bounds.
            if name == "Rosenbrock":
                expected_bounds = [(-5.0, 10.0) for _ in range(dim)]
            else:
                expected_bounds = [(0.0, 1.0) for _ in range(dim)]
            self.assertEqual(
                checked_cast(
                    BotorchTestProblemRunner, problem.runner
                ).test_problem._bounds,
                expected_bounds,
            )

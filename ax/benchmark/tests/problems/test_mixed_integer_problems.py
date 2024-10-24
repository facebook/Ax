# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import MagicMock, patch

import torch
from ax.benchmark.benchmark_problem import BenchmarkProblem

from ax.benchmark.problems.synthetic.discretized.mixed_integer import (
    get_discrete_ackley,
    get_discrete_hartmann,
    get_discrete_rosenbrock,
)
from ax.benchmark.runners.botorch_test import (
    BoTorchTestProblem,
    ParamBasedTestProblemRunner,
)
from ax.core.arm import Arm
from ax.core.parameter import ParameterType
from ax.core.trial import Trial
from ax.utils.common.testutils import TestCase
from botorch.test_functions.synthetic import Ackley, Hartmann, Rosenbrock
from pyre_extensions import assert_is_instance


class MixedIntegerProblemsTest(TestCase):
    def test_problems(self) -> None:
        for problem_cls, constructor, dim, dim_int in (
            (Hartmann, get_discrete_hartmann, 6, 4),
            (Ackley, get_discrete_ackley, 13, 10),
            (Rosenbrock, get_discrete_rosenbrock, 10, 6),
        ):
            name = problem_cls.__name__
            problem = constructor()
            self.assertEqual(f"Discrete {name}", problem.name)
            runner = assert_is_instance(problem.runner, ParamBasedTestProblemRunner)
            test_problem = assert_is_instance(runner.test_problem, BoTorchTestProblem)
            botorch_problem = test_problem.botorch_problem
            self.assertIsInstance(botorch_problem, problem_cls)
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
            self.assertEqual(botorch_problem._bounds, expected_bounds)
            self.assertGreaterEqual(problem.optimal_value, problem_cls().optimal_value)

        # Test that they match correctly to the original problems.
        cases: list[tuple[BenchmarkProblem, dict[str, float], torch.Tensor]] = [
            (
                get_discrete_hartmann(),
                {f"x{i+1}": 0.0 for i in range(6)},
                torch.zeros(6, dtype=torch.double),
            ),
            (
                get_discrete_hartmann(),
                {"x1": 3, "x2": 3, "x3": 19, "x4": 19, "x5": 1.0, "x6": 1.0},
                torch.ones(6, dtype=torch.double),
            ),
            (
                get_discrete_ackley(),
                {f"x{i+1}": 0.0 for i in range(13)},
                torch.zeros(13, dtype=torch.double),
            ),
            (
                get_discrete_ackley(),
                {
                    **{f"x{i+1}": 2 for i in range(0, 5)},
                    **{f"x{i+1}": 4 for i in range(5, 10)},
                    **{f"x{i+1}": 1.0 for i in range(10, 13)},
                },
                torch.ones(13, dtype=torch.double),
            ),
            (
                get_discrete_rosenbrock(),
                {f"x{i+1}": 0.0 for i in range(10)},
                torch.full((10,), -5.0, dtype=torch.double),
            ),
            (
                get_discrete_rosenbrock(),
                {
                    **{f"x{i+1}": 3 for i in range(0, 6)},
                    **{f"x{i+1}": 1.0 for i in range(6, 10)},
                },
                torch.full((10,), 10.0, dtype=torch.double),
            ),
        ]

        for problem, params, expected_arg in cases:
            runner = assert_is_instance(problem.runner, ParamBasedTestProblemRunner)
            test_problem = assert_is_instance(runner.test_problem, BoTorchTestProblem)
            trial = Trial(experiment=MagicMock())
            # pyre-fixme: Incompatible parameter type [6]: In call
            # `Arm.__init__`, for argument `parameters`, expected `Dict[str,
            # Union[None, bool, float, int, str]]` but got `dict[str, float]`.
            arm = Arm(parameters=params, name="--")
            trial.add_arm(arm)
            with patch.object(
                test_problem.botorch_problem,
                attribute="evaluate_true",
                wraps=test_problem.botorch_problem.evaluate_true,
            ) as mock_call:
                runner.run(trial)
            actual = mock_call.call_args[0][0]
            self.assertTrue(torch.allclose(actual, expected_arg))

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import MagicMock

import torch

from ax.benchmark.problems.synthetic.discretized.mixed_integer import (
    get_discrete_ackley,
    get_discrete_hartmann,
    get_discrete_rosenbrock,
)
from ax.benchmark.runners.botorch_test import BotorchTestProblemRunner
from ax.core.arm import Arm
from ax.core.parameter import ParameterType
from ax.core.trial import Trial
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast, not_none


class MixedIntegerProblemsTest(TestCase):
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

        # Test that they match correctly to the original problems.
        # Hartmann - evaluate at 0 - should correspond to 0.
        runner = checked_cast(BotorchTestProblemRunner, get_discrete_hartmann().runner)
        mock_call = MagicMock(return_value=torch.tensor(0.0))
        runner.test_problem.evaluate_true = mock_call
        trial = Trial(experiment=MagicMock())
        arm = Arm(parameters={f"x{i + 1}": 0.0 for i in range(6)}, name="--")
        trial.add_arm(arm)
        runner.run(trial)
        actual = mock_call.call_args[0][0]
        self.assertTrue(torch.allclose(actual, torch.zeros(6, dtype=actual.dtype)))
        # Evaluate at 3, 3, 19, 19, 1, 1 - corresponds to 1.
        arm = not_none(trial.arm)
        arm._parameters = {
            "x1": 3,
            "x2": 3,
            "x3": 19,
            "x4": 19,
            "x5": 1.0,
            "x6": 1.0,
        }
        runner.run(trial)
        actual = mock_call.call_args[0][0]
        self.assertTrue(torch.allclose(actual, torch.ones(6, dtype=actual.dtype)))
        # Ackley - evaluate at 0 - corresponds to 0.
        runner = checked_cast(BotorchTestProblemRunner, get_discrete_ackley().runner)
        runner.test_problem.evaluate_true = mock_call
        arm._parameters = {f"x{i+1}": 0.0 for i in range(13)}
        runner.run(trial)
        actual = mock_call.call_args[0][0]
        self.assertTrue(torch.allclose(actual, torch.zeros(13, dtype=actual.dtype)))
        # Evaluate at 2 x 5, 4 x 5, 1.0 x 3 - corresponds to 1.
        arm._parameters = {
            **{f"x{i+1}": 2 for i in range(0, 5)},
            **{f"x{i+1}": 4 for i in range(5, 10)},
            **{f"x{i+1}": 1.0 for i in range(10, 13)},
        }
        runner.run(trial)
        actual = mock_call.call_args[0][0]
        self.assertTrue(torch.allclose(actual, torch.ones(13, dtype=actual.dtype)))
        # Rosenbrock - evaluate at 0 - corresponds to -5.0.
        runner = checked_cast(
            BotorchTestProblemRunner, get_discrete_rosenbrock().runner
        )
        runner.test_problem.evaluate_true = mock_call
        arm._parameters = {f"x{i+1}": 0.0 for i in range(10)}
        runner.run(trial)
        actual = mock_call.call_args[0][0]
        self.assertTrue(
            torch.allclose(actual, torch.full((10,), -5.0, dtype=actual.dtype))
        )
        # Evaluate at 3 x 6, 1.0 x 4 - corresponds to 10.0.
        arm._parameters = {
            **{f"x{i+1}": 3 for i in range(0, 6)},
            **{f"x{i+1}": 1.0 for i in range(6, 10)},
        }
        runner.run(trial)
        actual = mock_call.call_args[0][0]
        self.assertTrue(
            torch.allclose(actual, torch.full((10,), 10.0, dtype=actual.dtype))
        )

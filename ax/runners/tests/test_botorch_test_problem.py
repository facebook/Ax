#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from itertools import product
from unittest.mock import Mock

import torch
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.runners.botorch_test_problem import BotorchTestProblemRunner
from ax.utils.common.testutils import TestCase
from botorch.test_functions.base import ConstrainedBaseTestProblem
from botorch.test_functions.synthetic import ConstrainedHartmann, Hartmann
from botorch.utils.transforms import normalize


class TestBotorchTestProblemRunner(TestCase):
    def test_botorch_test_problem_runner(self) -> None:
        for test_problem_class, modified_bounds in product(
            (Hartmann, ConstrainedHartmann), (None, [(0.0, 2.0)] * 6)
        ):
            test_problem = test_problem_class(dim=6).to(dtype=torch.double)
            test_problem_kwargs = {"dim": 6}
            runner = BotorchTestProblemRunner(
                test_problem_class=test_problem_class,
                test_problem_kwargs=test_problem_kwargs,
                modified_bounds=modified_bounds,
            )
            self.assertIsInstance(runner.test_problem, test_problem_class)
            self.assertEqual(runner.test_problem.dim, test_problem_kwargs["dim"])
            self.assertEqual(runner.test_problem.bounds.dtype, torch.double)
            self.assertEqual(
                runner._is_constrained,
                isinstance(test_problem, ConstrainedBaseTestProblem),
            )
            self.assertEqual(runner._modified_bounds, modified_bounds)
            # check equality with different class
            self.assertNotEqual(runner, Hartmann(dim=6))
            self.assertEqual(runner, runner)
            # test evaluate with original bounds
            X = torch.rand(1, 6, dtype=torch.double)
            res = runner.evaluate_with_original_bounds(X)
            if modified_bounds is not None:
                X_tf = normalize(X, torch.tensor(modified_bounds, dtype=torch.double).T)
            else:
                X_tf = X
            obj = test_problem(X_tf)
            if runner._is_constrained:
                expected_res = torch.cat(
                    [obj.view(-1), test_problem.evaluate_slack(X_tf).view(-1)], dim=-1
                )
            else:
                expected_res = obj
            self.assertTrue(torch.equal(res, expected_res))

            # test run
            trial = Mock()
            trial.arms = [
                Arm(
                    name="0_0",
                    parameters={f"x{i}": X[:, i].item() for i in range(6)},
                )
            ]
            trial.index = 0
            res = runner.run(trial=trial)
            self.assertEqual(res, {"Ys": {"0_0": expected_res.tolist()}})
            # test poll trial status
            self.assertEqual(
                {TrialStatus.COMPLETED: {0}}, runner.poll_trial_status([trial])
            )
            # test serialize args
            serialize_init_args = BotorchTestProblemRunner.serialize_init_args(
                obj=runner
            )
            self.assertEqual(
                serialize_init_args,
                {
                    "test_problem_module": runner._test_problem_class.__module__,
                    "test_problem_class_name": runner._test_problem_class.__name__,
                    "test_problem_kwargs": runner._test_problem_kwargs,
                    "modified_bounds": runner._modified_bounds,
                },
            )
            # test deserialize args
            deserialize_init_args = BotorchTestProblemRunner.deserialize_init_args(
                serialize_init_args
            )
            self.assertEqual(
                deserialize_init_args,
                {
                    "test_problem_class": test_problem_class,
                    "test_problem_kwargs": test_problem_kwargs,
                    "modified_bounds": modified_bounds,
                },
            )

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from itertools import product
from typing import Dict, Union
from unittest.mock import Mock

import torch
from ax.benchmark.runners.botorch_test import BotorchTestProblemRunner
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.trial import Trial
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast
from botorch.test_functions.base import ConstrainedBaseTestProblem
from botorch.test_functions.synthetic import ConstrainedHartmann, Hartmann
from botorch.utils.transforms import normalize


class TestBotorchTestProblemRunner(TestCase):
    def test_botorch_test_problem_runner(self) -> None:
        for test_problem_class, modified_bounds, noise_std in product(
            (Hartmann, ConstrainedHartmann), (None, [(0.0, 2.0)] * 6), (None, 0.1)
        ):
            test_problem = test_problem_class(dim=6).to(dtype=torch.double)
            test_problem_kwargs: Dict[str, Union[int, float]] = {"dim": 6}
            if noise_std is not None:
                test_problem_kwargs["noise_std"] = noise_std
            outcome_names = ["objective"]
            if test_problem_class == ConstrainedHartmann:
                outcome_names = outcome_names + ["constraint"]

            runner = BotorchTestProblemRunner(
                test_problem_class=test_problem_class,
                test_problem_kwargs=test_problem_kwargs,
                outcome_names=outcome_names,
                modified_bounds=modified_bounds,
            )

            test_description: str = (
                f"test problem: {test_problem_class.__name__}, "
                f"modified_bounds: {modified_bounds}, "
                f"noise_std: {noise_std}."
            )

            with self.subTest(f"Test basic construction, {test_description}"):
                self.assertIsInstance(runner.test_problem, test_problem_class)
                self.assertEqual(runner.test_problem.dim, test_problem_kwargs["dim"])
                self.assertEqual(runner.test_problem.bounds.dtype, torch.double)
                self.assertEqual(
                    runner._is_constrained,
                    isinstance(test_problem, ConstrainedBaseTestProblem),
                )
                self.assertEqual(runner._modified_bounds, modified_bounds)
                if noise_std is not None:
                    self.assertEqual(runner.get_noise_stds(), noise_std)
                else:
                    self.assertIsNone(runner.get_noise_stds())

                # check equality with different class
                self.assertNotEqual(runner, Hartmann(dim=6))
                self.assertEqual(runner, runner)

            with self.subTest(f"test `get_Y_true()`, {test_description}"):
                X = torch.rand(1, 6, dtype=torch.double)
                arm = Arm(
                    name="0_0",
                    parameters={f"x{i}": x.item() for i, x in enumerate(X.unbind(-1))},
                )
                Y = runner.get_Y_true(arm=arm)
                if modified_bounds is not None:
                    X_tf = normalize(
                        X, torch.tensor(modified_bounds, dtype=torch.double).T
                    )
                else:
                    X_tf = X
                obj = test_problem.evaluate_true(X_tf)
                if test_problem.negate:
                    obj = -obj
                if runner._is_constrained:
                    expected_Y = torch.cat(
                        [obj.view(-1), test_problem.evaluate_slack(X_tf).view(-1)],
                        dim=-1,
                    )
                else:
                    expected_Y = obj
                self.assertTrue(torch.allclose(Y, expected_Y))

            with self.subTest(f"test `run()`, {test_description}"):
                trial = Mock(spec=Trial)
                trial.arms = [arm]
                trial.arm = arm
                trial.index = 0
                res = runner.run(trial=trial)
                self.assertSetEqual(
                    set(res.keys()), {"Ys", "Ys_true", "Ystds", "outcome_names"}
                )
                self.assertSetEqual(set(res["Ys"].keys()), {"0_0"})
                self.assertEqual(res["Ys_true"]["0_0"], Y.tolist())
                if noise_std is not None:
                    self.assertEqual(res["Ystds"]["0_0"], [noise_std] * len(Y))
                else:
                    self.assertEqual(res["Ys"]["0_0"], Y.tolist())
                    self.assertEqual(res["Ystds"]["0_0"], [0.0] * len(Y))
                self.assertEqual(res["outcome_names"], outcome_names)

            with self.subTest(f"test `poll_trial_status()`, {test_description}"):
                self.assertEqual(
                    {TrialStatus.COMPLETED: {0}}, runner.poll_trial_status([trial])
                )

            with self.subTest(f"test `serialize_init_args()`, {test_description}"):
                serialize_init_args = BotorchTestProblemRunner.serialize_init_args(
                    obj=runner
                )
                self.assertEqual(
                    serialize_init_args,
                    {
                        "test_problem_module": runner._test_problem_class.__module__,
                        "test_problem_class_name": runner._test_problem_class.__name__,
                        "test_problem_kwargs": runner._test_problem_kwargs,
                        "outcome_names": runner._outcome_names,
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
                        "outcome_names": outcome_names,
                        "modified_bounds": modified_bounds,
                    },
                )

    def test_botorch_test_problem_runner_heterogeneous_noise(self) -> None:
        runner = BotorchTestProblemRunner(
            test_problem_class=ConstrainedHartmann,
            test_problem_kwargs={
                "dim": 6,
                "noise_std": 0.1,
                "constraint_noise_std": 0.05,
            },
            outcome_names=["objective", "constraint"],
        )
        self.assertDictEqual(
            checked_cast(dict, runner.get_noise_stds()),
            {"objective": 0.1, "constraint": 0.05},
        )

        X = torch.rand(1, 6, dtype=torch.double)
        arm = Arm(
            name="0_0",
            parameters={f"x{i}": x.item() for i, x in enumerate(X.unbind(-1))},
        )
        trial = Mock(spec=Trial)
        trial.arms = [arm]
        trial.arm = arm
        trial.index = 0
        res = runner.run(trial=trial)
        self.assertSetEqual(
            set(res.keys()), {"Ys", "Ys_true", "Ystds", "outcome_names"}
        )
        self.assertSetEqual(set(res["Ys"].keys()), {"0_0"})
        self.assertEqual(res["Ystds"]["0_0"], [0.1, 0.05])
        self.assertEqual(res["outcome_names"], ["objective", "constraint"])

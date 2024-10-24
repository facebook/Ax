#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from contextlib import nullcontext
from dataclasses import replace
from itertools import product
from unittest.mock import Mock, patch

import numpy as np

import torch
from ax.benchmark.problems.synthetic.hss.jenatton import get_jenatton_benchmark_problem
from ax.benchmark.runners.botorch_test import (
    BoTorchTestProblem,
    ParamBasedTestProblemRunner,
)
from ax.benchmark.runners.surrogate import SurrogateTestFunction
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.trial import Trial
from ax.exceptions.core import UnsupportedError
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast
from ax.utils.testing.benchmark_stubs import (
    get_soo_surrogate_test_function,
    TestParamBasedTestProblem,
)
from botorch.test_functions.synthetic import Ackley, ConstrainedHartmann, Hartmann
from botorch.utils.transforms import normalize


class TestSyntheticRunner(TestCase):
    def test_runner_raises_for_botorch_attrs(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "noise_std should be set on the runner, not the test problem."
        ):
            ParamBasedTestProblemRunner(
                test_problem=BoTorchTestProblem(
                    botorch_problem=Hartmann(dim=6, noise_std=0.1)
                ),
                outcome_names=["objective"],
            )
        with self.assertRaisesRegex(
            ValueError,
            "constraint_noise_std should be set on the runner, not the test problem.",
        ):
            ParamBasedTestProblemRunner(
                test_problem=BoTorchTestProblem(
                    botorch_problem=ConstrainedHartmann(dim=6, constraint_noise_std=0.1)
                ),
                outcome_names=["objective", "constraint"],
            )
        with self.assertRaisesRegex(
            ValueError, "negate should be set on the runner, not the test problem."
        ):
            ParamBasedTestProblemRunner(
                test_problem=BoTorchTestProblem(
                    botorch_problem=Hartmann(dim=6, negate=True)
                ),
                outcome_names=["objective"],
            )

    def setUp(self) -> None:
        super().setUp()
        self.maxDiff = None

    def test_synthetic_runner(self) -> None:
        botorch_cases = [
            (
                BoTorchTestProblem(
                    botorch_problem=test_problem_class(dim=6),
                    modified_bounds=modified_bounds,
                ),
                noise_std,
            )
            for test_problem_class, modified_bounds, noise_std in product(
                (Hartmann, ConstrainedHartmann),
                (None, [(0.0, 2.0)] * 6),
                (None, 0.1),
            )
        ]
        param_based_cases = [
            (
                TestParamBasedTestProblem(num_objectives=num_objectives, dim=6),
                noise_std,
            )
            for num_objectives, noise_std in product((1, 2), (None, 0.0, 1.0))
        ]
        surrogate_cases = [
            (get_soo_surrogate_test_function(lazy=False), noise_std)
            for noise_std in (None, 0.0, 1.0)
        ]
        for test_problem, noise_std in (
            botorch_cases + param_based_cases + surrogate_cases
        ):
            num_objectives = test_problem.num_objectives

            outcome_names = [f"objective_{i}" for i in range(num_objectives)]
            is_constrained = isinstance(
                test_problem, BoTorchTestProblem
            ) and isinstance(test_problem.botorch_problem, ConstrainedHartmann)
            if is_constrained:
                outcome_names = outcome_names + ["constraint"]

            runner = ParamBasedTestProblemRunner(
                test_problem=test_problem,
                outcome_names=outcome_names,
                noise_std=noise_std,
            )
            modified_bounds = (
                test_problem.modified_bounds
                if isinstance(test_problem, BoTorchTestProblem)
                else None
            )

            test_description: str = (
                f"test problem: {test_problem.__class__.__name__}, "
                f"modified_bounds: {modified_bounds}, "
                f"noise_std: {noise_std}."
            )
            is_botorch = isinstance(test_problem, BoTorchTestProblem)

            with self.subTest(f"Test basic construction, {test_description}"):
                self.assertIs(runner.test_problem, test_problem)
                self.assertEqual(runner._is_constrained, is_constrained)
                self.assertEqual(runner.outcome_names, outcome_names)
                if noise_std is not None:
                    self.assertEqual(runner.get_noise_stds(), noise_std)
                else:
                    self.assertIsNone(runner.get_noise_stds())

                # check equality
                new_runner = replace(
                    runner, test_problem=BoTorchTestProblem(botorch_problem=Ackley())
                )
                self.assertNotEqual(runner, new_runner)

                self.assertEqual(runner, runner)
                if isinstance(test_problem, BoTorchTestProblem):
                    self.assertEqual(
                        test_problem.botorch_problem.bounds.dtype, torch.double
                    )

            with self.subTest(f"test `get_Y_true()`, {test_description}"):
                dim = 6 if is_botorch else 9
                X = torch.rand(1, dim, dtype=torch.double)
                param_names = (
                    [f"x{i}" for i in range(6)]
                    if is_botorch
                    else list(
                        get_jenatton_benchmark_problem().search_space.parameters.keys()
                    )
                )
                params = dict(zip(param_names, (x.item() for x in X.unbind(-1))))

                with (
                    nullcontext()
                    if not isinstance(test_problem, SurrogateTestFunction)
                    else patch.object(
                        # pyre-fixme: ParamBasedTestProblem` has no attribute
                        # `_surrogate`.
                        runner.test_problem._surrogate,
                        "predict",
                        return_value=({"objective_0": [4.2]}, None),
                    )
                ):
                    Y = runner.get_Y_true(params=params)
                    oracle = runner.evaluate_oracle(parameters=params)

                if (
                    isinstance(test_problem, BoTorchTestProblem)
                    and test_problem.modified_bounds is not None
                ):
                    X_tf = normalize(
                        X,
                        torch.tensor(
                            test_problem.modified_bounds, dtype=torch.double
                        ).T,
                    )
                else:
                    X_tf = X
                if isinstance(test_problem, BoTorchTestProblem):
                    botorch_problem = test_problem.botorch_problem
                    obj = botorch_problem.evaluate_true(X_tf)
                    if runner.negate:
                        obj = -obj
                    if runner._is_constrained:
                        expected_Y = torch.cat(
                            [
                                obj.view(-1),
                                botorch_problem.evaluate_slack(X_tf).view(-1),
                            ],
                            dim=-1,
                        )
                    else:
                        expected_Y = obj
                elif isinstance(test_problem, SurrogateTestFunction):
                    expected_Y = torch.tensor([4.2], dtype=torch.double)
                else:
                    expected_Y = torch.full(
                        torch.Size([2]), X.pow(2).sum().item(), dtype=torch.double
                    )
                self.assertTrue(torch.allclose(Y, expected_Y))
                self.assertTrue(np.equal(Y.numpy(), oracle).all())

            with self.subTest(f"test `run()`, {test_description}"):
                trial = Mock(spec=Trial)
                # pyre-fixme[6]: Incomptabile parameter type: params is a
                # mutable subtype of the type expected by `Arm`.
                arm = Arm(name="0_0", parameters=params)
                trial.arms = [arm]
                trial.arm = arm
                trial.index = 0

                with (
                    nullcontext()
                    if not isinstance(test_problem, SurrogateTestFunction)
                    else patch.object(
                        # pyre-fixme: ParamBasedTestProblem` has no attribute
                        # `_surrogate`.
                        runner.test_problem._surrogate,
                        "predict",
                        return_value=({"objective_0": [4.2]}, None),
                    )
                ):
                    res = runner.run(trial=trial)
                self.assertEqual({"Ys", "Ystds", "outcome_names"}, res.keys())
                self.assertEqual({"0_0"}, res["Ys"].keys())
                if isinstance(noise_std, float):
                    self.assertEqual(res["Ystds"]["0_0"], [noise_std] * len(Y))
                elif isinstance(noise_std, dict):
                    self.assertEqual(
                        res["Ystds"]["0_0"],
                        [noise_std[k] for k in runner.outcome_names],
                    )
                else:
                    self.assertEqual(res["Ys"]["0_0"], Y.tolist())
                    self.assertEqual(res["Ystds"]["0_0"], [0.0] * len(Y))
                self.assertEqual(res["outcome_names"], outcome_names)

            with self.subTest(f"test `poll_trial_status()`, {test_description}"):
                self.assertEqual(
                    {TrialStatus.COMPLETED: {0}}, runner.poll_trial_status([trial])
                )

            with self.subTest(f"test `serialize_init_args()`, {test_description}"):
                with self.assertRaisesRegex(
                    UnsupportedError, "serialize_init_args is not a supported method"
                ):
                    ParamBasedTestProblemRunner.serialize_init_args(obj=runner)
                with self.assertRaisesRegex(
                    UnsupportedError, "deserialize_init_args is not a supported method"
                ):
                    ParamBasedTestProblemRunner.deserialize_init_args({})

    def test_botorch_test_problem_runner_heterogeneous_noise(self) -> None:
        runner = ParamBasedTestProblemRunner(
            test_problem=BoTorchTestProblem(botorch_problem=ConstrainedHartmann(dim=6)),
            noise_std=0.1,
            constraint_noise_std=0.05,
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
        self.assertSetEqual(set(res.keys()), {"Ys", "Ystds", "outcome_names"})
        self.assertSetEqual(set(res["Ys"].keys()), {"0_0"})
        self.assertEqual(res["Ystds"]["0_0"], [0.1, 0.05])
        self.assertEqual(res["outcome_names"], ["objective", "constraint"])

    def test_unsupported_error(self) -> None:
        test_function = BoTorchTestProblem(botorch_problem=Hartmann(dim=6))
        with self.assertRaisesRegex(
            UnsupportedError, "`evaluate_slack_true` is only supported when"
        ):
            test_function.evaluate_slack_true({"a": 3})

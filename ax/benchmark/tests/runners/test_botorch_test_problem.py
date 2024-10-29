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
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.test_functions.synthetic import Ackley, ConstrainedHartmann, Hartmann
from botorch.utils.transforms import normalize


class TestBoTorchTestProblem(TestCase):
    def setUp(self) -> None:
        super().setUp()
        botorch_base_test_functions = {
            "base Hartmann": Hartmann(dim=6),
            "negated Hartmann": Hartmann(dim=6, negate=True),
            "constrained Hartmann": ConstrainedHartmann(dim=6),
            "negated constrained Hartmann": ConstrainedHartmann(dim=6, negate=True),
        }
        self.botorch_test_problems = {
            k: BoTorchTestProblem(botorch_problem=v)
            for k, v in botorch_base_test_functions.items()
        }

    def test_negation(self) -> None:
        params = {f"x{i}": 0.5 for i in range(6)}
        evaluate_true_results = {
            k: v.evaluate_true(params) for k, v in self.botorch_test_problems.items()
        }
        self.assertEqual(
            evaluate_true_results["base Hartmann"],
            evaluate_true_results["constrained Hartmann"][0],
        )
        self.assertEqual(
            evaluate_true_results["base Hartmann"],
            -evaluate_true_results["negated Hartmann"],
        )
        self.assertEqual(
            evaluate_true_results["negated Hartmann"],
            evaluate_true_results["negated constrained Hartmann"][0],
        )
        self.assertEqual(
            evaluate_true_results["constrained Hartmann"][1],
            evaluate_true_results["negated constrained Hartmann"][1],
        )

    def test_raises_for_botorch_attrs(self) -> None:
        msg = "noise should be set on the `BenchmarkRunner`, not the test function."
        with self.assertRaisesRegex(ValueError, msg):
            BoTorchTestProblem(botorch_problem=Hartmann(dim=6, noise_std=0.1))
        with self.assertRaisesRegex(ValueError, msg):
            BoTorchTestProblem(
                botorch_problem=ConstrainedHartmann(dim=6, constraint_noise_std=0.1)
            )

    def test_tensor_shapes(self) -> None:
        params = {f"x{i}": 0.5 for i in range(6)}
        evaluate_true_results = {
            k: v.evaluate_true(params) for k, v in self.botorch_test_problems.items()
        }
        evaluate_true_results["BraninCurrin"] = BoTorchTestProblem(
            botorch_problem=BraninCurrin()
        ).evaluate_true(params)
        expected_len = {
            "base Hartmann": 1,
            "constrained Hartmann": 2,
            "negated Hartmann": 1,
            "negated constrained Hartmann": 2,
            "BraninCurrin": 2,
        }
        for name, result in evaluate_true_results.items():
            with self.subTest(name=name):
                self.assertEqual(result.dtype, torch.double)
                self.assertEqual(result.shape, torch.Size([expected_len[name]]))


class TestSyntheticRunner(TestCase):
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
                num_outcomes,
            )
            for (test_problem_class, num_outcomes) in (
                (Hartmann, 1),
                (ConstrainedHartmann, 2),
            )
            for modified_bounds, noise_std in product(
                (None, [(0.0, 2.0)] * 6),
                (0.0, [0.1] * num_outcomes),
            )
        ]
        param_based_cases = [
            (
                TestParamBasedTestProblem(dim=6, num_outcomes=num_outcomes),
                noise_std,
                num_outcomes,
            )
            for num_outcomes in (1, 2)
            for noise_std in (0.0, [float(i) for i in range(num_outcomes)])
        ]
        surrogate_cases = [
            (get_soo_surrogate_test_function(lazy=False), noise_std, 1)
            for noise_std in (0.0, 1.0, [0.0], [1.0])
        ]
        for test_problem, noise_std, num_outcomes in (
            botorch_cases + param_based_cases + surrogate_cases
        ):
            # Set up outcome names
            if isinstance(test_problem, BoTorchTestProblem):
                if isinstance(test_problem.botorch_problem, ConstrainedHartmann):
                    outcome_names = ["objective_0", "constraint"]
                else:
                    outcome_names = ["objective_0"]
            elif isinstance(test_problem, TestParamBasedTestProblem):
                outcome_names = [f"objective_{i}" for i in range(num_outcomes)]
            else:  # SurrogateTestFunction
                outcome_names = ["branin"]

            # Set up runner
            runner = ParamBasedTestProblemRunner(
                test_problem=test_problem,
                outcome_names=outcome_names,
                noise_std=noise_std,
            )

            test_description = f"{test_problem=}, {noise_std=}"
            with self.subTest(
                f"Test basic construction, {test_problem=}, {noise_std=}"
            ):
                self.assertIs(runner.test_problem, test_problem)
                self.assertEqual(runner.outcome_names, outcome_names)
                if isinstance(noise_std, list):
                    self.assertEqual(
                        runner.get_noise_stds(),
                        dict(zip(runner.outcome_names, noise_std)),
                    )
                else:  # float
                    self.assertEqual(
                        runner.get_noise_stds(),
                        {name: noise_std for name in runner.outcome_names},
                    )

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

            is_botorch = isinstance(test_problem, BoTorchTestProblem)
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
                        return_value=({"branin": [4.2]}, None),
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
                    if isinstance(botorch_problem, ConstrainedHartmann):
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
                        runner.test_problem._surrogate,
                        "predict",
                        return_value=({"branin": [4.2]}, None),
                    )
                ):
                    res = runner.run(trial=trial)
                self.assertEqual({"Ys", "Ystds", "outcome_names"}, res.keys())
                self.assertEqual({"0_0"}, res["Ys"].keys())

                if isinstance(noise_std, list):
                    self.assertEqual(res["Ystds"]["0_0"], noise_std)
                    if all((n == 0 for n in noise_std)):
                        self.assertEqual(res["Ys"]["0_0"], Y.tolist())
                else:  # float
                    self.assertEqual(res["Ystds"]["0_0"], [noise_std] * len(Y))
                    if noise_std == 0:
                        self.assertEqual(res["Ys"]["0_0"], Y.tolist())
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
        for noise_std in [[0.1, 0.05], {"objective": 0.1, "constraint": 0.05}]:
            runner = ParamBasedTestProblemRunner(
                test_problem=BoTorchTestProblem(
                    botorch_problem=ConstrainedHartmann(dim=6)
                ),
                noise_std=noise_std,
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

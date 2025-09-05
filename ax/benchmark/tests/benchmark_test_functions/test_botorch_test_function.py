#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from dataclasses import replace

import torch
from ax.benchmark.benchmark_test_functions.botorch_test import BoTorchTestFunction
from ax.utils.common.testutils import TestCase
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.test_functions.synthetic import ConstrainedHartmann, Hartmann


class TestBoTorchTestFunction(TestCase):
    def setUp(self) -> None:
        super().setUp()
        botorch_base_test_functions = {
            "base Hartmann": Hartmann(dim=6),
            "negated Hartmann": Hartmann(dim=6, negate=True),
            "constrained Hartmann": ConstrainedHartmann(dim=6),
            "negated constrained Hartmann": ConstrainedHartmann(dim=6, negate=True),
        }
        self.outcome_names = {
            "Hartmann": ["y"],
            "constrained Hartmann": ["y", "c"],
        }
        self.botorch_test_functions = {
            k: BoTorchTestFunction(
                botorch_problem=v,
                outcome_names=self.outcome_names[
                    "constrained Hartmann" if "constrained" in k else "Hartmann"
                ],
            )
            for k, v in botorch_base_test_functions.items()
        }

    def test_negation(self) -> None:
        params = {f"x{i}": 0.5 for i in range(6)}
        evaluate_true_results = {
            k: v.evaluate_true(params) for k, v in self.botorch_test_functions.items()
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
            BoTorchTestFunction(
                botorch_problem=Hartmann(dim=6, noise_std=0.1),
                outcome_names=self.outcome_names["Hartmann"],
            )
        with self.assertRaisesRegex(ValueError, msg):
            BoTorchTestFunction(
                botorch_problem=ConstrainedHartmann(dim=6, constraint_noise_std=0.1),
                outcome_names=self.outcome_names["constrained Hartmann"],
            )

    def test_tensor_shapes(self) -> None:
        params = {f"x{i}": 0.5 for i in range(6)}
        evaluate_true_results = {
            k: v.evaluate_true(params) for k, v in self.botorch_test_functions.items()
        }
        evaluate_true_results["BraninCurrin"] = BoTorchTestFunction(
            botorch_problem=BraninCurrin(), outcome_names=["f1", "f2"]
        ).evaluate_true({"x0": 0.0, "x1": 0.0})
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
                self.assertEqual(result.shape, torch.Size([expected_len[name], 1]))

    def test_input_dimensions(self) -> None:
        test_function = self.botorch_test_functions["base Hartmann"]
        with self.assertRaisesRegex(ValueError, "Expected 6 parameters, got 7"):
            test_function.evaluate_true({f"x{i}": 0.5 for i in range(7)})

        with self.assertRaisesRegex(ValueError, "Expected 7 parameters, got 6"):
            test_function = replace(test_function, dummy_param_names={"ignore me"})
            test_function.evaluate_true({f"x{i}": 0.5 for i in range(6)})

    def test_n_steps_is_one(self) -> None:
        self.assertEqual(self.botorch_test_functions["base Hartmann"].n_steps, 1)

    def test_dummy_dimensions(self) -> None:
        test_function = self.botorch_test_functions["base Hartmann"]
        embedded_test_function = replace(test_function, dummy_param_names={"ignore me"})
        params = {f"x{i}": 0.5 for i in range(6)}
        embedded_params = {**params, "ignore me": 0.5}
        self.assertEqual(
            test_function.evaluate_true(params=params),
            embedded_test_function.evaluate_true(params=embedded_params),
        )

    def test_with_steps(self) -> None:
        ha_params = {f"x{i}": 0.5 for i in range(6)}
        br_params = {"x0": 0.0, "x1": 0.0}
        for name, botorch_problem, outcome_names, params in [
            ("Unconstrained", Hartmann(dim=6), ["y"], ha_params),
            ("Constrained", ConstrainedHartmann(dim=6), ["y", "c"], ha_params),
            ("Moo", BraninCurrin(), ["y1", "y2"], br_params),
        ]:
            for n_steps in [1, 3]:
                with self.subTest(name=name):
                    test_function = BoTorchTestFunction(
                        outcome_names=outcome_names,
                        botorch_problem=botorch_problem,
                        n_steps=n_steps,
                    )
                    self.assertEqual(test_function.n_steps, n_steps)
                    result = test_function.evaluate_true(params=params)
                    self.assertEqual(
                        result.shape, torch.Size([len(outcome_names), n_steps])
                    )
                    if n_steps == 2:
                        # data is simply repeated down the step dimension
                        self.assertEqual(result[0, 0].item(), result[0, 1].item())

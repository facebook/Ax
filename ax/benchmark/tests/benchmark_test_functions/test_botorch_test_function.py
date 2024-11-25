#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


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
        self.botorch_test_problems = {
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
            k: v.evaluate_true(params) for k, v in self.botorch_test_problems.items()
        }
        evaluate_true_results["BraninCurrin"] = BoTorchTestFunction(
            botorch_problem=BraninCurrin(), outcome_names=["f1", "f2"]
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

    def test_n_steps_is_one(self) -> None:
        self.assertEqual(self.botorch_test_problems["base Hartmann"].n_steps, 1)

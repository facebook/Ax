#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import logging

import numpy as np
from ax.models.discrete.full_factorial import FullFactorialGenerator
from ax.utils.common.testutils import TestCase


class FullFactorialGeneratorTest(TestCase):
    def test_FullFactorial(self) -> None:
        generator = FullFactorialGenerator()
        parameter_values = [[1, 2], ["foo", "bar"]]
        generated_points, weights, _ = generator.gen(
            n=-1,
            parameter_values=parameter_values,
            objective_weights=np.ones(1),
        )
        expected_points = [[1, "foo"], [1, "bar"], [2, "foo"], [2, "bar"]]
        self.assertEqual(generated_points, expected_points)
        self.assertEqual(weights, [1 for _ in range(len(expected_points))])

    def test_FullFactorialValidation(self) -> None:
        # Raise error because cardinality exceeds max cardinality
        generator = FullFactorialGenerator(max_cardinality=5, check_cardinality=True)
        parameter_values = [[1, 2], ["foo", "bar"], [True, False]]
        with self.assertRaises(ValueError):
            generated_points, weights, _ = generator.gen(
                n=-1,
                parameter_values=parameter_values,
                objective_weights=np.ones(1),
            )

        # Raise error because n != -1
        generator = FullFactorialGenerator()
        parameter_values = [[1, 2], ["foo", "bar"]]
        with self.assertLogs(
            FullFactorialGenerator.__module__, logging.WARNING
        ) as logger:
            generated_points, weights, _ = generator.gen(
                n=5,
                parameter_values=parameter_values,
                objective_weights=np.ones(1),
            )
            warning_msg = "FullFactorialGenerator will ignore the specified value of n."
            self.assertTrue(
                any(warning_msg in output for output in logger.output),
                logger.output,
            )

    def test_FullFactorialFixedFeatures(self) -> None:
        generator = FullFactorialGenerator(max_cardinality=5, check_cardinality=True)
        parameter_values = [[1, 2], ["foo", "bar"]]
        generated_points, weights, _ = generator.gen(
            n=-1,
            parameter_values=parameter_values,
            objective_weights=np.ones(1),
            fixed_features={1: "foo"},
        )
        expected_points = [[1, "foo"], [2, "foo"]]
        self.assertEqual(generated_points, expected_points)
        self.assertEqual(weights, [1 for _ in range(len(expected_points))])

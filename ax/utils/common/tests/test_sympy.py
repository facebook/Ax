#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.exceptions.core import UserInputError
from ax.utils.common.sympy import extract_coefficient_dict_from_equality
from ax.utils.common.testutils import TestCase
from sympy import Symbol


def _to_str_keys(d: dict[Symbol, float]) -> dict[str, float]:
    return {str(k): v for k, v in d.items()}


class ExtractCoefficientDictFromEqualityTest(TestCase):
    def test_valid_expressions(self) -> None:
        cases = [
            ("x + y == 1.0", {"x": 1.0, "y": 1.0, "1": -1.0}),
            ("2 * x + 3 * y == 5.0", {"x": 2.0, "y": 3.0, "1": -5.0}),
            ("x == 3", {"x": 1.0, "1": -3.0}),
            ("-x + y == 0", {"x": -1.0, "y": 1.0}),
            ("- x + y == 1.0", {"x": -1.0, "y": 1.0, "1": -1.0}),
            ("x + y  ==  1", {"x": 1.0, "y": 1.0, "1": -1.0}),
        ]
        for expr, expected in cases:
            with self.subTest(expr=expr):
                result = _to_str_keys(extract_coefficient_dict_from_equality(expr))
                self.assertEqual(result, expected)

    def test_sanitized_names(self) -> None:
        result = _to_str_keys(
            extract_coefficient_dict_from_equality("foo.bar + foo.baz == 1")
        )
        self.assertEqual(result.pop("1"), -1.0)
        self.assertEqual(len(result), 2)

    def test_error_missing_operator(self) -> None:
        with self.assertRaisesRegex(UserInputError, "=="):
            extract_coefficient_dict_from_equality("x + y <= 1")

    def test_error_multiple_operators(self) -> None:
        with self.assertRaisesRegex(UserInputError, "=="):
            extract_coefficient_dict_from_equality("x == y == 1")

    def test_error_empty_string(self) -> None:
        with self.assertRaisesRegex(UserInputError, "=="):
            extract_coefficient_dict_from_equality("")

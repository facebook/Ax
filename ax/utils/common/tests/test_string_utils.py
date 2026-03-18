#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.utils.common.string_utils import sanitize_name, unsanitize_name
from ax.utils.common.testutils import TestCase


class StringUtilsTest(TestCase):
    def test_sanitize_name(self) -> None:
        self.assertEqual(sanitize_name("foo.bar.baz"), "foo__dot__bar__dot__baz")
        self.assertEqual(
            sanitize_name("foo.bar/11:Baz|qux"),
            "foo__dot__bar__slash__11__colon__Baz__pipe__qux",
        )
        self.assertEqual(
            sanitize_name("foo.bar + 0.1 * baz"), "foo__dot__bar + 0.1 * baz"
        )
        self.assertEqual(
            sanitize_name("~treatment_percent_"), "__tilde__treatment_percent_"
        )
        self.assertEqual(sanitize_name("foo, ~bar"), "foo, __tilde__bar")
        for s in ("foo;", "foo\\", "'foo"):
            with self.assertRaisesRegex(
                ValueError, "has forbidden control characters."
            ):
                sanitize_name(s=s)
        self.assertEqual(sanitize_name(s="foo"), "foo")

    def test_sanitize_name_sympy_conflicts(self) -> None:
        """Test that sanitize_name detects conflicts with sympy's global dict."""
        # Test with 'test' which is a known sympy function
        with self.assertRaisesRegex(
            ValueError,
            r"contains identifiers that conflict with sympy's built-in names",
        ):
            sanitize_name("test + 1")

        # Test with multiple conflicts
        with self.assertRaisesRegex(
            ValueError,
            r"contains identifiers that conflict with sympy's built-in names",
        ):
            sanitize_name("E + I + pi")

        # Test with 'S' which is also a sympy symbol
        with self.assertRaisesRegex(
            ValueError,
            r"contains identifiers that conflict with sympy's built-in names",
        ):
            sanitize_name("S * 2")

        # Test that non-conflicting names still work
        self.assertEqual(sanitize_name("my_test + 1"), "my_test + 1")
        self.assertEqual(
            sanitize_name("testing + another_var"), "testing + another_var"
        )

        # Test that E in scientific notation is not flagged (it's not an identifier)
        self.assertEqual(sanitize_name("1E5 + 2.3E-4"), "1E5 + 2.3E-4")

        # Test that parens and equals are preserved (not sanitized) by default.
        self.assertEqual(
            sanitize_name("sin(1) + cos(2)"),
            "sin(1) + cos(2)",
        )

    def test_sanitize_parens(self) -> None:
        """Test that sanitize_parens=True sanitizes parens in metric names."""
        # Parenthesized suffix with purely identifier content is sanitized.
        self.assertEqual(
            sanitize_name("metric_(p50)", sanitize_parens=True),
            "metric___lparen__p50__rparen__",
        )
        self.assertEqual(
            sanitize_name("score(0_2_5)", sanitize_parens=True),
            "score__lparen__0_2_5__rparen__",
        )
        # Metric name with colons AND parentheses.
        self.assertEqual(
            sanitize_name(
                "scope:sub:metric_(p99)",
                sanitize_parens=True,
            ),
            "scope__colon__sub__colon__metric___lparen__p99__rparen__",
        )
        # Mathematical grouping is NOT sanitized (no preceding identifier).
        self.assertEqual(
            sanitize_name("(a + b) * c", sanitize_parens=True),
            "(a + b) * c",
        )
        # Parens with operator content inside are NOT sanitized.
        self.assertEqual(
            sanitize_name("sin(x + y)", sanitize_parens=True),
            "sin(x + y)",
        )
        # sanitize_parens=False (default) does NOT sanitize parens.
        self.assertEqual(
            sanitize_name("metric_(p50)"),
            "metric_(p50)",
        )

    def test_unsanitize_name_roundtrip(self) -> None:
        """Test that unsanitize_name reverses sanitize_name including parens."""
        names = [
            "foo.bar.baz",
            "foo.bar/11:Baz|qux",
            "~treatment_percent_",
            "metric-name",
        ]
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(unsanitize_name(sanitize_name(name)), name)

        # Round-trip with sanitize_parens=True
        paren_names = [
            "score(0_2_5)",
            "metric_(p50)",
            "scope:sub:metric_(p99)",
        ]
        for name in paren_names:
            with self.subTest(name=name):
                self.assertEqual(
                    unsanitize_name(sanitize_name(name, sanitize_parens=True)),
                    name,
                )

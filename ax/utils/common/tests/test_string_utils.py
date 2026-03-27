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

    def test_sanitize_at_sign(self) -> None:
        """Test that @ in metric names is sanitized to avoid sympy misparse."""
        self.assertEqual(
            sanitize_name("metric@region"),
            "metric__at__region",
        )
        self.assertEqual(
            sanitize_name("scope:sub:metric@region"),
            "scope__colon__sub__colon__metric__at__region",
        )

    def test_unsanitize_name_roundtrip(self) -> None:
        """Test that unsanitize_name reverses sanitize_name including parens."""
        names = [
            "foo.bar.baz",
            "foo.bar/11:Baz|qux",
            "~treatment_percent_",
            "metric-name",
            "metric@region",
            "scope:sub:metric@region",
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

    def test_sanitize_name_colon_space_and_paren_patterns(self) -> None:
        """Test sanitization of colons adjacent to spaces, parens, and combos.

        Covers colon-space, space-before-paren, paren-then-colon, and complex
        production metric names. Each case also verifies the round-trip.
        """
        cases: list[tuple[str, str, bool]] = [
            # (input, expected_sanitized, sanitize_parens)
            # Colon followed by space.
            ("scope: value", "scope__colon____space__value", False),
            ("a: b: c", "a__colon____space__b__colon____space__c", False),
            # Space before paren.
            (
                "Reliability (UserRID)",
                "Reliability__space____lparen__UserRID__rparen__",
                True,
            ),
            # Paren then colon.
            (
                "metric(p50): value",
                "metric__lparen__p50__rparen____colon____space__value",
                True,
            ),
            # Space after close-paren.
            ("end) next", "end)__space__next", False),
            (
                "fn(x) next",
                "fn__lparen__x__rparen____space__next",
                True,
            ),
        ]
        for name, expected, parens in cases:
            with self.subTest(name=name):
                sanitized = sanitize_name(name, sanitize_parens=parens)
                self.assertEqual(sanitized, expected)
                self.assertEqual(unsanitize_name(sanitized), name)

    def test_sanitize_name_complex_metrics(self) -> None:
        """Regression tests for production metric names (T261468156).

        Complex metrics with colons, spaces, parens, and hyphens must
        round-trip and must not contain raw special characters after
        sanitization.
        """
        metrics = [
            "Scope: Long Name (Qualifier): PROD:All: By Interface",
            (
                "Messaging: Threads Message Request TTRC"
                " - Hidden Requests Inbox Reliability"
                " (User RID): PROD:All: By Interface"
            ),
        ]
        for metric in metrics:
            with self.subTest(metric=metric):
                sanitized = sanitize_name(metric, sanitize_parens=True)
                for char in (":", " ", "(", ")"):
                    self.assertNotIn(char, sanitized)
                self.assertEqual(unsanitize_name(sanitized), metric)

    def test_sanitize_name_hyphen_no_colon_is_subtraction(self) -> None:
        """Without colons, ` - ` is left as subtraction (not sanitized)."""
        self.assertEqual(
            sanitize_name("m1 - m2", sanitize_parens=True),
            "m1 - m2",
        )

    def test_sanitize_name_operators_unchanged(self) -> None:
        """Spaces around mathematical operators must NOT be sanitized."""
        cases = [
            ("m1 + m2", "m1 + m2", False),
            ("m1 * m2", "m1 * m2", False),
            ("m1 >= 100", "m1 >= 100", False),
            ("(a + b) * c", "(a + b) * c", True),
        ]
        for name, expected, parens in cases:
            with self.subTest(name=name):
                self.assertEqual(sanitize_name(name, sanitize_parens=parens), expected)

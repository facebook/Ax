#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.utils.common.string_utils import sanitize_name
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
        for s in ("foo;", "foo\\", "'foo"):
            with self.assertRaisesRegex(
                ValueError, "has forbidden control characters."
            ):
                sanitize_name(s=s)
        self.assertEqual(sanitize_name(s="foo"), "foo")

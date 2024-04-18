#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.utils.common.testutils import TestCase
from ax.utils.common.validation import is_valid_name


class ValidationTest(TestCase):

    def test_is_valid_name(self) -> None:
        self.assertTrue(is_valid_name("foo"))
        self.assertTrue(is_valid_name("_foo"))
        self.assertTrue(is_valid_name("foo1"))
        self.assertTrue(is_valid_name("foo_bar"))
        self.assertTrue(is_valid_name("foo:bar"))

        self.assertFalse(is_valid_name(""))
        self.assertFalse(is_valid_name("1foo"))
        self.assertFalse(is_valid_name("foo bar"))
        self.assertFalse(is_valid_name("foo/bar"))

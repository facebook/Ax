#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import doctest
import unittest

from ax.utils.common import testutils
from ax.utils.testing.manifest import ModuleInfo, populate_test_class


def run_doctests(t: unittest.TestCase, m: ModuleInfo) -> None:
    results = doctest.testmod(m.module, optionflags=doctest.ELLIPSIS)
    assert results.failed == 0


@populate_test_class(run_doctests)
class TestDocTests(testutils.TestCase):
    """
    Run all the doctests in the main library.

    This is a support file for `ae_unittest`.
    """

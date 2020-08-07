#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import doctest
import unittest

from ax.fb.utils.testing.manifest import ModuleInfo, populate_test_class
from ax.utils.common import testutils


def run_doctests(t: unittest.TestCase, m: ModuleInfo) -> None:
    results = doctest.testmod(m.module, optionflags=doctest.ELLIPSIS)
    assert results.failed == 0


@populate_test_class(run_doctests)
class TestDocTests(testutils.TestCase):
    """
    Run all the doctests in the main library.

    This is a support file for `ae_unittest`.
    """

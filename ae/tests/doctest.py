#!/usr/bin/env python3
import doctest
import unittest

from ae.lazarus.ae.tests.manifest import ModuleInfo, populate_test_class
from ae.lazarus.ae.utils.common import testutils


def run_doctests(t: unittest.TestCase, m: ModuleInfo) -> None:
    results = doctest.testmod(m.module, optionflags=doctest.ELLIPSIS)
    assert results.failed == 0


@populate_test_class(run_doctests)
class TestDocTests(testutils.TestCase):
    """
    Run all the doctests in the main library.

    This is a support file for `ae_unittest`.
    """

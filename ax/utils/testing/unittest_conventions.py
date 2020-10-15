#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import pathlib
import sys
import unittest

import __test_modules__
from ax.utils.common import testutils


def get_all_subclasses(cls):
    """Reccursively get all the subclasses of cls"""
    for x in cls.__subclasses__():  # subclasses only contains direct decendants
        yield x
        yield from get_all_subclasses(x)


class TestUnittestConventions(testutils.TestCase):
    def test_uses_ae_unittest(self):
        """Check that all of our tests are inheriting from our own base class

        Our base class does a bit more (like making sure we don't use any of python's
        deprecated `assert` functions) so we want to enforce its usage everywhere.
        """
        test_modules = set(__test_modules__.TEST_MODULES)
        # Make sure everything is loaded
        for m in test_modules:
            importlib.import_module(m)
        test_cases = [
            cls
            for cls in get_all_subclasses(unittest.TestCase)
            if cls.__module__ in test_modules
        ]
        base = testutils.TestCase
        for t in test_cases:
            with self.subTest(t.__name__):
                if not issubclass(t, base):
                    abs_path = pathlib.Path(sys.modules[t.__module__].__file__)
                    root = pathlib.Path(__test_modules__.__file__).parent
                    filename = abs_path.relative_to(root)
                    self.fail(
                        f"in {filename}: {t.__qualname__} should inherit from "
                        f"{base.__module__}.{base.__name__}"
                    )

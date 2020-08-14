#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import unittest

from ax.utils.common import testutils
from ax.utils.testing.manifest import ModuleInfo, populate_test_class


def test_pyre_strict(t: unittest.TestCase, m: ModuleInfo) -> None:
    with open(m.file) as fd:
        for line in fd:
            if line == "# pyre-strict\n" or line == "# no-strict-types\n":
                return
    raise Exception(f"{m.path}'s header should contain '# pyre-strict'")


@populate_test_class(test_pyre_strict)
class TestPyreStrict(testutils.TestCase):
    """
    Test that all the files start are marked pyre strict.
    """

    pass

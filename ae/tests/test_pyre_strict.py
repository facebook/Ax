#!/usr/bin/env python3

import unittest

from ae.lazarus.ae.tests.manifest import ModuleInfo, populate_test_class
from ae.lazarus.ae.utils.common import testutils


def check_pyre_strict(t: unittest.TestCase, m: ModuleInfo) -> None:
    with open(m.file) as fd:
        for line in fd:
            if line == "# pyre-strict\n":
                return
    raise Exception(f"{m.path}'s header should contain '# pyre-strict'")


@populate_test_class(check_pyre_strict)
class TestPyreStrict(testutils.TestCase):
    """
    Test that all the files start are marked pyre strict.
    """

    pass

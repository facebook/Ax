#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from ax.utils.common.testutils import TestCase


class InitTest(TestCase):
    def testInitFiles(self) -> None:
        for root, _dirs, files in os.walk("./ax/ax", topdown=False):
            self.assertTrue(
                "__init__.py" in files,
                "directory " + root + " does not contain a .__init__.py file",
            )

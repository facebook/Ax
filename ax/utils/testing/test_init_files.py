#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from glob import glob

from ax.utils.common.testutils import TestCase


class InitTest(TestCase):
    def testInitFiles(self) -> None:
        """__init__.py files are necessary when not using buck targets"""
        for root, _dirs, files in os.walk("./ax/ax", topdown=False):
            if len(glob(f"{root}/**/*.py", recursive=True)) > 0:
                with self.subTest(root):
                    self.assertTrue(
                        "__init__.py" in files,
                        "directory " + root + " does not contain a .__init__.py file",
                    )

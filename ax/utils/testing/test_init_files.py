#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
from glob import glob

from ax.utils.common.testutils import TestCase

DIRS_TO_SKIP = ["ax/fb", "ax/github", "tests"]


class InitTest(TestCase):
    def test_InitFiles(self) -> None:
        """__init__.py files are necessary for the inclusion of the directories
        in pip builds."""
        for root, _, files in os.walk("./ax", topdown=False):
            if any(s in root for s in DIRS_TO_SKIP):
                continue
            if len(glob(f"{root}/**/*.py", recursive=True)) > 0:
                with self.subTest(root):
                    self.assertTrue(
                        "__init__.py" in files,
                        "directory " + root + " does not contain a .__init__.py file",
                    )

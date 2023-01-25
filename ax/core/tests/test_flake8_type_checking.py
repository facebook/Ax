# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import glob
import importlib
from types import ModuleType

from unittest import TestCase

# determinator is otherwise complaining about importing flake8_type_checking
checker: ModuleType = importlib.import_module("flake8_type_checking.checker")


class Flake8TypeCheckingTestCase(TestCase):
    GLOB = "ax/**/*.py"
    IGNORE = ["TC101", "TC100", "TC200", "TC201", "TC006"]

    def test_type_checking_imports(self) -> None:
        errors = []
        for file_path in glob.glob(self.GLOB):
            with open(file_path, "r") as f:
                code = f.read()
                tree = ast.parse(code)
                visitor = checker.TypingOnlyImportsChecker(tree, options=None)
                errors.extend(
                    [f"{file_path}:{e[0]}:{e[1]}: {e[2]}" for e in visitor.errors]
                )

        errors = [
            error for error in errors if not any(ig in error for ig in self.IGNORE)
        ]
        self.assertEqual(len(errors), 0, "\n".join(errors))

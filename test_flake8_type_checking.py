# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import glob

from unittest import TestCase

from flake8_type_checking.checker import TypingOnlyImportsChecker


# # you could replace this with loading the code from the file directly, etc.
# # this is just an example of code loaded as a string
# my_code = """
# from pandas import DataFrame

# foo: DataFrame~
# """

# # Parse the python code as an [AST](https://docs.python.org/3/library/ast.html)
# tree = ast.parse(my_code)

# # Load the checker from the plugin
# options = {
#     "type_checking_strict": False
# }  # if you want to specify options, add them here
# visitor = TypingOnlyImportsChecker(tree, options)

# for e in visitor.errors:
#     print(e)


class Flake8TypeCheckingTestCase(TestCase):
    GLOB = "ax/**/*.py"
    OPTIONS = {
        "ignore": ["TC200"],
        "type-checking-exempt-modules": ["logging"],
    }
    IGNORE = ["TC101", "TC100", "TC200", "TC201", "TC006"]

    def test_type_checking_imports(self) -> None:
        errors = []
        for file_path in glob.glob(self.GLOB):
            with open(file_path, "r") as f:
                code = f.read()
                tree = ast.parse(code)
                visitor = TypingOnlyImportsChecker(tree, self.OPTIONS)
                errors.extend(
                    [f"{file_path}:{e[0]}:{e[1]}: {e[2]}" for e in visitor.errors]
                )

        errors = [
            error for error in errors if not any(ig in error for ig in self.IGNORE)
        ]
        self.assertEqual(len(errors), 0, "\n".join(errors))

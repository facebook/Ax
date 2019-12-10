#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import itertools
from pathlib import Path
from typing import Callable, List, NamedTuple, Type


class Error(NamedTuple):
    lineno: int
    col: int
    message: str
    type: Type


def should_check(filename):
    # Getting options for plugins in flake8 is a bit of a hassle so we just hardcode
    # our conventions.

    path = Path(filename)
    if path.parent.name not in ("tests", "experimental", "flake8_plugins"):
        return True
    with path.open() as fd:
        for line in itertools.islice(fd, 5):
            if line == "# check-docstrings\n":
                return True
    return False


class DocstringChecker:
    """
    A flake8 plug-in that makes sure all public functions have a docstring
    """

    name: str = "docstring checker"
    version: str = "1.0.0"
    fikename: str
    tree: ast.Module

    def __init__(self, tree, filename):
        self.filename = filename
        self.tree = tree

    def run(self):
        if not should_check(self.filename):
            return
        visitor = DocstringCheckerVisitor()
        visitor.visit(self.tree)
        yield from visitor.errors


def is_copy_doc_call(c):
    """Tries to guess if this is a call to the ``copy_doc`` decorator. This is
    a purely syntactic check so if the decorator was aliased as another name]
    or wrapped in another function we will fail.
    """
    if not isinstance(c, ast.Call):
        return False
    func = c.func
    if isinstance(func, ast.Attribute):
        name = func.attr
    elif isinstance(func, ast.Name):
        name = func.id
    else:
        return False
    return name == "copy_doc"


class DocstringCheckerVisitor(ast.NodeVisitor):
    errors: List[Error]

    def __init__(self) -> None:
        self.errors = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.check_A000(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.check_A000(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.ClassDef) -> None:
        self.check_A000(node)
        self.generic_visit(node)

    def check_A000(self, node: ast.AST) -> None:
        if node.name.startswith("_"):
            return
        docstring = ast.get_docstring(node)
        if docstring is None and not any(
            is_copy_doc_call(dec) for dec in node.decorator_list
        ):
            self.errors.append(A000(node))


# Error classes E, C, W and F are used by flake8. T by mypy and B by bugbear
def new_error(errorid: str, msg: str) -> Callable[[ast.AST], Error]:
    full_message = f"{errorid} {msg}"

    def mk_error(node: ast.AST) -> Error:
        return Error(
            lineno=node.lineno,
            col=node.col_offset,
            message=full_message,
            type=DocstringChecker,
        )

    mk_error.__name__ = errorid
    return mk_error


A000 = new_error(
    "A000",
    "Missing docstring. All public classes, functions and methods should have "
    "docstrings (cf https://fburl.com/wiki/wbcrsoeo).",
)

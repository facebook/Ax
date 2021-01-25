#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import unittest
from types import FunctionType
from typing import Set, Type

from ax.utils.common import testutils
from ax.utils.testing.manifest import ModuleInfo, populate_test_class


def fnid(m: ModuleInfo, fn: FunctionType) -> str:
    code = fn.__code__
    return f"{m.path}:{code.co_firstlineno} {fn.__qualname__}"


def check_annotation(m: ModuleInfo, fn: FunctionType) -> None:
    """Check that a function is properly annotated"""
    # local func within `equality_typechecker decorator`; can't be
    # annotated as it breaks sphinx-autodoc-typehints
    if fn.__name__ == "_type_safe_equals":
        return
    # Certain decorators (e.g. dataclass) can set callables without
    # proper annotations. Let's check if the function's source isn't
    # Ax (raised an error) before running through the annotation checks
    try:
        inspect.getsource(fn)
    except OSError:
        return

    sig = inspect.signature(fn)
    untyped_args = []
    for x in sig.parameters.values():
        # This is somewhat of a hack to allow module functions to not have
        # annotations on their first argument
        if x.name == "self" or x.name == "cls":
            continue
        if x.annotation == inspect._empty:
            untyped_args.append(x.name)
    assert untyped_args == [], f"{fnid(m, fn)} untyped arguments {untyped_args!r}"
    assert (
        sig.return_annotation != inspect._empty
    ), f"{fnid(m, fn)} missing return annotation for {fn.__qualname__}"


def _recurse(t: unittest.TestCase, m: ModuleInfo, obj, visited: Set[Type]) -> None:
    if inspect.isclass(obj) and obj not in visited:
        if obj.__module__ != m.module.__name__:
            return
        visited.add(obj)
        for val in obj.__dict__.values():
            _recurse(t, m, val, visited)
    elif inspect.isfunction(obj):
        if obj.__module__ != m.module.__name__:
            return
        with t.subTest(fn=obj.__qualname__):
            check_annotation(m, obj)


def check_fully_annotated(t: unittest.TestCase, m: ModuleInfo) -> None:
    """Check that every function in the module have type annotation"""
    for val in m.module.__dict__.values():
        _recurse(t, m, val, set())


@populate_test_class(check_fully_annotated)
class TestFullyAnnotated(testutils.TestCase):
    """
    Test that all the functions in the modules contain type annotations.
    """

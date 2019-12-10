#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pydoc
from types import FunctionType
from typing import Any, Callable


# https://stackoverflow.com/a/39235373
def named_tuple_to_dict(data: Any) -> Any:
    """Recursively convert NamedTuples to dictionaries."""
    if isinstance(data, dict):
        return {key: named_tuple_to_dict(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [named_tuple_to_dict(value) for value in data]
    elif _is_named_tuple(data):
        return {
            key: named_tuple_to_dict(value) for key, value in data._asdict().items()
        }
    elif isinstance(data, tuple):
        return tuple(named_tuple_to_dict(value) for value in data)
    else:
        return data


# https://stackoverflow.com/a/2166841
def _is_named_tuple(x: Any) -> bool:
    """Return True if x is an instance of NamedTuple."""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False  # pragma nocover
    return all(type(n) == str for n in f)


def callable_to_reference(callable: Callable) -> str:
    """Obtains path to the callable of form <module>.<name>."""
    if not isinstance(callable, (FunctionType, type)):
        raise TypeError(f"Expected to encode function or class, got: {callable}.")
    name = f"{callable.__module__}.{callable.__qualname__}"
    try:
        assert pydoc.locate(name) is callable
        return name
    except Exception as err:
        raise TypeError(
            f"Callable {callable.__qualname__} is not properly exposed in "
            f"{callable.__module__} (exception: {err})."
        )


def callable_from_reference(path: str) -> Callable:
    """Retrieves a callable by its path."""
    return pydoc.locate(path)  # pyre-ignore[7]

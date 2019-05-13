#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any


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

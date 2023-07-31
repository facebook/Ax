#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
import pydoc
from abc import ABC
from types import FunctionType
from typing import Any, Callable, Dict, List, Optional, Type, Union

import pandas as pd


# https://stackoverflow.com/a/39235373
# pyre-fixme[3]: Return annotation cannot be `Any`.
# pyre-fixme[2]: Parameter annotation cannot be `Any`.
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
# pyre-fixme[2]: Parameter annotation cannot be `Any`.
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


# pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
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


# pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
def callable_from_reference(path: str) -> Callable:
    """Retrieves a callable by its path."""
    return pydoc.locate(path)  # pyre-ignore[7]


# TODO: update signature to avoid shadowing python `object` fn.
def serialize_init_args(
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    object: Any,
    exclude_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Given an object, return a dictionary of the arguments that are
    needed by its constructor.
    """
    properties = {}
    exclude_args = ["self", "args", "kwargs"] + (exclude_fields or [])
    signature = inspect.signature(object.__class__.__init__)
    for arg in signature.parameters:
        if arg in exclude_args:
            continue
        try:
            value = getattr(object, arg)
        except AttributeError:
            raise AttributeError(
                f"{object.__class__} is missing a value for {arg}, "
                f"which is needed by its constructor."
            )
        properties[arg] = value
    return properties


# pyre-fixme[24]: Generic type `type` expects 1 type parameter, use `typing.Type` to
#  avoid runtime subscripting errors.
def extract_init_args(args: Dict[str, Any], class_: Type) -> Dict[str, Any]:
    """Given a dictionary, extract the arguments required for the
    given class's constructor.
    """
    init_args = {}
    signature = inspect.signature(class_.__init__)
    for arg, info in signature.parameters.items():
        if arg in ["self", "args", "kwargs"]:
            continue
        value = args.get(arg)
        if value is None:
            # Only necessary to raise an exception if there is no default
            # value for this argument
            if info.default is inspect.Parameter.empty:
                raise ValueError(
                    f"Cannot decode to class {class_} because required argument {arg} "
                    "is missing. If that's not the class you were intending to decode, "
                    "make sure you have updated your metric or runner registries."
                )
            else:
                # Constructor will use default value
                continue
        init_args[arg] = value
    return init_args


class SerializationMixin(ABC):
    @classmethod
    def serialize_init_args(cls, obj: SerializationMixin) -> Dict[str, Any]:
        """Serialize the properties needed to initialize the object.
        Used for storage.
        """
        return serialize_init_args(object=obj)

    @classmethod
    def deserialize_init_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        """Given a dictionary, deserialize the properties needed to initialize the
        object. Used for storage.
        """
        return extract_init_args(args=args, class_=cls)


def dataframe_from_json(value: Union[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Two different input formats are permitted.

    The currently-recommended format, produced by `dataframe_to_dict`, stores
    dtype information as well as DataFrame values. The "legacy" format does not
    store dtypes explicitly and may not exactly recover inputs, especially
    high-precision floats.

    Example:

        Current format:
        >>> object_json = {'values': '{"x":{"0":"0.12341234123412341"}}',
        ... 'dtypes': '{"x":"float64"}'}}
        >>> dataframe_from_json(object_json)

        Legacy format:
        >>> object_json = '{"x":{"0":"0.12341234123412341"}}'
        >>> dataframe_from_json(object_json)
    """

    is_legacy_format = "dtypes" not in value
    if is_legacy_format:
        # Turn off automatic dtype inference so that arm names like "3_1" are
        # not parsed as 31. (As per PEP 515, 3_1 is the same as 31.)
        return pd.read_json(value, dtype=False)

    dtypes = pd.read_json(value["dtypes"], typ="series")
    return pd.read_json(
        value["values"],
        dtype=dict(dtypes),
        # Don't convert columns with names like "timestamp" to timestamps, since
        # `dtypes` may have specified otherwise
        keep_default_dates=False,
        precise_float=True,
    )

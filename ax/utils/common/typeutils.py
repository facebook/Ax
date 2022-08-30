#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
from typeguard import check_type


T = TypeVar("T")
V = TypeVar("V")
K = TypeVar("K")
X = TypeVar("X")
Y = TypeVar("Y")


def not_none(val: Optional[T], message: Optional[str] = None) -> T:
    """
    Unbox an optional type.

    Args:
      val: the value to cast to a non ``None`` type
      message: optional override of the default error message
    Returns:
      V:  ``val`` when ``val`` is not ``None``
    Throws:
      ValueError if ``val`` is ``None``
    """
    if val is None:
        raise ValueError(message or "Argument to `not_none` was None.")
    return val


def checked_cast(typ: Type[T], val: V) -> T:
    """
    Cast a value to a type (with a runtime safety check).

    Returns the value unchanged and checks its type at runtime. This signals to the
    typechecker that the value has the designated type.

    Like `typing.cast`_ ``check_cast`` performs no runtime conversion on its argument,
    but, unlike ``typing.cast``, ``checked_cast`` will throw an error if the value is
    not of the expected type. The type passed as an argument should be a python class.

    Args:
        typ: the type to cast to
        val: the value that we are casting
    Returns:
        the ``val`` argument, unchanged

    .. _typing.cast: https://docs.python.org/3/library/typing.html#typing.cast
    """
    if not isinstance(val, typ):
        raise ValueError(f"Value was not of type {typ}:\n{val}")
    return val


def checked_cast_complex(typ: Type[T], val: V, message: Optional[str] = None) -> T:
    """
    Cast a value to a type (with a runtime safety check).  Used for subscripted generics
    which isinstance cannot run against.

    Returns the value unchanged and checks its type at runtime. This signals to the
    typechecker that the value has the designated type.

    Like `typing.cast`_ ``check_cast`` performs no runtime conversion on its argument,
    but, unlike ``typing.cast``, ``checked_cast`` will throw an error if the value is
    not of the expected type.

    Args:
        typ: the type to cast to
        val: the value that we are casting
        message: message to print on error
    Returns:
        the ``val`` argument casted to typ

    .. _typing.cast: https://docs.python.org/3/library/typing.html#typing.cast
    """
    try:
        check_type("val", val, typ)
        return cast(T, val)
    except TypeError:
        raise ValueError(message or f"Value was not of type {typ}: {val}")


def checked_cast_optional(typ: Type[T], val: Optional[V]) -> Optional[T]:
    """Calls checked_cast only if value is not None."""
    if val is None:
        return val
    return checked_cast(typ, val)


def checked_cast_list(typ: Type[T], old_l: List[V]) -> List[T]:
    """Calls checked_cast on all items in a list."""
    new_l = []
    for val in old_l:
        val = checked_cast(typ, val)
        new_l.append(val)
    return new_l


def checked_cast_dict(
    key_typ: Type[K], value_typ: Type[V], d: Dict[X, Y]
) -> Dict[K, V]:
    """Calls checked_cast on all keys and values in the dictionary."""
    new_dict = {}
    for key, val in d.items():
        val = checked_cast(value_typ, val)
        key = checked_cast(key_typ, key)
        new_dict[key] = val
    return new_dict


# pyre-fixme[34]: `T` isn't present in the function's parameters.
def checked_cast_to_tuple(typ: Tuple[Type[V], ...], val: V) -> T:
    """
    Cast a value to a union of multiple types (with a runtime safety check).
    This function is similar to `checked_cast`, but allows for the type to be
    defined as a tuple of types, in which case the value is cast as a union of
    the types in the tuple.

    Args:
        typ: the tuple of types to cast to
        val: the value that we are casting
    Returns:
        the ``val`` argument, unchanged
    """
    if not isinstance(val, typ):
        raise ValueError(f"Value was not of type {type!r}:\n{val!r}")
    # pyre-fixme[7]: Expected `T` but got `V`.
    return val


# pyre-fixme[3]: Return annotation cannot be `Any`.
# pyre-fixme[2]: Parameter annotation cannot be `Any`.
def numpy_type_to_python_type(value: Any) -> Any:
    """If `value` is a Numpy int or float, coerce to a Python int or float.
    This is necessary because some of our transforms return Numpy values.
    """
    if isinstance(value, np.integer):
        value = int(value)  # pragma: nocover (covered by generator tests)
    if isinstance(value, np.floating):
        value = float(value)  # pragma: nocover  (covered by generator tests)
    return value

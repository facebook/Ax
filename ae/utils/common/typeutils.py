#!/usr/bin/env python3
# pyre-strict
from typing import List, Optional, Type, TypeVar


T = TypeVar("T")
V = TypeVar("V")


def not_none(val: Optional[T]) -> T:
    """
    Unbox an optional type.

    Args:
      val: the value to cast to a non ``None`` type
    Retruns:
      V:  ``val`` when ``val`` is not ``None``
    Throws:
      ValueError if ``val`` is ``None``
    """
    if val is None:
        raise ValueError("Argument to not_none was None")
    return val


def checked_cast(typ: Type[V], val: V) -> T:
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
        raise ValueError(f"Value was not of type {type!r}:\n{val!r}")
    return val


def checked_cast_optional(typ: Type[V], val: Optional[V]) -> Optional[T]:
    """Calls checked_cast only if value is not None."""
    if val is None:
        return val
    return checked_cast(typ, val)


def checked_cast_list(typ: Type[V], l: List[V]) -> List[T]:
    """Calls checked_cast on all items in a list."""
    new_l = []
    for val in l:
        # pyre: Globally accessible variable `val` has type `undefined` but no
        # pyre-fixme[5]: type is specified.
        val = checked_cast(typ, val)
        new_l.append(val)
    return l

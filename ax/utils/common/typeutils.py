# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, TypeVar

from pyre_extensions import assert_is_instance

T = TypeVar("T")
V = TypeVar("V")
K = TypeVar("K")
X = TypeVar("X")
Y = TypeVar("Y")


def assert_is_instance_optional(val: V | None, typ: type[T]) -> T | None:
    """
    Asserts that the value is an instance of the given type if it is not None.

    Args:
        val: the value to check
        typ: the type to check against
    Returns:
        the `val` argument, unchanged
    """
    if val is None:
        return val
    return assert_is_instance(val, typ)


def assert_is_instance_list(old_l: list[V], typ: type[T]) -> list[T]:
    """
    Asserts that all items in a list are instances of the given type.

    Args:
        old_l: the list to check
        typ: the type to check against
    Returns:
        the `old_l` argument, unchanged
    """
    return [assert_is_instance(val, typ) for val in old_l]


def assert_is_instance_dict(
    d: dict[X, Y], key_type: type[K], val_type: type[V]
) -> dict[K, V]:
    """
    Asserts that all keys and values in the dictionary are instances
    of the given classes.

    Args:
        d: the dictionary to check
        key_type: the type to check against for keys
        val_type: the type to check against for values
    Returns:
        the `d` argument, unchanged
    """
    new_dict = {}
    for key, val in d.items():
        key = assert_is_instance(key, key_type)
        val = assert_is_instance(val, val_type)
        new_dict[key] = val
    return new_dict


# pyre-fixme[34]: `T` isn't present in the function's parameters.
def assert_is_instance_of_tuple(val: V, typ: tuple[type[V], ...]) -> T:
    """
    Asserts that a value is an instance of any type in a tuple of types.

    Args:
        typ: the tuple of types to check against
        val: the value that we are checking
    Returns:
        the `val` argument, unchanged
    """
    if not isinstance(val, typ):
        raise TypeError(f"Value was not of any type {typ!r}:\n{val!r}")
    # pyre-fixme[7]: Expected `T` but got `V`.
    return val


# pyre-fixme[2]: Parameter annotation cannot be `Any`.
# pyre-fixme[24]: Generic type `type` expects 1 type parameter, use `typing.Type` to
#  avoid runtime subscripting errors.
def _argparse_type_encoder(arg: Any) -> type:
    """
    Transforms arguments passed to `optimizer_argparse.__call__`
    at runtime to construct the key used for method lookup as
    `tuple(map(arg_transform, args))`.

    This custom arg_transform allow type variables to be passed
    at runtime.
    """
    # Allow type variables to be passed as arguments at runtime
    return arg if isinstance(arg, type) else type(arg)

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar, Union


T = TypeVar("T")
TDecoderRegistry = dict[str, Union[type[T], Callable[..., T]]]
TClassDecoderRegistry = dict[str, Callable[[dict[str, Any]], Any]]


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
    return all(isinstance(n, str) for n in f)


def serialize_init_args(
    obj: Any,
    exclude_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Given an object, return a dictionary of the arguments that are
    needed by its constructor.
    """
    properties = {}
    exclude_args = ["self", "args", "kwargs"] + (exclude_fields or [])
    signature = inspect.signature(obj.__class__.__init__)
    for arg in signature.parameters:
        if arg in exclude_args:
            continue
        try:
            value = getattr(obj, arg)
        except AttributeError:
            raise AttributeError(
                f"{obj.__class__} is missing a value for {arg}, "
                f"which is needed by its constructor."
            )
        properties[arg] = value
    return properties


# pyre-fixme[24]: Generic type `type` expects 1 type parameter, use `typing.Type` to
#  avoid runtime subscripting errors.
def extract_init_args(args: dict[str, Any], class_: type) -> dict[str, Any]:
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


class SerializationMixin:
    """Base class for Ax objects that define their JSON serialization and
    deserialization logic at the class level, e.g. most commonly ``Runner``
    and ``Metric`` subclasses.

    NOTE: Using this class for Ax objects that receive other Ax objects
    as inputs, is recommended only iff the parent object (that would be
    inheriting from this base class) is not enrolled into
    CORE_ENCODER/DECODER_REGISTRY. Inheriting from this mixin with an Ax
    object that is in CORE_ENCODER/DECODER_REGISTRY, will result in a
    circular dependency, so such classes should inplement their encoding
    and decoding logic within the `json_store` module and not on the classes.

    For example, TransitionCriterion take TrialStatus as inputs and are defined
    on the CORE_ENCODER/DECODER_REGISTRY, so TransitionCriterion should not inherit
    from SerializationMixin and should define custom encoding/decoding logic within
    the json_store module.
    """

    @classmethod
    def serialize_init_args(cls, obj: SerializationMixin) -> dict[str, Any]:
        """Serialize the properties needed to initialize the object.
        Used for storage.
        """
        return serialize_init_args(obj=obj)

    @classmethod
    def deserialize_init_args(
        cls,
        args: dict[str, Any],
        decoder_registry: TDecoderRegistry | None = None,
        class_decoder_registry: TClassDecoderRegistry | None = None,
    ) -> dict[str, Any]:
        """Given a dictionary, deserialize the properties needed to initialize the
        object. Used for storage.
        """
        return extract_init_args(args=args, class_=cls)

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from inspect import signature
from typing import Any, Type, TypeVar

import numpy as np
from typeguard import check_type

T = TypeVar("T")
V = TypeVar("V")
K = TypeVar("K")
X = TypeVar("X")
Y = TypeVar("Y")


def version_safe_check_type(argname: str, value: T, expected_type: Type[T]) -> None:
    """Excecute the check_type function if it has the expected signature, otherwise
    warn.  This is done to support newer versions of typeguard with minimal loss
    of functionality for users that have dependency conflicts"""
    # Get the signature of the check_type function
    sig = signature(check_type)
    # Get the parameters of the check_type function
    params = sig.parameters
    # Check if the check_type function has the expected signature
    params = set(params.keys())
    if all(arg in params for arg in ["argname", "value", "expected_type"]):
        check_type(argname, value, expected_type)


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

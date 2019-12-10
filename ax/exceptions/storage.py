#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ax.exceptions.core import AxError


class JSONDecodeError(AxError):
    """Raised when an error occurs during JSON decoding."""

    pass


class JSONEncodeError(AxError):
    """Raised when an error occurs during JSON encoding."""

    pass


class SQADecodeError(AxError):
    """Raised when an error occurs during SQA decoding."""

    pass


class SQAEncodeError(AxError):
    """Raised when an error occurs during SQA encoding."""

    pass


class ImmutabilityError(AxError):
    """Raised when an attempt is made to update an immutable object."""

    pass

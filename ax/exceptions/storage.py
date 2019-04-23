#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


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

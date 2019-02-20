#!/usr/bin/env python3


from ae.lazarus.ae.exceptions.core import AEError


class JSONDecodeError(AEError):
    """Raised when an error occurs during JSON decoding."""

    pass


class JSONEncodeError(AEError):
    """Raised when an error occurs during JSON encoding."""

    pass


class SQADecodeError(AEError):
    """Raised when an error occurs during SQA decoding."""

    pass


class SQAEncodeError(AEError):
    """Raised when an error occurs during SQA encoding."""

    pass


class ImmutabilityError(AEError):
    """Raised when an attempt is made to update an immutable object."""

    pass

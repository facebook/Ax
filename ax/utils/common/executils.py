#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
from logging import Logger
from typing import Any, List, Optional, Tuple, Type


def retry_on_exception(
    exception_types: Optional[Tuple[Type[Exception]]] = None,
    no_retry_on_exception_types: Optional[Tuple[Type[Exception], ...]] = None,
    check_message_contains: Optional[List[str]] = None,
    retries: int = 3,
    suppress_all_errors: bool = False,
    logger: Optional[Logger] = None,
    default_return_on_suppression: Optional[Any] = None,
) -> Optional[Any]:
    """
    A decorator **for instance methods** to be retried on failure.

    Warnings:
    If the variable `check_message_contains` is supplied and the error message
    contains any of the strings provided, the error will be suppressed
    and default value returned.

    If the variable `suppress_all_errors` is supplied and set to True,
    the error will be suppressed  and default value returned.

    Args:
        exception_types: A tuple of exception(s) types to catch in the decorated
            function. If none is provided, baseclass Exception will be used.

        no_retry_on_exception_types: Exception types to consider non-retryable even
            if their supertype appears in `exception_types` or the only exceptions to
            not retry on if no `exception_types` are specified.

        check_message_contains: A list of strings to be provided. If the error
            message contains any one of these messages, the exception will
            be suppressed.

        retries: Number of retries.

        suppress_all_errors: A flag which will allow you to suppress all exceptions
            of the type provided after all retries have been exhausted.

        logger: A handle for the logger to be used.

        default_return_on_suppression: If the error is suppressed, then the default
            value to be returned once all retries are exhausted.
    """

    def func_wrapper(func):
        @functools.wraps(func)
        def actual_wrapper(self, *args, **kwargs):
            retriable_exceptions = exception_types
            if exception_types is None:
                # If no exception type provided, we catch all errors
                retriable_exceptions = (Exception,)
            if not isinstance(retriable_exceptions, tuple):
                raise ValueError("Expected a tuple of exception types.")

            if no_retry_on_exception_types is not None:
                if not isinstance(no_retry_on_exception_types, tuple):
                    raise ValueError(
                        "Expected a tuple of non-retriable exception types."
                    )
                if set(no_retry_on_exception_types).intersection(
                    set(retriable_exceptions)
                ):
                    raise ValueError(
                        "Same exception type cannot appear in both "
                        "`exception_types` and `no_retry_on_exception_types`."
                    )

            suppress_errors = False
            if suppress_all_errors or (
                "suppress_all_errors" in kwargs and kwargs["suppress_all_errors"]
            ):
                # If we are provided with a flag to suppress all errors
                # inside either the function kwargs or the decorator parameters
                suppress_errors = True

            for i in range(retries):
                try:
                    return func(self, *args, **kwargs)
                except no_retry_on_exception_types or ():
                    raise
                except retriable_exceptions as err:  # Exceptions is a tuple.
                    if suppress_errors or i < retries - 1:
                        # We are either explicitly asked to suppress the error
                        # or we have retries left.
                        if logger is not None:
                            logger.exception(err)
                        continue
                    err_msg = getattr(err, "message", repr(err))
                    if check_message_contains is not None and any(
                        message in err_msg for message in check_message_contains
                    ):
                        # In this case, the error is just logged, suppressed and default
                        # value returned
                        if logger is not None:
                            logger.exception(err)
                        continue
                    raise
            # If we are here, it means the retries were finished but
            # The error was somehow suppressed. Hence return the default value provided
            return default_return_on_suppression

        return actual_wrapper

    return func_wrapper

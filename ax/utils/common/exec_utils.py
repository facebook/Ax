#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
from logging import Logger
from typing import Any, List, Optional, Tuple, Type


def retry_on_exception(
    exception_type: Optional[Tuple[Type[Exception]]] = None,
    check_message_contains: Optional[List[str]] = None,
    retries: int = 3,
    suppress_all_errors: bool = False,
    logger: Optional[Logger] = None,
    default_return_on_suppression: Optional[Any] = None,
) -> Optional[Any]:
    """
    A decorator for functions to be retried on failure.

    Warnings:
    If the variable `check_message_contains` is supplied and the error message
    contains any of the strings provided, the error will be suppressed
    and default value returned.

    If the variable `suppress_all_errors` is supplied and set to True,
    the error will be suppressed  and default value returned.

    Args:
        exception_type: A tuple of exception(s) type you want to catch. If none is
            provided, baseclass Exception will be used

        check_message_contains: A list of strings to be provided. If the error
            message contains any one of these messages, the exception will
            be suppressed

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
            exception_to_catch = exception_type
            if exception_type is None:
                # If no exception type provided, we catch all errors
                exception_to_catch = Exception

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
                except exception_to_catch as err:
                    if suppress_errors or i < retries - 1:
                        # We are either explicitly asked to suppress the error
                        # or we have retries left
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

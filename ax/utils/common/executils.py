#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
from logging import Logger
from typing import Any, List, Optional, Tuple, Type


def retry_on_exception(
    exception_types: Optional[Tuple[Type[Exception], ...]] = None,
    no_retry_on_exception_types: Optional[Tuple[Type[Exception], ...]] = None,
    check_message_contains: Optional[List[str]] = None,
    retries: int = 3,
    suppress_all_errors: bool = False,
    logger: Optional[Logger] = None,
    default_return_on_suppression: Optional[Any] = None,
    wrap_error_message_in: Optional[str] = None,
) -> Optional[Any]:
    """
    A decorator for instance methods or standalone functions that makes them
    retry on failure and allows to specify on which types of exceptions the
    function should and should not retry.

    NOTE: If the argument `suppress_all_errors` is supplied and set to True,
    the error will be suppressed  and default value returned.

    Args:
        exception_types: A tuple of exception(s) types to catch in the decorated
            function. If none is provided, baseclass Exception will be used.

        no_retry_on_exception_types: Exception types to consider non-retryable even
            if their supertype appears in `exception_types` or the only exceptions to
            not retry on if no `exception_types` are specified.

        check_message_contains: A list of strings, against which to match error
            messages. If the error message contains any one of these strings,
            the exception will cause a retry. NOTE: This argument works in
            addition to `exception_types`; if those are specified, only the
            specified types of exceptions will be caught and retried on if they
            contain the strings provided as `check_message_contains`.

        retries: Number of retries to perform.

        suppress_all_errors: If true, after all the retries are exhausted, the
            error will still be suppressed and `default_return_on_suppresion`
            will be returned from the function. NOTE: If using this argument,
            the decorated function may not actually get fully executed, if
            it consistently raises an exception.

        logger: A handle for the logger to be used.

        default_return_on_suppression: If the error is suppressed after all the
            retries, then this default value will be returned from the function.
            Defaults to None.

        wrap_error_message_in: If raising the error message after all the retries,
            a string wrapper for the error message (useful for making error
            messages more user-friendly). NOTE: Format of resulting error will be:
            "<wrap_error_message_in>: <original_error_type>: <original_error_msg>",
            with the stack trace of the original message.

    """

    def func_wrapper(func):
        @functools.wraps(func)
        def actual_wrapper(*args, **kwargs):
            retriable_exceptions = exception_types
            if exception_types is None:
                # If no exception type provided, we catch all errors.
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

            # `suppress_all_errors` could be a flag to the underlying function
            # when used on instance methods.
            suppress_errors = suppress_all_errors or (
                "suppress_all_errors" in kwargs and kwargs["suppress_all_errors"]
            )

            for i in range(retries):
                try:
                    return func(*args, **kwargs)
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
                            logger.exception(
                                wrap_error_message_in
                            )  # Automatically logs `err` and its stack trace.
                        continue
                    if not wrap_error_message_in:
                        raise
                    else:
                        msg = (
                            f"{wrap_error_message_in}: {type(err).__name__}: {str(err)}"
                        )
                    raise type(err)(msg).with_traceback(err.__traceback__)
            # If we are here, it means the retries were finished but
            # The error was somehow suppressed. Hence return the default value provided
            return default_return_on_suppression

        return actual_wrapper

    return func_wrapper

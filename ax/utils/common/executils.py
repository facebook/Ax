#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import functools
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import partial
from logging import Logger
from typing import Any, TypeVar


MAX_WAIT_SECONDS: int = 600
T = TypeVar("T")


# pyre-fixme[3]: Return annotation cannot be `Any`.
def retry_on_exception(
    exception_types: tuple[type[Exception], ...] | None = None,
    no_retry_on_exception_types: tuple[type[Exception], ...] | None = None,
    check_message_contains: list[str] | None = None,
    retries: int = 3,
    suppress_all_errors: bool = False,
    logger: Logger | None = None,
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    default_return_on_suppression: Any | None = None,
    wrap_error_message_in: str | None = None,
    initial_wait_seconds: int | None = None,
) -> Any | None:
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

        initial_wait_seconds: Initial length of time to wait between failures,
            doubled after each failure up to a maximum of 10 minutes. If unspecified
            then there is no wait between retries.
    """

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def func_wrapper(func):
        # Depending on whether `func` is async or not, we use a slightly different
        # wrapper; if wrapping an async function, decorator will await it.
        # `async_actual_wrapper` and `actual_wrapper` are almost exactly the same,
        # except that the former is async and awaits the wrapped function.
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            # pyre-fixme[53]: Captured variable `func` is not annotated.
            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            async def async_actual_wrapper(*args, **kwargs):
                (
                    retry_exceptions,
                    no_retry_exceptions,
                    suppress_errors,
                ) = _validate_and_fill_defaults(
                    retry_on_exception_types=exception_types,
                    no_retry_on_exception_types=no_retry_on_exception_types,
                    suppress_errors=suppress_all_errors,
                    **kwargs,
                )

                for i in range(retries):
                    with handle_exceptions_in_retries(
                        no_retry_exceptions=no_retry_exceptions,
                        retry_exceptions=retry_exceptions,
                        suppress_errors=suppress_errors,
                        # pyre-fixme[6]: For 4th param expected `Optional[str]` but
                        #  got `Optional[List[str]]`.
                        check_message_contains=check_message_contains,
                        last_retry=i >= retries - 1,
                        logger=logger,
                        wrap_error_message_in=wrap_error_message_in,
                    ):
                        if i > 0 and initial_wait_seconds is not None:
                            wait_interval = min(
                                MAX_WAIT_SECONDS, initial_wait_seconds * 2 ** (i - 1)
                            )
                            # pyre-fixme[1001]: `asyncio.sleep(wait_interval)` is
                            #  never awaited.
                            asyncio.sleep(wait_interval)
                        return await func(*args, **kwargs)
                # If we are here, it means the retries were finished but
                # The error was suppressed. Hence return the default value provided.
                return default_return_on_suppression

            return async_actual_wrapper

        @functools.wraps(func)
        # pyre-fixme[53]: Captured variable `func` is not annotated.
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def actual_wrapper(*args, **kwargs):
            (
                retry_exceptions,
                no_retry_exceptions,
                suppress_errors,
            ) = _validate_and_fill_defaults(
                retry_on_exception_types=exception_types,
                no_retry_on_exception_types=no_retry_on_exception_types,
                suppress_errors=suppress_all_errors,
                **kwargs,
            )

            for i in range(retries):
                with handle_exceptions_in_retries(
                    no_retry_exceptions=no_retry_exceptions,
                    retry_exceptions=retry_exceptions,
                    suppress_errors=suppress_errors,
                    # pyre-fixme[6]: For 4th param expected `Optional[str]` but got
                    #  `Optional[List[str]]`.
                    check_message_contains=check_message_contains,
                    last_retry=i >= retries - 1,
                    logger=logger,
                    wrap_error_message_in=wrap_error_message_in,
                ):
                    if i > 0 and initial_wait_seconds is not None:
                        wait_interval = min(
                            MAX_WAIT_SECONDS, initial_wait_seconds * 2 ** (i - 1)
                        )
                        time.sleep(wait_interval)
                    return func(*args, **kwargs)

            # If we are here, it means the retries were finished but
            # The error was suppressed. Hence return the default value provided.
            return default_return_on_suppression

        return actual_wrapper

    return func_wrapper


@contextmanager
def handle_exceptions_in_retries(
    no_retry_exceptions: tuple[type[Exception], ...],
    retry_exceptions: tuple[type[Exception], ...],
    suppress_errors: bool,
    check_message_contains: str | None,
    last_retry: bool,
    logger: Logger | None,
    wrap_error_message_in: str | None,
) -> Generator[None, None, None]:
    try:
        yield  # Perform action within the context manager.
    except no_retry_exceptions:
        raise
    except retry_exceptions as err:  # Exceptions is a tuple.
        err_msg = getattr(err, "message", repr(err))

        if not last_retry or suppress_errors:
            # We are either explicitly asked to suppress the error
            # or we have retries left.
            if logger is not None:
                # `logger.exception` automatically logs `err` and its stack trace.
                logger.exception(err)

        elif (
            not last_retry
            and check_message_contains is not None
            and any(message in err_msg for message in check_message_contains)
        ):
            # In this case, the error is just logged, suppressed and default
            # value returned
            if logger is not None:
                logger.exception(wrap_error_message_in)

        elif not wrap_error_message_in:
            raise

        else:
            msg = f"{wrap_error_message_in}: {type(err).__name__}: {str(err)}"
            raise type(err)(msg).with_traceback(err.__traceback__)


def _validate_and_fill_defaults(
    retry_on_exception_types: tuple[type[Exception], ...] | None,
    no_retry_on_exception_types: tuple[type[Exception], ...] | None,
    suppress_errors: bool,
    **kwargs: Any,
) -> tuple[tuple[type[Exception], ...], tuple[type[Exception], ...], bool]:
    if retry_on_exception_types is None:
        # If no exception type provided, we catch all errors.
        retry_on_exception_types = (Exception,)
    if not isinstance(retry_on_exception_types, tuple):
        raise ValueError("Expected a tuple of exception types.")

    if no_retry_on_exception_types is not None:
        if not isinstance(no_retry_on_exception_types, tuple):
            raise ValueError("Expected a tuple of non-retriable exception types.")
        if set(no_retry_on_exception_types).intersection(set(retry_on_exception_types)):
            raise ValueError(
                "Same exception type cannot appear in both "
                "`exception_types` and `no_retry_on_exception_types`."
            )

    # `suppress_all_errors` could be a flag to the underlying function
    # when used on instance methods.
    suppress_errors = suppress_errors or kwargs.get("suppress_all_errors", False)
    return retry_on_exception_types, no_retry_on_exception_types or (), suppress_errors


def execute_with_timeout(partial_func: Callable[..., T], timeout: float) -> T:
    """Execute a function in a thread that we can abandon if it takes too long.
    The thread cannot actually be terminated, so the process will keep executing
    after timeout, but not on the main thread.

    Args:
        partial_func: A partial function to execute.  This should either be a
            function that takes no arguments, or a functools.partial function
            with all arguments bound.
        timeout: The timeout in seconds.

    Returns:
        The return value of the partial function when called.
    """
    # since threads cannot return values or raise exceptions in the main thread,
    # we pass it a context dict and have it update it with the return value or
    # exception.
    context_dict = {}

    def execute_partial_with_context(context: dict[str, Any]) -> None:
        """Execute the partial function and update the context dict, which
        will either store the result or exception because threads cannot
        directly return results.

        NOTE: There is a bug specifically when using this with sqlite that the
        database is not initialized and saving within the thread will not
        work if the database connection is initialized outside.
        """
        try:
            context["return_value"] = partial_func()
        except Exception as e:
            context["exception"] = e

    thread = threading.Thread(
        target=partial(execute_partial_with_context, context_dict)
    )
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError("Function timed out")
    if "exception" in context_dict:
        raise context_dict["exception"]
    return context_dict["return_value"]

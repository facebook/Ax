#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
from functools import wraps
from typing import Any, Callable, TypeVar

AX_ROOT_LOGGER_NAME = "ax"
DEFAULT_LOG_LEVEL: int = logging.INFO
T = TypeVar("T")


class AxOutputNameFilter(logging.Filter):
    """This is a filter which sets the record's output_name, if
    not configured
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "output_name"):
            # pyre-ignore[16]: Record supports arbitrary attributes
            record.output_name = record.name
        return True


def get_logger(name: str) -> logging.Logger:
    """Get an Axlogger.

    To set a human-readable "output_name" that appears in logger outputs,
    add `{"output_name": "[MY_OUTPUT_NAME]"}` to the logger's contextual
    information. By default, we use the logger's `name`

    NOTE: To change the log level on particular outputs (e.g. STDERR logs),
    set the proper log level on the relevant handler, instead of the logger
    e.g. logger.handers[0].setLevel(INFO)

    Args:
        name: The name of the logger.

    Returns:
        The logging.Logger object.
    """
    logger = logging.getLogger(name)
    logger.addFilter(AxOutputNameFilter())
    return logger


# pyre-fixme[24]: Generic type `logging.StreamHandler` expects 1 type parameter.
def build_stream_handler(level: int = DEFAULT_LOG_LEVEL) -> logging.StreamHandler:
    """Build the default stream handler used for most Ax logging. Sets
    default level to INFO, instead of WARNING.

    Args:
        level: The log level. By default, sets level to INFO

    Returns:
        A logging.StreamHandler instance
    """
    console = logging.StreamHandler()
    console.setLevel(level=level)
    formatter = _build_stream_formatter()
    console.setFormatter(formatter)
    return console


def build_file_handler(
    filepath: str,
    level: int = DEFAULT_LOG_LEVEL
    # pyre-fixme[24]: Generic type `logging.StreamHandler` expects 1 type parameter.
) -> logging.StreamHandler:
    """Build a file handle that logs entries to the given file, using the
    same formatting as the stream handler.

    Args:
        filepath: Location of the file to log output to. If the file exists, output
            will be appended. If it does not exist, a new file will be created.
        level: The log level. By default, sets level to INFO

    Returns:
        A logging.FileHandler instance
    """
    if os.path.isfile(filepath):
        get_logger(__name__).warning(
            f"Log file ({filepath}) already exists, appending logs."
        )
    logfile = logging.FileHandler(filepath)
    logfile.setLevel(level=level)
    formatter = _build_stream_formatter()
    logfile.setFormatter(formatter)
    return logfile


def _build_stream_formatter() -> logging.Formatter:
    """Default formatter for log messages. Add timestamps to log messages."""
    return logging.Formatter(
        fmt="[%(levelname)s %(asctime)s] %(output_name)s: %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )


# pyre-ignore (ignoring Any in argument and output typing)
def _round_floats_for_logging(item: Any, decimal_places: int = 2) -> Any:
    """Round a number or numbers in a mapping to a given number of decimal places.
    If item or values in dictionary is not a number, returns it as it.
    """
    if isinstance(item, float):
        return round(item, 2)
    elif isinstance(item, dict):
        return {
            k: _round_floats_for_logging(item=v, decimal_places=decimal_places)
            for k, v in item.items()
        }
    elif isinstance(item, list):
        return [
            _round_floats_for_logging(item=i, decimal_places=decimal_places)
            for i in item
        ]
    elif isinstance(item, tuple):
        return tuple(
            _round_floats_for_logging(item=i, decimal_places=decimal_places)
            for i in item
        )
    return item


def set_stderr_log_level(level: int) -> None:
    """Set the log level for stream handler, such that logs of given level
    are printed to STDERR by the root logger
    """
    ROOT_STREAM_HANDLER.setLevel(level)


# pyre-ignore[3] Supports generic callables
def disable_logger(
    name: str, level: int = logging.ERROR
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Disables a specific logger by name (e.g. module path) by setting the
    log level at the given one for the duration of the decorated function's call
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def inner(*args: Any, **kwargs: Any) -> T:
            logger = get_logger(name)
            prev_level = logger.getEffectiveLevel()
            logger.setLevel(level)
            try:
                return func(*args, **kwargs)
            finally:
                logger.setLevel(prev_level)

        return inner

    return decorator


"""Sets up Ax's root logger to not propogate to Python's root logger and
use the default stream handler.
"""
ROOT_LOGGER: logging.Logger = get_logger(AX_ROOT_LOGGER_NAME)
ROOT_LOGGER.propagate = False
# Uses a permissive level on the logger, instead make each
# handler as permissive/restrictive as desired
ROOT_LOGGER.setLevel(logging.DEBUG)
# pyre-fixme[24]: Generic type `logging.StreamHandler` expects 1 type parameter.
ROOT_STREAM_HANDLER: logging.StreamHandler = build_stream_handler()
ROOT_LOGGER.addHandler(ROOT_STREAM_HANDLER)

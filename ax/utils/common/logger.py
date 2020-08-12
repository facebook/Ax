#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
from typing import Any


AX_ROOT_LOGGER = "ax"
DEFAULT_LOG_LEVEL: int = logging.INFO


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

    Args:
        name: The name of the logger.

    Returns:
        The logging.Logger object.
    """
    logger = logging.getLogger(name)
    logger.addFilter(AxOutputNameFilter())
    return logger


def get_root_logger() -> logging.Logger:
    return get_logger(AX_ROOT_LOGGER)


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
    filepath: str, level: int = DEFAULT_LOG_LEVEL
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
    """Default formatter for log messages. Add timestamps to log messages.
    """
    return logging.Formatter(
        fmt="[%(levelname)s %(asctime)s] %(output_name)s: %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )


def init_loggers() -> None:
    """Sets up Ax's root logger to not propogate to Python's root logger and
    use the default stream handler.
    """
    root_logger = get_root_logger()
    root_logger.propagate = False
    root_logger.setLevel(DEFAULT_LOG_LEVEL)
    stream_handler = build_stream_handler()
    root_logger.addHandler(stream_handler)


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


init_loggers()

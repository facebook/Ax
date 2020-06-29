#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
from typing import Any, Optional


def get_logger(
    name: str,
    filepath: Optional[str] = None,
    level: int = logging.INFO,
    output_name: Optional[str] = None,
) -> logging.Logger:
    """Get an Axlogger.

    Sets default level to INFO, instead of WARNING. Adds timestamps to logger messages.

    Args:
        name: The name of the logger.
        filepath: Location of the file to log output to. If the file exists, output
            will be appended. If it does not exist, a new file will be created.
        level: The log level.
        output_name: The name of the logger to appear in the logged output. Useful to
            abbreviate long logger names.

    Returns:
        The logging.Logger object.
    """
    if output_name is None:
        output_name = name
    formatter = logging.Formatter(
        fmt=f"[%(levelname)s %(asctime)s] {output_name}: %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    # Add timestamps to log messages.
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(level=level)
        console.setFormatter(formatter)
        logger.addHandler(console)
        logger.propagate = False
    if filepath is None:
        return logger
    if os.path.isfile(filepath):
        logger.warning(f"Log file ({filepath}) already exists, appending logs.")
    logfile = logging.FileHandler(filepath)
    logfile.setLevel(level=level)
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)
    return logger


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

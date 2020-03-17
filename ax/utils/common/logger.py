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
    name: str, filepath: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """Get an Axlogger.

    Sets default level to INFO, instead of WARNING.
    Adds timestamps to logger messages.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    # Add timestamps to log messages.
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(level=level)
        formatter = logging.Formatter(
            fmt="[%(levelname)s %(asctime)s] %(name)s: %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )
        console.setFormatter(formatter)
        logger.addHandler(console)
        logger.propagate = False
    if filepath is None:
        return logger
    if os.path.isfile(filepath):
        logger.warning(f"Log file ({filepath}) already exists, appending logs.")
    logfile = logging.FileHandler(filepath)
    logfile.setLevel(level=level)
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

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# pyre-strict

import logging


def get_logger(name: str) -> logging.Logger:
    """Get an Axlogger.

    Sets default level to INFO, instead of WARNING.
    Adds timestamps to logger messages.
    """
    logger = logging.getLogger(name)
    if logger.level == 0:
        logger.setLevel(logging.INFO)
    # Add timestamps to log messages.
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(levelname)s %(asctime)s] %(name)s: %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )
        console.setFormatter(formatter)
        logger.addHandler(console)
        logger.propagate = False
    return logger

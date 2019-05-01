#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import random

import numpy as np


def randomized_round(x: float) -> int:
    """Randomized round of x"""
    z = math.floor(x)
    return int(z + float(random.random() <= (x - z)))


def randomized_onehot_round(x: np.ndarray) -> np.ndarray:
    """Randomized rounding of x to a one-hot vector.
    x should be 0 <= x <= 1."""
    if len(x) == 1:
        return np.array([randomized_round(x[0])])
    if sum(x) == 0:
        x = np.ones_like(x)
    w = x / sum(x)
    hot = np.random.choice(len(w), size=1, p=w)[0]
    z = np.zeros_like(x)
    z[hot] = 1
    return z


def strict_onehot_round(x: np.ndarray) -> np.ndarray:
    """Round x to a one-hot vector by selecting the max element.
    Ties broken randomly."""
    if len(x) == 1:
        return np.round(x)
    argmax = x == max(x)
    x[argmax] = 1
    x[~argmax] = 0
    return randomized_onehot_round(x)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from copy import copy
from typing import Set

import numpy as np
from ax.core.parameter_constraint import OrderConstraint
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization


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


def contains_constrained_integer(
    search_space: SearchSpace, transform_parameters: Set[str]
) -> bool:
    """Check if any integer parameters are present in parameter_constraints.

    Order constraints are ignored since strict rounding preserves ordering.
    """
    for constraint in search_space.parameter_constraints:
        if isinstance(constraint, OrderConstraint):
            continue
        constraint_params = set(constraint.constraint_dict.keys())
        if constraint_params.intersection(transform_parameters):
            return True
    return False


def randomized_round_parameters(
    parameters: TParameterization, transform_parameters: Set[str]
) -> TParameterization:
    rounded_parameters = copy(parameters)
    for p_name in transform_parameters:
        # pyre: param is declared to have type `float` but is used as
        # pyre-fixme[9]: type `Optional[typing.Union[bool, float, str]]`.
        param: float = parameters.get(p_name)
        rounded_parameters[p_name] = randomized_round(param)
    return rounded_parameters

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from math import isnan
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from ax.core.observation import ObservationData
from ax.core.parameter import Parameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import RobustSearchSpace, SearchSpace
from scipy.stats import norm


class ClosestLookupDict(dict):
    """A dictionary with numeric keys that looks up the closest key."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._keys = sorted(self.keys())

    def __setitem__(self, key: Number, val: Any) -> None:
        if not isinstance(key, Number):
            raise ValueError("ClosestLookupDict only allows numerical keys.")
        super().__setitem__(key, val)
        ipos = np.searchsorted(self._keys, key)
        self._keys.insert(ipos, key)

    def __getitem__(self, key: Number) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            if not self.keys():
                raise RuntimeError("ClosestLookupDict is empty.")
            ipos = np.searchsorted(self._keys, key)
            if ipos == 0:
                return super().__getitem__(self._keys[0])
            elif ipos == len(self._keys):
                return super().__getitem__(self._keys[-1])
            lkey, rkey = self._keys[ipos - 1 : ipos + 1]
            if np.abs(key - lkey) <= np.abs(key - rkey):  # pyre-ignore [58]
                return super().__getitem__(lkey)
            else:
                return super().__getitem__(rkey)


def get_data(
    observation_data: List[ObservationData], metric_names: Union[List[str], None] = None
) -> Dict[str, List[float]]:
    """Extract all metrics if `metric_names` is None."""
    Ys = defaultdict(list)
    for obsd in observation_data:
        for i, m in enumerate(obsd.metric_names):
            if metric_names is None or m in metric_names:
                Ys[m].append(obsd.means[i])
    return Ys


def match_ci_width_truncated(
    mean: float,
    variance: float,
    transform: Callable[[float], float],
    level: float = 0.95,
    margin: float = 0.001,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    clip_mean: bool = False,
) -> Tuple[float, float]:
    """Estimate a transformed variance using the match ci width method.

    See log_y transform for the original. Here, bounds are forced to lie
    within a [lower_bound, upper_bound] interval after transformation."""
    fac = norm.ppf(1 - (1 - level) / 2)
    d = fac * np.sqrt(variance)
    if clip_mean:
        mean = np.clip(mean, lower_bound + margin, upper_bound - margin)
    right = min(mean + d, upper_bound - margin)
    left = max(mean - d, lower_bound + margin)
    width_asym = transform(right) - transform(left)
    new_mean = transform(mean)
    new_variance = float("nan") if isnan(variance) else (width_asym / 2 / fac) ** 2
    return new_mean, new_variance


def construct_new_search_space(
    search_space: SearchSpace,
    parameters: List[Parameter],
    parameter_constraints: Optional[List[ParameterConstraint]] = None,
) -> SearchSpace:
    """Construct a search space with the transformed arguments.

    If the `search_space` is a `RobustSearchSpace`, this will use its
    environmental variables and distributions, and remove the environmental
    variables from `parameters` before constructing.

    Args:
        parameters: List of transformed parameter objects.
        parameter_constraints: List of parameter constraints.

    Returns:
        The new search space instance.
    """
    new_kwargs: Dict[str, Any] = {
        "parameters": parameters,
        "parameter_constraints": parameter_constraints,
    }
    if isinstance(search_space, RobustSearchSpace):
        env_vars = list(search_space._environmental_variables.values())
        if env_vars:
            # Add environmental variables and remove them from parameters.
            new_kwargs["environmental_variables"] = env_vars
            new_kwargs["parameters"] = [p for p in parameters if p not in env_vars]
        new_kwargs["parameter_distributions"] = search_space.parameter_distributions
        new_kwargs["num_samples"] = search_space.num_samples
    return search_space.__class__(**new_kwargs)

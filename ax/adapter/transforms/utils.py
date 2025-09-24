#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Callable
from numbers import Number
from typing import Any, TYPE_CHECKING

import numpy as np
from ax.adapter.transforms.derelativize import Derelativize
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import Parameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import HierarchicalSearchSpace, RobustSearchSpace, SearchSpace
from ax.exceptions.core import UserInputError
from numpy import typing as npt
from pyre_extensions import none_throws
from scipy.stats import norm

T_MATCH_CI_WIDTH = Callable[[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]]


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


# pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use `typing.Dict`
#  to avoid runtime subscripting errors.
class ClosestLookupDict(dict):
    """A dictionary with numeric keys that looks up the closest key."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # pyre-fixme[4]: Attribute must be annotated.
        self._keys = sorted(self.keys())

    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def __setitem__(self, key: Number, val: Any) -> None:
        if not isinstance(key, Number):
            raise ValueError("ClosestLookupDict only allows numerical keys.")
        super().__setitem__(key, val)
        # pyre-fixme[6]: For 2nd argument expected `Union[bytes, complex, float,
        #  int, generic, str]` but got `Number`.
        ipos = np.searchsorted(self._keys, key)
        self._keys.insert(ipos, key)

    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def __getitem__(self, key: Number) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            if not self.keys():
                raise RuntimeError("ClosestLookupDict is empty.")
            # pyre-fixme[6]: For 2nd argument expected `Union[bytes, complex, float,
            #  int, generic, str]` but got `Number`.
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


def match_ci_width(
    *,
    mean: npt.NDArray,
    sem: npt.NDArray | None,
    variance: npt.NDArray | None,
    transform: Callable[[npt.NDArray], npt.NDArray],
    level: float = 0.95,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Transform the mean and update the sem / variance to match the width of
    the confidence interval. The size of the transformed confidence interval
    will be proportional to the size of the original confidence interval. If the
    mean is doubled after the transform, the size of the confidence interval will
    also be doubled.

    Args:
        mean: The mean of the posterior.
        sem: The standard error of the posterior. Only one of
            `sem` or `variance` should be provided.
        variance: The variance of the posterior. Only one of
            `sem` or `variance` should be provided.
        transform: The transform to apply to the mean.
        level: The confidence level of the confidence interval used to match the width.
        lower_bound: If given, the mean and the ci bounds (computed after first clipping
            the mean) will be clipped to this value before applying the transform.
        upper_bound: If given, the mean and the ci bounds (computed after first clipping
            the mean) will be clipped to this value before applying the transform.

    Returns:
        A tuple of the transformed mean and the new sem or variance, depending on which
        one of them was provided as the input.
    """
    if variance is not None:
        if sem is not None:
            raise UserInputError("Only one of `sem` or `variance` should be provided.")
        sem = np.sqrt(variance)
    sem = none_throws(sem)
    if lower_bound is not None or upper_bound is not None:
        mean = np.clip(a=mean, a_min=lower_bound, a_max=upper_bound)
    new_mean = transform(mean)
    if np.all(np.isnan(sem)):
        # If SEM is NaN, we don't need to transform it.
        new_sem = sem
    else:
        fac = norm.ppf(1 - (1 - level) / 2)
        d = fac * sem
        right = mean + d
        left = mean - d
        if lower_bound is not None or upper_bound is not None:
            left = np.clip(a=left, a_min=lower_bound, a_max=upper_bound)
            right = np.clip(a=right, a_min=lower_bound, a_max=upper_bound)
        width_asym = transform(right) - transform(left)
        new_sem = width_asym / (2 * fac)
    if variance is not None:
        return new_mean, new_sem**2
    return new_mean, new_sem


def construct_new_search_space(
    search_space: SearchSpace,
    parameters: list[Parameter],
    parameter_constraints: list[ParameterConstraint] | None = None,
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
    new_kwargs: dict[str, Any] = {
        "parameters": parameters,
        "parameter_constraints": parameter_constraints,
    }
    if isinstance(search_space, HierarchicalSearchSpace):
        # Temporarily relax the `requires_root` flag for the new search space. This is
        # fine because this function is typically called during transforms.
        new_kwargs["requires_root"] = False

    if isinstance(search_space, RobustSearchSpace):
        env_vars = list(search_space._environmental_variables.values())
        if env_vars:
            # Add environmental variables and remove them from parameters.
            new_kwargs["environmental_variables"] = env_vars
            new_kwargs["parameters"] = [p for p in parameters if p not in env_vars]
        new_kwargs["parameter_distributions"] = search_space.parameter_distributions
        new_kwargs["num_samples"] = search_space.num_samples
    return search_space.__class__(**new_kwargs)


def derelativize_optimization_config_with_raw_status_quo(
    optimization_config: OptimizationConfig, adapter: adapter_module.base.Adapter
) -> OptimizationConfig:
    """Derelativize optimization_config using raw status-quo values"""
    tf = Derelativize(
        search_space=adapter.model_space.clone(),
        config={"use_raw_status_quo": True},
    )
    return tf.transform_optimization_config(
        optimization_config=optimization_config.clone(), adapter=adapter
    )

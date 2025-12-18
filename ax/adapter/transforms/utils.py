#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import numpy as np
from ax.adapter.transforms.derelativize import Derelativize
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import Parameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UserInputError
from numpy import typing as npt
from pyre_extensions import none_throws
from scipy.stats import norm

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


T_MATCH_CI_WIDTH = Callable[[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]]

HSS_ERROR_MSG_TEMPLATE = (
    "{name} would encode {p}, which is a hierarchical parameter. This is problematic "
    "as this would lead to losing information about the structure of the search space. "
    "If you do not want to leverage the hierarchical structure, use `Cast` transform "
    "with `flatten_hss=True` (default). Alternatively, you can use a different set of "
    "transforms to retain the hierarchical structure."
)


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

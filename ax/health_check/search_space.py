#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict
from typing import Dict, List, Tuple

from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization

from ax.utils.common.typeutils import checked_cast


def search_space_update_recommendation(
    search_space: SearchSpace,
    parametrizations: List[TParameterization],
    boundary_prop: float = 0.5,
    tol: float = 1e-6,
) -> Tuple[Dict[str, Tuple[float, float]], str]:
    r"""
    Recommendation to update the search space boundaries depending on the
    generated candidates. If most of them fall on some of the boundaries, we
    suggest expanding the search space along those boundaries.

    Note that this function does not handle parameter constraints.
    TODO: Add support for search space update in the presence of
    paramater constraints.

    Args:
        search_space: Search space.
        parametrizations: A list of suggested parametrizations (parameter values).
        boundary_prop: The minimal proportion of suggested parametrizations
            that land at the boundary of the search space for us to recommend
            expanding the search space.
        tol: Tolerance for the difference between parameters and the boundary
            lower and upper bound.

    Returns: A tuple consisting of
        - a dictionary mapping parameter names into the proportion of suggested
            parameterizations that landed on the lower and upper bound of
            that parameter respectively and
        - a human readable string containing the recommendation on the
            search space expansion to expand the boundaries
            where more than a specified proportion of parametrizations land.
    """
    # mapping from range parameter names to the proportions of suggested
    # parametrizations that are at that parameter lower and upper boundary,
    # e.g., {"a": (0.1, 0.2)} means 10% of the suggested parametrizations
    # have parameter "a" value equal to a.lower and 20% have parameter "a" value
    # equal to a.upper.
    param_boundary_prop = defaultdict()
    msg = str()

    num_suggestions = len(parametrizations)

    for parameter_name, parameter in search_space.range_parameters.items():
        lb = 0  # counts how many parameters are equal to the boundary's lower bound
        ub = 0  # counts how many parameters are equal to the boundary's lower bound
        for parametrization in parametrizations:
            value = parametrization[parameter_name]
            value = checked_cast(float, value)
            if abs(value - parameter.lower) < tol:
                lb += 1
            elif abs(value - parameter.upper) < tol:
                ub += 1

        prob_lower = lb / float(num_suggestions)
        prob_upper = ub / float(num_suggestions)
        param_boundary_prop[parameter_name] = (prob_lower, prob_upper)

        if prob_lower >= boundary_prop:
            msg += (
                f"\n Parameter {parameter_name} values are at the lower bound "
                f"({parameter.lower}) of the search space "
                f"for at least {boundary_prop * 100}% of all suggested parameters. "
                "Consider decreasing this lower bound on the search space and "
                "re-generating the candidates inside the expanded search space. "
            )

        if prob_upper >= boundary_prop:
            msg += (
                f"\n Parameter {parameter_name} values are at the upper bound "
                f"({parameter.upper}) of the search space "
                f"for at last {boundary_prop * 100}% of all suggested parameters. "
                "Consider increasing this upper bound on the search space and "
                "re-generating the candidates inside the expanded search space. "
            )

    return param_boundary_prop, msg

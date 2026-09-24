#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Utilities for working with Ax parameters."""

from ax.core.parameter import ChoiceParameter, Parameter, ParameterType, RangeParameter
from ax.exceptions.core import UserInputError


def is_unordered_choice(
    p: Parameter, min_choices: int | None = None, max_choices: int | None = None
) -> bool:
    """Returns whether a parameter is an unordered choice (categorical) parameter.

    You can also specify `min_choices` and `max_choices` to restrict how many
    possible values the parameter can take on.

    Args:
        p: Parameter.
        min_choices: The minimum number of possible values for the parameter.
        max_choices: The maximum number of possible values for the parameter.

    Returns:
        A boolean indicating whether p is an unordered choice parameter or not.
    """
    if min_choices is not None and min_choices < 0:
        raise UserInputError("`min_choices` must be a non-negative integer.")
    if max_choices is not None and max_choices < 0:
        raise UserInputError("`max_choices` must be a non-negative integer.")
    if (
        min_choices is not None
        and max_choices is not None
        and min_choices > max_choices
    ):
        raise UserInputError("`min_choices` cannot be larger than than `max_choices`.")
    return (
        isinstance(p, ChoiceParameter)
        and not p.is_ordered
        and (min_choices is None or min_choices <= len(p.values))
        and (max_choices is None or max_choices >= len(p.values))
    )


def can_map_to_binary(p: Parameter) -> bool:
    """Returns whether a parameter can be transformed to a binary parameter.

    Any choice/range parameters with exactly two values can be transformed to a
    binary parameter.

    Args:
        p: Parameter.

    Returns
        A boolean indicating whether p can be transformed to a binary parameter.
    """
    return (isinstance(p, ChoiceParameter) and len(p.values) == 2) or (
        isinstance(p, RangeParameter)
        and p.parameter_type == ParameterType.INT
        and p.lower == p.upper - 1
    )

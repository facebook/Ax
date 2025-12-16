# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.exceptions.core import UserInputError
from ax.utils.common.string_utils import sanitize_name
from sympy.core.relational import GreaterThan, LessThan
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify


def extract_coefficient_dict_from_inequality(
    inequality_str: str,
) -> dict[Symbol, float]:
    """
    Use SymPy to parse a string into an inequality, invert if necessary to enforce a
    less-than relationship, move all terms to the left side, and return the
    coefficients as a dictionary. This is useful for parsing parameter and outcome
    constraints.
    """
    # Parse the constraint string into a SymPy inequality
    inequality = sympify(sanitize_name(inequality_str))

    # Check the SymPy object is a valid inequality
    if not isinstance(inequality, GreaterThan | LessThan):
        raise UserInputError(f"Expected an inequality, found {inequality_str}")

    # Move all terms to the left side of the inequality and invert if necessary to
    # enforce a less-than relationship
    if isinstance(inequality, LessThan):
        expression = inequality.lhs - inequality.rhs
    else:
        expression = inequality.rhs - inequality.lhs

    return {
        key: float(value) for key, value in expression.as_coefficients_dict().items()
    }

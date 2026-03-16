# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.exceptions.core import UserInputError
from ax.utils.common.string_utils import sanitize_name, unsanitize_name
from sympy.core.expr import Expr
from sympy.core.numbers import Number
from sympy.core.relational import GreaterThan, LessThan
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify, SympifyError


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
    try:
        inequality = sympify(sanitize_name(inequality_str))
    except SympifyError:
        raise UserInputError(f"Expected an inequality, found {inequality_str}")

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


def parse_objective_expression(expression_str: str) -> Expr | tuple[Expr, ...]:
    """Sanitize and sympify an objective expression string.

    For multi-objective expressions (comma-separated), returns a tuple of Expr.
    For single-objective expressions, returns a single Expr.

    Args:
        expression_str: A string representing the objective expression.
            Examples: "m1", "-m1", "2*m1 + m2", "m1, -m2" (multi-objective).

    Returns:
        A single SymPy Expr or a tuple of Expr for multi-objective.

    Raises:
        UserInputError: If the expression string is empty or invalid.
    """
    if len(expression_str) == 0:
        raise UserInputError("Objective expression string must not be empty.")

    sanitized = sanitize_name(expression_str)
    parsed = sympify(sanitized)

    if isinstance(parsed, tuple):
        if any(not isinstance(p, Expr) for p in parsed):
            raise UserInputError(f"Invalid objective expression: {expression_str}")

        return parsed

    if not isinstance(parsed, Expr):
        raise UserInputError(f"Invalid objective expression: {expression_str}")

    return parsed


def extract_metric_names_from_objective_expr(expression: Expr) -> list[str]:
    """Extract unsanitized metric names from a SymPy objective expression.

    Uses ``expression.free_symbols`` to find all referenced metrics and
    unsanitizes their names (reverting SymPy-safe placeholder substitutions).

    Args:
        expression: A SymPy expression representing an objective.

    Returns:
        A sorted list of metric name strings (unsanitized).
    """
    return sorted(unsanitize_name(str(s)) for s in expression.free_symbols)


def extract_metric_weights_from_objective_expr(
    expression: Expr,
) -> list[tuple[str, float]]:
    """Extract (metric_name, weight) pairs from a SymPy objective expression.

    Handles the following cases:
    - Symbol ``x`` -> ``[("x", 1.0)]``
    - Negation ``-x`` -> ``[("x", -1.0)]``
    - Weighted sum ``2*x + y`` -> ``[("x", 2.0), ("y", 1.0)]``

    Uses ``as_coefficients_dict`` to decompose linear expressions.

    Args:
        expression: A SymPy expression representing a (single) objective.

    Returns:
        A list of ``(metric_name, weight)`` tuples sorted by metric name.

    Raises:
        UserInputError: If the expression contains a constant term (a term
            with no metric symbol).
    """
    coeff_dict = expression.as_coefficients_dict()

    result: list[tuple[str, float]] = []
    for term, coeff in coeff_dict.items():
        if isinstance(term, Number) or term.is_number:
            # Constant offset — skip if zero, error if non-zero
            if float(coeff) * float(term) != 0.0:
                raise UserInputError(
                    f"Objective expression '{expression}' contains a constant term. "
                    "Objective expressions must be purely symbolic."
                )
            continue
        # term is a Symbol (possibly multiplied by a coefficient already
        # absorbed into coeff)
        metric_names_in_term = [unsanitize_name(str(s)) for s in term.free_symbols]
        if len(metric_names_in_term) != 1:
            raise UserInputError(
                f"Objective expression '{expression}' contains a non-linear term "
                f"'{term}'. Objective expressions must be linear combinations of "
                "metrics."
            )
        result.append((metric_names_in_term[0], float(coeff)))

    return sorted(result, key=lambda x: x[0])


def _format_number(value: float) -> str:
    """Format a float, dropping the trailing '.0' for whole numbers."""
    return str(int(value)) if value == int(value) else str(value)


def build_constraint_expression_str(
    metric_weights: list[tuple[str, float]],
    op: str,
    bound: float,
    relative: bool,
) -> str:
    """Reconstruct a constraint expression string from components.

    Builds a human-readable inequality string from metric weights, comparison
    operator, bound, and relativity.

    Args:
        metric_weights: List of ``(metric_name, weight)`` tuples.
        op: The comparison operator string, either ``">="`` or ``"<="``.
        bound: The numeric bound.
        relative: Indicates whether the constraint is relative to the status quo.

    Returns:
        A constraint expression string, e.g. ``"qps >= 700"`` or
        ``"loss <= 0.5 * baseline"``.
    """
    parts: list[str] = []
    for name, weight in metric_weights:
        if weight == 1.0:
            parts.append(name)
        elif weight == -1.0:
            parts.append(f"-{name}")
        else:
            parts.append(f"{_format_number(weight)}*{name}")

    lhs = " + ".join(parts).replace(" + -", " - ")

    if relative:
        multiplier = 1 + bound / 100
        if multiplier == 1:
            bound_str = "baseline"
        else:
            bound_str = f"{_format_number(multiplier)} * baseline"
        return f"{lhs} {op} {bound_str}"

    bound_str = _format_number(bound)
    return f"{lhs} {op} {bound_str}"

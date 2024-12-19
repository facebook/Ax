# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import re
from typing import Sequence

from ax.core.map_metric import MapMetric

from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import (
    ComparisonOp,
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.core.parameter_constraint import ParameterConstraint
from ax.exceptions.core import UserInputError
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.relational import GreaterThan, LessThan
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify

DOT_PLACEHOLDER = "__dot__"


def optimization_config_from_string(
    objective_str: str, outcome_constraint_strs: Sequence[str] | None = None
) -> OptimizationConfig:
    """
    Create an OptimizationConfig from objective and outcome constraint strings.

    Note that outcome constraints may not be placed on the objective metric except in
    the multi-objective case where they will be converted to objective thresholds.
    """

    objective = parse_objective(objective_str=objective_str)

    if outcome_constraint_strs is not None:
        outcome_constraints = [
            parse_outcome_constraint(constraint_str=constraint_str)
            for constraint_str in outcome_constraint_strs
        ]
    else:
        outcome_constraints = None

    if isinstance(objective, MultiObjective):
        # Convert OutcomeConstraints to ObjectiveThresholds if relevant
        objective_metric_names = {metric.name for metric in objective.metrics}
        true_outcome_constraints = []
        objective_thresholds: list[ObjectiveThreshold] = []
        for outcome_constraint in outcome_constraints or []:
            if (
                not isinstance(outcome_constraint, ScalarizedOutcomeConstraint)
                and outcome_constraint.metric.name in objective_metric_names
            ):
                objective_thresholds.append(
                    ObjectiveThreshold(
                        metric=outcome_constraint.metric,
                        bound=outcome_constraint.bound,
                        relative=outcome_constraint.relative,
                        op=outcome_constraint.op,
                    )
                )
            else:
                true_outcome_constraints.append(outcome_constraint)

        return MultiObjectiveOptimizationConfig(
            objective=objective,
            outcome_constraints=true_outcome_constraints,
            objective_thresholds=objective_thresholds,
        )

    # Ensure that outcome constraints are not placed on the objective metric
    objective_metric_names = {metric.name for metric in objective.metrics}
    for outcome_constraint in outcome_constraints or []:
        if outcome_constraint.metric.name in objective_metric_names:
            raise UserInputError(
                "Outcome constraints may not be placed on the objective metric "
                f"except in the multi-objective case, found {objective_str} and "
                f"{outcome_constraint_strs}"
            )

    return OptimizationConfig(
        objective=objective,
        outcome_constraints=outcome_constraints,
    )


def parse_parameter_constraint(constraint_str: str) -> ParameterConstraint:
    """
    Parse a parameter constraint string into a ParameterConstraint object using SymPy.
    Currently only supports linear constraints of the form "a * x + b * y >= k" or
    "a * x + b * y <= k".
    """
    coefficient_dict = _extract_coefficient_dict_from_inequality(
        inequality_str=constraint_str
    )

    # Iterate through the coefficients to extract the parameter names and weights and
    # the bound
    constraint_dict = {}
    bound = 0
    for term, coefficient in coefficient_dict.items():
        if term.is_symbol:
            constraint_dict[_unsanitize_dot(term.name)] = coefficient
        elif term.is_number:
            # Invert because we are "moving" the bound to the right hand side
            bound = -1 * coefficient
        else:
            raise UserInputError(
                "Only linear inequality parameter constraints are supported, found "
                f"{constraint_str}"
            )

    return ParameterConstraint(constraint_dict=constraint_dict, bound=bound)


def parse_objective(objective_str: str) -> Objective:
    """
    Parse an objective string into an Objective object using SymPy.

    Currently only supports linear objectives of the form "a * x + b * y" and tuples of
    linear objectives.
    """
    # Parse the objective string into a SymPy expression
    expression = sympify(_sanitize_dot(objective_str))

    if isinstance(expression, tuple):  # Multi-objective
        return MultiObjective(
            objectives=[
                _create_single_objective(expression=term) for term in expression
            ]
        )

    return _create_single_objective(expression=expression)


def parse_outcome_constraint(constraint_str: str) -> OutcomeConstraint:
    """
    Parse an outcome constraint string into an OutcomeConstraint object using SymPy.
    Currently only supports linear constraints of the form "a * x + b * y >= k" or
    "a * x + b * y <= k".

    To indicate a relative constraint (i.e. performance relative to some baseline)
    multiply your bound by "baseline". For example "qps >= 0.95 * baseline" will
    constrain such that the QPS is at least 95% of the baseline arm's QPS.
    """
    coefficient_dict = _extract_coefficient_dict_from_inequality(
        inequality_str=constraint_str
    )

    # Iterate through the coefficients to extract the parameter names and weights and
    # the bound
    constraint_dict: dict[str, float] = {}
    bound = 0
    is_relative = False
    for term, coefficient in coefficient_dict.items():
        if term.is_symbol:
            if term.name == "baseline":
                # Invert because we are "moving" the bound to the right hand side
                bound = -1 * coefficient
                is_relative = True
            else:
                constraint_dict[term.name] = coefficient
        elif term.is_number:
            # Invert because we are "moving" the bound to the right hand side
            bound = -1 * coefficient
        else:
            raise UserInputError(
                "Only linear outcome constraints are supported, found "
                f"{constraint_str}"
            )

    if len(constraint_dict) == 1:
        term, coefficient = next(iter(constraint_dict.items()))

        return OutcomeConstraint(
            metric=MapMetric(name=_unsanitize_dot(term)),
            op=ComparisonOp.LEQ if coefficient > 0 else ComparisonOp.GEQ,
            bound=bound / coefficient,
            relative=is_relative,
        )

    names, coefficients = zip(*constraint_dict.items())
    return ScalarizedOutcomeConstraint(
        metrics=[MapMetric(name=_unsanitize_dot(name)) for name in names],
        op=ComparisonOp.LEQ,
        weights=[*coefficients],
        bound=bound,
        relative=is_relative,
    )


def _create_single_objective(expression: Expr) -> Objective:
    """
    Create an Objective or ScalarizedObjective from a linear SymPy expression.

    All expressions are assumed to represent maximization objectives.
    """

    # If the expression is a just a Symbol it represents a single metric objective
    if isinstance(expression, Symbol):
        return Objective(
            metric=MapMetric(name=_unsanitize_dot(str(expression.name))), minimize=False
        )

    # If the expression is a Mul it likely represents a single metric objective but
    # some additional validation is required
    if isinstance(expression, Mul):
        symbol, *other_symbols = expression.free_symbols
        if len(other_symbols) > 0:
            raise UserInputError(
                f"Only linear objectives are supported, found {expression}."
            )

        # Since the objectives 1 * loss and 2 * loss are equivalent, we can just use
        # the sign from the coefficient rather than its value
        minimize = bool(expression.as_coefficient(symbol) < 0)

        return Objective(
            metric=MapMetric(name=_unsanitize_dot(str(symbol))), minimize=minimize
        )

    # If the expression is an Add it represents a scalarized objective
    elif isinstance(expression, Add):
        names, coefficients = zip(*expression.as_coefficients_dict().items())
        return ScalarizedObjective(
            metrics=[MapMetric(name=_unsanitize_dot(str(name))) for name in names],
            weights=[float(coefficient) for coefficient in coefficients],
            minimize=False,
        )

    raise UserInputError(f"Only linear objectives are supported, found {expression}.")


def _extract_coefficient_dict_from_inequality(
    inequality_str: str,
) -> dict[Symbol, float]:
    """
    Use SymPy to parse a string into an inequality, invert if necessary to enforce a
    less-than relationship, move all terms to the left side, and return the
    coefficients as a dictionary. This is useful for parsing parameter and outcome
    constraints.
    """
    # Parse the constraint string into a SymPy inequality
    inequality = sympify(_sanitize_dot(inequality_str))

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


def _sanitize_dot(s: str) -> str:
    """
    Converts a string with normal dots to a string with sanitized dots. This is
    temporarily necessary because SymPy symbol names must be valid Python identifiers,
    but some legacy Ax users may include dots in their parameter names.
    """
    return re.sub(r"([a-zA-Z])\.([a-zA-Z])", r"\1__dot__\2", s)


def _unsanitize_dot(s: str) -> str:
    """
    Converts a string with sanitized dots back to a string with normal dots. This is
    temporarily necessary because SymPy symbol names must be valid Python identifiers,
    but some legacy Ax users may include dots in their parameter names.
    """
    return re.sub(r"__dot__", ".", s)

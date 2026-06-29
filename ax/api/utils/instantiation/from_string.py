# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Sequence

from ax.core.objective import Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import _parse_constraint_expression, OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.utils.common.sympy import (
    extract_metric_names_from_objective_expr,
    parse_objective_expression,
)


def _identity_mapping_from_objective_str(expr: str) -> dict[str, str]:
    """Extract metric names from an objective expression and build an identity
    mapping. Used as a placeholder until the Experiment attaches real
    signatures."""
    parsed = parse_objective_expression(expr)
    sub_exprs = parsed if isinstance(parsed, tuple) else (parsed,)
    names: list[str] = []
    for se in sub_exprs:
        names.extend(extract_metric_names_from_objective_expr(se))
    return {n: n for n in names}


def _identity_mapping_from_constraint_str(expr: str) -> dict[str, str]:
    """Extract metric names from a constraint expression and build an identity
    mapping."""
    metric_weights, _, _, _ = _parse_constraint_expression(expr)
    return {name: name for name, _ in metric_weights}


def optimization_config_from_string(
    objective_str: str, outcome_constraint_strs: Sequence[str] | None = None
) -> OptimizationConfig:
    """
    Create an OptimizationConfig from objective and outcome constraint strings.

    Note that outcome constraints may not be placed on the objective metric except in
    the multi-objective case where they will be converted to objective thresholds.
    """

    objective = Objective(
        expression=objective_str,
        metric_name_to_signature=_identity_mapping_from_objective_str(objective_str),
    )

    outcome_constraints: list[OutcomeConstraint] | None = (
        [
            OutcomeConstraint(
                expression=s,
                metric_name_to_signature=_identity_mapping_from_constraint_str(s),
            )
            for s in outcome_constraint_strs
        ]
        if outcome_constraint_strs is not None
        else None
    )

    if objective.is_multi_objective:
        # A single-metric constraint on an objective metric becomes an
        # objective threshold only when it bounds the objective from its
        # optimization direction (i.e. an upper bound on a minimized
        # objective or a lower bound on a maximized one). A constraint that
        # bounds against the optimization direction (e.g. ``flops >= 42``
        # while minimizing ``flops``) cannot be expressed as a threshold, so
        # it is kept as a true outcome constraint -- which MOO supports.
        minimize_by_metric_name = {
            name: weight < 0
            for sub_nw in objective._parsed[1]
            for name, weight in sub_nw
        }
        true_outcome_constraints = []
        objective_thresholds: list[OutcomeConstraint] = []
        for outcome_constraint in outcome_constraints or []:
            metric_name = (
                outcome_constraint.metric_names[0]
                if len(outcome_constraint.metric_names) == 1
                else None
            )
            if metric_name is not None and metric_name in minimize_by_metric_name:
                minimize = minimize_by_metric_name[metric_name]
                bounded_above = outcome_constraint.op == ComparisonOp.LEQ
                if minimize == bounded_above:
                    objective_thresholds.append(outcome_constraint)
                else:
                    true_outcome_constraints.append(outcome_constraint)
            else:
                true_outcome_constraints.append(outcome_constraint)

        return MultiObjectiveOptimizationConfig(
            objective=objective,
            outcome_constraints=true_outcome_constraints,
            objective_thresholds=objective_thresholds,
        )

    # Ensure that outcome constraints are not placed on the objective metric
    objective_metric_names = set(objective.metric_names)
    for outcome_constraint in outcome_constraints or []:
        if outcome_constraint.metric_names[0] in objective_metric_names:
            raise UserInputError(
                "Outcome constraints may not be placed on the objective metric "
                f"except in the multi-objective case, found {objective_str} and "
                f"{outcome_constraint_strs}"
            )

    return OptimizationConfig(
        objective=objective,
        outcome_constraints=outcome_constraints,
    )

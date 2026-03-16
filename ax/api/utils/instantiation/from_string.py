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
from ax.core.outcome_constraint import OutcomeConstraint
from ax.exceptions.core import UserInputError


def optimization_config_from_string(
    objective_str: str, outcome_constraint_strs: Sequence[str] | None = None
) -> OptimizationConfig:
    """
    Create an OptimizationConfig from objective and outcome constraint strings.

    Note that outcome constraints may not be placed on the objective metric except in
    the multi-objective case where they will be converted to objective thresholds.
    """

    objective = Objective(expression=objective_str)

    outcome_constraints: list[OutcomeConstraint] | None = (
        [OutcomeConstraint(expression=s) for s in outcome_constraint_strs]
        if outcome_constraint_strs is not None
        else None
    )

    if objective.is_multi_objective:
        # Convert OutcomeConstraints to ObjectiveThresholds if relevant
        objective_metric_names = set(objective.metric_names)
        true_outcome_constraints = []
        objective_thresholds: list[OutcomeConstraint] = []
        for outcome_constraint in outcome_constraints or []:
            if (
                len(outcome_constraint.metric_names) == 1
                and outcome_constraint.metric_names[0] in objective_metric_names
            ):
                objective_thresholds.append(outcome_constraint)
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

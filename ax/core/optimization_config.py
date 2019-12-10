#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from itertools import groupby
from typing import Dict, List, Optional

from ax.core.base import Base
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp


MAX_OBJECTIVES: int = 4
OC_TEMPLATE: str = (
    "OptimizationConfig(objective={objective}, outcome_constraints=[{constraints}])"
)


class OptimizationConfig(Base):
    """An optimization configuration, which comprises an objective
    and outcome constraints.

    There is no minimum or maximum number of outcome constraints, but an
    individual metric can have at most two constraints--which is how we
    represent metrics with both upper and lower bounds.
    """

    def __init__(
        self,
        objective: Objective,
        outcome_constraints: Optional[List[OutcomeConstraint]] = None,
    ) -> None:
        """Inits OptimizationConfig.

        Args:
            objective: Metric+direction to use for the optimization.
            outcome_constraints: Constraints on metrics.
        """
        constraints: List[
            OutcomeConstraint
        ] = [] if outcome_constraints is None else outcome_constraints
        self._validate_optimization_config(
            objective=objective, outcome_constraints=constraints
        )
        self._objective: Objective = objective
        self._outcome_constraints: List[OutcomeConstraint] = constraints

    def clone(self) -> "OptimizationConfig":
        """Make a copy of this optimization config."""
        return OptimizationConfig(
            self.objective.clone(),
            [constraint.clone() for constraint in self.outcome_constraints],
        )

    @property
    def objective(self) -> Objective:
        """Get objective."""
        return self._objective

    @objective.setter
    def objective(self, objective: Objective) -> None:
        """Set objective if not present in outcome constraints."""
        self._validate_optimization_config(objective, self.outcome_constraints)
        self._objective = objective

    @property
    def outcome_constraints(self) -> List[OutcomeConstraint]:
        """Get outcome constraints."""
        return self._outcome_constraints

    @property
    def metrics(self) -> Dict[str, Metric]:
        constraint_metrics = {
            oc.metric.name: oc.metric for oc in self._outcome_constraints
        }
        objective_metrics = {metric.name: metric for metric in self.objective.metrics}
        return {**constraint_metrics, **objective_metrics}

    @outcome_constraints.setter
    def outcome_constraints(self, outcome_constraints: List[OutcomeConstraint]) -> None:
        """Set outcome constraints if valid, else raise."""
        self._validate_optimization_config(
            objective=self.objective, outcome_constraints=outcome_constraints
        )
        self._outcome_constraints = outcome_constraints

    @staticmethod
    def _validate_optimization_config(
        objective: Objective, outcome_constraints: List[OutcomeConstraint]
    ) -> None:
        """Ensure outcome constraints are valid.

        Either one or two outcome constraints can reference one metric.
        If there are two constraints, they must have different 'ops': one
            LEQ and one GEQ.
        If there are two constraints, the bound of the GEQ op must be less
            than the bound of the LEQ op.

        Args:
            outcome_constraints: Constraints to validate.
        """
        constraint_metrics = [
            constraint.metric.name for constraint in outcome_constraints
        ]
        objective_metric_names = [metric.name for metric in objective.metrics]
        for name in objective_metric_names:
            if name in constraint_metrics:
                raise ValueError("Cannot constrain on objective metric.")

        def get_metric_name(oc: OutcomeConstraint) -> str:
            return oc.metric.name

        sorted_constraints = sorted(outcome_constraints, key=get_metric_name)
        for metric_name, constraints_itr in groupby(
            sorted_constraints, get_metric_name
        ):
            constraints: List[OutcomeConstraint] = list(constraints_itr)
            constraints_len = len(constraints)
            if constraints_len == 2:
                if constraints[0].op == constraints[1].op:
                    raise ValueError(f"Duplicate outcome constraints {metric_name}")
                lower_bound_idx = 0 if constraints[0].op == ComparisonOp.GEQ else 1
                upper_bound_idx = 1 - lower_bound_idx
                lower_bound = constraints[lower_bound_idx].bound
                upper_bound = constraints[upper_bound_idx].bound
                if lower_bound >= upper_bound:
                    raise ValueError(
                        f"Lower bound {lower_bound} is >= upper bound "
                        + f"{upper_bound} for {metric_name}"
                    )
            elif constraints_len > 2:
                raise ValueError(f"Duplicate outcome constraints {metric_name}")

    def __repr__(self) -> str:
        return OC_TEMPLATE.format(
            objective=repr(self.objective),
            constraints=", ".join(
                constraint.__repr__() for constraint in self.outcome_constraints
            ),
        )

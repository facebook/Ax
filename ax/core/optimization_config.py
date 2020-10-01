#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union

from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.utils.common.equality import Base
from ax.utils.common.logger import get_logger


logger = get_logger(__name__)

TRefPoint = Dict[str, Tuple[Union[int, float], bool]]

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
        ref_point: Optional[TRefPoint] = None,
    ) -> None:
        """Inits OptimizationConfig.

        Args:
            objective: Metric+direction to use for the optimization.
            outcome_constraints: Constraints on metrics.
            ref_point: A dict from metric_names to (val, is_relative) pairs.
                Ex. {"a": (10, False), "b": (-1.0, True)} gives metric "a"
                an absolute threshold of 10 and metric b a relative threshold
                of -1% worse than the status_quo.
        """
        constraints: List[
            OutcomeConstraint
        ] = [] if outcome_constraints is None else outcome_constraints
        ref_point = ref_point or {}
        ref_constraints = extract_constraints_from_ref_point(
            ref_point=ref_point, objective=objective
        )
        constraints.extend(ref_constraints)
        self._validate_optimization_config(
            objective=objective, outcome_constraints=constraints
        )
        self._objective: Objective = objective
        self._outcome_constraints: List[OutcomeConstraint] = constraints

    def clone(self) -> "OptimizationConfig":
        """Make a copy of this optimization config."""
        return self.clone_with_args()

    def clone_with_args(
        self,
        objective: Optional[Objective] = None,
        outcome_constraints: Optional[List[OutcomeConstraint]] = None,
        ref_point: Optional[TRefPoint] = None,
    ) -> "OptimizationConfig":
        """Make a copy of this optimization config."""
        objective = objective or self.objective.clone()
        outcome_constraints = outcome_constraints or [
            constraint.clone() for constraint in self.outcome_constraints
        ]
        ref_point = ref_point or deepcopy(self.ref_point)
        return OptimizationConfig(
            objective=objective,
            outcome_constraints=outcome_constraints,
            ref_point=ref_point,
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
        all_constraints = self._outcome_constraints
        objective_metric_names = {metric.name for metric in self.objective.metrics}
        non_objective_constraints = [
            c for c in all_constraints if c.metric.name not in objective_metric_names
        ]
        return non_objective_constraints

    @property
    def objective_constraints(self) -> List[OutcomeConstraint]:
        """Get outcome constraints."""
        all_constraints = self._outcome_constraints
        objective_metric_names = {metric.name for metric in self.objective.metrics}
        objective_outcome_constraints = [
            c for c in all_constraints if c.metric.name in objective_metric_names
        ]
        return objective_outcome_constraints

    @property
    def all_constraints(self) -> List[OutcomeConstraint]:
        """Get outcome constraints."""
        return self._outcome_constraints

    @property
    def ref_point(self) -> TRefPoint:
        """Get reference point."""
        ref_point = {}
        for constraint in self.objective_constraints:
            metric_name = constraint.metric.name
            metric_lower_is_better = constraint.metric.lower_is_better
            constraint_bounds_above = constraint.op == ComparisonOp.LEQ
            # Only include constraints that bound in the correct direction.
            if metric_lower_is_better != constraint_bounds_above:
                raise ValueError(
                    make_wrong_direction_warning(
                        metric_name=metric_name,
                        constraint_bounds_above=constraint_bounds_above,
                        metric_lower_is_better=metric_lower_is_better,
                    )
                )
            ref_point[metric_name] = (constraint.bound, constraint.relative)
        return ref_point

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
        unconstrainable_metrics = objective.get_unconstrainable_metrics()
        for metric in unconstrainable_metrics:
            if metric.name in constraint_metrics:
                raise ValueError("Cannot constrain on objective metric.")

        def get_metric_name(oc: OutcomeConstraint) -> str:
            return oc.metric.name

        # Verify we aren't optimizing too many objectives.
        objective_metrics_by_name = {
            metric.name: metric for metric in objective.metrics
        }
        if len(objective_metrics_by_name) > MAX_OBJECTIVES:
            raise ValueError(
                f"Objective: {objective} optimizes more than the maximum allowed "
                f"{MAX_OBJECTIVES} metrics."
            )
        # Warn if constraints on objective_metrics have the wrong direction.
        for constraint in outcome_constraints:
            metric_name = constraint.metric.name
            if metric_name in objective_metrics_by_name:
                metric_lower_is_better = constraint.metric.lower_is_better
                constraint_bounds_above = constraint.op == ComparisonOp.LEQ
                if metric_lower_is_better != constraint_bounds_above:
                    raise ValueError(
                        make_wrong_direction_warning(
                            metric_name=metric_name,
                            constraint_bounds_above=constraint_bounds_above,
                            metric_lower_is_better=metric_lower_is_better,
                        )
                    )

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


def make_wrong_direction_warning(
    metric_name: str,
    constraint_bounds_above: bool,
    metric_lower_is_better: Optional[bool],
) -> str:
    return (
        f"Constraint on {metric_name} bounds from "
        f"{'above' if constraint_bounds_above else 'below'} "
        f"but {metric_name} is being "
        f"{'minimized' if metric_lower_is_better else 'maximized'}."
    ).format(metric_name)


def extract_constraints_from_ref_point(
    ref_point: TRefPoint, objective: Objective
) -> List[OutcomeConstraint]:
    """Extract outcome constraints on objective metrics from a reference point.

    Only metrics in the objective will be constrained.

    Args:
        ref_point: reference point to convert
        objective: objective containing metrics that can be validly constrained.

    Return:
        A list of constraints on objective metrics.
    """
    constraints = []
    objective_metrics_by_name = {metric.name: metric for metric in objective.metrics}
    for metric_name in objective_metrics_by_name.keys():
        if metric_name not in ref_point:
            continue
        bound_config = ref_point[metric_name]
        val = bound_config[0]
        rel = bound_config[1]
        metric = objective_metrics_by_name[metric_name]
        op = ComparisonOp.LEQ if metric.lower_is_better else ComparisonOp.GEQ
        # Add constraint based on ref_point. Any constraint on objectives is
        # treated like  a reference point coordinate.
        constraints.append(OutcomeConstraint(metric, op=op, bound=val, relative=rel))
    for metric_name in ref_point:
        if metric_name not in objective_metrics_by_name:
            logger.warning(
                f"ref_point includes metric_name {metric_name} not present in "
                f"objective metrics {list(objective_metrics_by_name.keys())}"
            )
    return constraints

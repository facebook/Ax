#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.generators.types import TConfig
from ax.utils.common.logger import get_logger
from pyre_extensions import none_throws


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


logger: logging.Logger = get_logger(__name__)


class ObjectiveAsConstraint(Transform):
    """Adds objective metric(s) as absolute outcome constraint(s) when there
    is a status_quo but no feasible points in the data.

    A point is considered feasible if it satisfies all outcome constraints
    AND (for multi-objective optimization with objective thresholds) dominates
    all objective thresholds.

    When no observed points are feasible, this transform adds constraint(s)
    on the objective metric(s) requiring them to be no worse than the status
    quo value(s) (e.g., ``objective >= sq_value`` for maximization, or
    ``objective <= sq_value`` for minimization).

    For single-objective optimization, a single constraint is added on the
    objective metric. For multi-objective optimization without objective
    thresholds, constraints are added on all objective metrics. For MOO with
    objective thresholds, no constraints are added (the thresholds define
    the bounds for feasibility).

    This encourages the optimizer to search for points that are both feasible
    with respect to the original constraints and at least as good as the
    status quo on the objective(s).

    This transform is a no-op if:
      - There is no status_quo.
      - There are no outcome constraints (for SOO) or no outcome constraints
          and no objective thresholds (for MOO).
      - There exist feasible points in the data.
      - For MOO: objective thresholds are specified (they define feasibility).
      - There are relative constraints (Derelativize has not been applied).
    """

    requires_data_for_initialization: bool = True
    no_op_for_experiment_data: bool = True
    _should_add_constraint: bool
    _objective_metrics_added: list[str]
    _scalarized_objective_constraint_added: bool

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )

        self._should_add_constraint = False
        self._objective_metrics_added = []
        self._scalarized_objective_constraint_added = False
        if adapter is None or adapter.status_quo is None:
            return
        if experiment_data is None or experiment_data.observation_data.empty:
            return
        self._should_add_constraint = self._check_no_feasible_points(
            experiment_data=experiment_data,
        )
        if self._should_add_constraint:
            opt_config = adapter._optimization_config
            if isinstance(opt_config, MultiObjectiveOptimizationConfig):
                logger.info(
                    "No feasible points found. Adding objective metrics as "
                    "absolute constraints at the status quo values."
                )
            else:
                logger.info(
                    "No feasible points found. Adding objective metric as an "
                    "absolute constraint at the status quo value."
                )

    def _check_no_feasible_points(
        self,
        experiment_data: ExperimentData,
    ) -> bool:
        """Check if there are no feasible points in the data.

        A point is feasible if it satisfies all outcome constraints AND
        (for multi-objective optimization with thresholds) dominates all
        objective thresholds.

        For MOO with objective thresholds, this method returns False (no-op)
        since the thresholds already define feasibility bounds.

        Returns:
            True if there are no feasible points (and there are constraints
            to check, and for MOO no thresholds are specified), False otherwise.
        """
        adapter = none_throws(self.adapter)
        opt_config = adapter._experiment.optimization_config
        if opt_config is None:
            return False

        outcome_constraints = opt_config.outcome_constraints

        # For MOO with objective thresholds, we don't add constraints.
        # The thresholds already define the bounds for feasibility.
        if isinstance(opt_config, MultiObjectiveOptimizationConfig):
            if len(opt_config.objective_thresholds) > 0:
                return False

        # No-op if there are no constraints to check.
        if len(outcome_constraints) == 0:
            return False

        # Get status quo data for evaluating relative constraints.
        sq_obs = adapter.status_quo
        sq_data = sq_obs.data if sq_obs is not None else None

        observation_data = experiment_data.observation_data

        for _, row in observation_data.iterrows():
            if _is_point_feasible(
                row=row,
                constraints=outcome_constraints,
                sq_data=sq_data,
            ):
                return False

        return True

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        adapter: adapter_module.base.Adapter | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        if not self._should_add_constraint:
            return optimization_config

        # Raise error if there are relative constraints (Derelativize not applied).
        relative_constraints = [
            c for c in optimization_config.outcome_constraints if c.relative
        ]
        if relative_constraints:
            raise ValueError(
                "ObjectiveAsConstraint requires all outcome constraints "
                "to be absolute (non-relative). Found relative constraints: "
                f"{relative_constraints}. Ensure this transform is placed "
                "after Derelativize in the transform pipeline."
            )

        sq_obs = none_throws(
            none_throws(self.adapter).status_quo,
            "Status quo must be set when adding objective as constraint.",
        )
        sq_data = sq_obs.data

        # Get the objective to determine how to add constraints.
        objective = optimization_config.objective

        # Handle ScalarizedObjective: create a single ScalarizedOutcomeConstraint
        # with the bound equal to the status quo value of the scalarized objective.
        if isinstance(objective, ScalarizedObjective):
            scalarized_sq_value = 0.0
            for metric, weight in objective.metric_weights:
                metric_idx = sq_data.metric_signatures.index(metric.signature)
                scalarized_sq_value += weight * sq_data.means[metric_idx]

            op = ComparisonOp.LEQ if objective.minimize else ComparisonOp.GEQ
            new_constraint = ScalarizedOutcomeConstraint(
                metrics=[m.clone() for m in objective.metrics],
                weights=list(objective.weights),
                op=op,
                bound=float(scalarized_sq_value),
                relative=False,
            )
            optimization_config._outcome_constraints.append(new_constraint)
            self._scalarized_objective_constraint_added = True
            return optimization_config

        # Get list of objectives to add constraints for.
        if isinstance(optimization_config, MultiObjectiveOptimizationConfig):
            # MultiObjectiveOptimizationConfig can have MultiObjective or
            # ScalarizedObjective. Only MultiObjective has multiple objectives.
            if isinstance(objective, MultiObjective):
                objectives = objective.objectives
            else:
                objectives = [
                    Objective(metric=objective.metric, minimize=objective.minimize)
                ]
        else:
            objectives = [objective]

        # Add a constraint for each objective at the status quo value.
        for obj in objectives:
            if isinstance(obj, ScalarizedObjective):
                # Create a ScalarizedOutcomeConstraint for scalarized sub-objectives.
                scalarized_sq_value = 0.0
                for metric, weight in obj.metric_weights:
                    metric_idx = sq_data.metric_signatures.index(metric.signature)
                    scalarized_sq_value += weight * sq_data.means[metric_idx]

                op = ComparisonOp.LEQ if obj.minimize else ComparisonOp.GEQ
                new_constraint = ScalarizedOutcomeConstraint(
                    metrics=[m.clone() for m in obj.metrics],
                    weights=list(obj.weights),
                    op=op,
                    bound=float(scalarized_sq_value),
                    relative=False,
                )
                optimization_config._outcome_constraints.append(new_constraint)
                self._scalarized_objective_constraint_added = True
            else:
                metric = obj.metric
                metric_idx = sq_data.metric_signatures.index(metric.signature)
                sq_value = sq_data.means[metric_idx]

                op = ComparisonOp.LEQ if obj.minimize else ComparisonOp.GEQ
                new_constraint = OutcomeConstraint(
                    metric=metric.clone(),
                    op=op,
                    bound=float(sq_value),
                    relative=False,
                )

                optimization_config._outcome_constraints.append(new_constraint)
                self._objective_metrics_added.append(metric.name)

        return optimization_config

    def untransform_outcome_constraints(
        self,
        outcome_constraints: list[OutcomeConstraint],
        fixed_features: ObservationFeatures | None = None,
    ) -> list[OutcomeConstraint]:
        if self._scalarized_objective_constraint_added:
            outcome_constraints = [
                oc
                for oc in outcome_constraints
                if not isinstance(oc, ScalarizedOutcomeConstraint)
            ]
        if self._objective_metrics_added:
            outcome_constraints = [
                oc
                for oc in outcome_constraints
                if oc.metric.name not in self._objective_metrics_added
            ]
        return outcome_constraints


def _is_point_feasible(
    row: pd.Series,
    constraints: list[OutcomeConstraint],
    sq_data: ObservationData | None = None,
) -> bool:
    """Check if a single observation satisfies all outcome constraints.

    Uses the mean values of the observations to check feasibility.

    Args:
        row: A row from the observation_data DataFrame, with multi-indexed
            columns where ("mean", metric_signature) gives the mean value.
        constraints: The outcome constraints to check against. May be
            absolute or relative.
        sq_data: Status quo observation data, required for evaluating
            relative constraints. If None and a relative constraint is
            encountered, the constraint is skipped.

    Returns:
        True if the point satisfies all constraints, False otherwise.
    """
    for constraint in constraints:
        metric_sig = constraint.metric.signature
        try:
            mean_val = row["mean", metric_sig]
        except KeyError:
            continue

        if pd.isna(mean_val):
            continue

        # Compute the effective bound.
        if constraint.relative:
            # Relative constraint: bound is a percentage of the status quo.
            if sq_data is None:
                # Can't evaluate relative constraint without status quo.
                continue
            try:
                sq_idx = sq_data.metric_signatures.index(metric_sig)
                sq_val = sq_data.means[sq_idx]
            except (ValueError, IndexError):
                # Status quo doesn't have this metric.
                continue
            # Relative bound: sq_val * (1 + bound/100) for GEQ,
            # sq_val * (1 + bound/100) for LEQ.
            bound = sq_val * (1 + constraint.bound / 100.0)
        else:
            bound = constraint.bound

        if constraint.op == ComparisonOp.GEQ:
            if mean_val < bound:
                return False
        else:
            if mean_val > bound:
                return False

    return True

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from itertools import groupby

from ax.core.arm import Arm
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.outcome_constraint import (
    ComparisonOp,
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.exceptions.core import UserInputError
from ax.utils.common.base import Base
from pyre_extensions import assert_is_instance


TRefPoint = list[ObjectiveThreshold]

# Sentinels for default arguments when None is a valid input
_NO_OUTCOME_CONSTRAINTS = [
    OutcomeConstraint(Metric("placeholder", lower_is_better=True), ComparisonOp.GEQ, 0)
]
_NO_OBJECTIVE_THRESHOLDS = [
    ObjectiveThreshold(Metric("placeholder", lower_is_better=True), 0)
]

_NO_PRUNING_TARGET_PARAMETERIZATION = Arm(parameters={})


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
        outcome_constraints: list[OutcomeConstraint] | None = None,
        pruning_target_parameterization: Arm | None = None,
    ) -> None:
        """Inits OptimizationConfig.

        Args:
            objective: Metric+direction to use for the optimization.
            outcome_constraints: Constraints on metrics.
            pruning_target_parameterization: Arm containing the target values for
                irrelevant parameters. The target values are used to prune irrelevant
                parameters from candidates generated via Bayesian optimization: when
                Ax considers arms to suggest for the next trial, it will seek to have
                the proposed arms differ from the target arm as little as possible,
                without a loss in optimization performance. I.e. when suggested arms
                include parameter values that differ from the corresponding value in
                the target arm, the pruning methodology will check if that difference
                is expected to be meaningful w.r.t. the performance of the arm in
                consideration, and if not, the parameter value will be replaced with
                the corresponding value in the target arm.
        """
        constraints: list[OutcomeConstraint] = (
            [] if outcome_constraints is None else outcome_constraints
        )
        self._validate_transformed_optimization_config(
            objective=objective,
            outcome_constraints=constraints,
        )
        self._objective: Objective = objective
        self._outcome_constraints: list[OutcomeConstraint] = constraints
        self.pruning_target_parameterization = pruning_target_parameterization

    def clone(self) -> "OptimizationConfig":
        """Make a copy of this optimization config."""
        return self.clone_with_args()

    def clone_with_args(
        self,
        objective: Objective | None = None,
        outcome_constraints: None | (list[OutcomeConstraint]) = _NO_OUTCOME_CONSTRAINTS,
        pruning_target_parameterization: Arm
        | None = _NO_PRUNING_TARGET_PARAMETERIZATION,
    ) -> "OptimizationConfig":
        """Make a copy of this optimization config."""
        objective = self.objective.clone() if objective is None else objective
        outcome_constraints = (
            [constraint.clone() for constraint in self.outcome_constraints]
            if outcome_constraints is _NO_OUTCOME_CONSTRAINTS
            else outcome_constraints
        )
        pruning_target_parameterization = (
            self.pruning_target_parameterization
            if pruning_target_parameterization is _NO_PRUNING_TARGET_PARAMETERIZATION
            else pruning_target_parameterization
        )

        return OptimizationConfig(
            objective=objective,
            outcome_constraints=outcome_constraints,
            pruning_target_parameterization=pruning_target_parameterization,
        )

    @property
    def objective(self) -> Objective:
        """Get objective."""
        return self._objective

    @objective.setter
    def objective(self, objective: Objective) -> None:
        """Set objective if not present in outcome constraints."""
        self._validate_transformed_optimization_config(
            objective, self.outcome_constraints
        )
        self._objective = objective

    @property
    def all_constraints(self) -> list[OutcomeConstraint]:
        """Get outcome constraints."""
        return self.outcome_constraints

    @property
    def outcome_constraints(self) -> list[OutcomeConstraint]:
        """Get outcome constraints."""
        return self._outcome_constraints

    @property
    def metrics(self) -> dict[str, Metric]:
        """Returns mapping of name to metric."""
        constraint_metrics = {
            oc.metric.name: oc.metric
            for oc in self.all_constraints
            if not isinstance(oc, ScalarizedOutcomeConstraint)
        }
        scalarized_constraint_metrics = {
            metric.name: metric
            for oc in self.all_constraints
            if isinstance(oc, ScalarizedOutcomeConstraint)
            for metric in oc.metrics
        }
        objective_metrics = {metric.name: metric for metric in self.objective.metrics}
        return {
            **constraint_metrics,
            **scalarized_constraint_metrics,
            **objective_metrics,
        }

    @property
    def is_moo_problem(self) -> bool:
        return self.objective is not None and isinstance(self.objective, MultiObjective)

    @property
    def is_bope_problem(self) -> bool:
        """Whether this is a preference optimization config for BO with
        Preference Exploration (BOPE) problems.

        Returns False for base OptimizationConfig. PreferenceOptimizationConfig
        overrides this to return True.
        """
        return False

    @outcome_constraints.setter
    def outcome_constraints(self, outcome_constraints: list[OutcomeConstraint]) -> None:
        """Set outcome constraints if valid, else raise."""
        self._validate_transformed_optimization_config(
            objective=self.objective,
            outcome_constraints=outcome_constraints,
        )
        self._outcome_constraints = outcome_constraints

    @staticmethod
    def _validate_transformed_optimization_config(
        objective: Objective,
        outcome_constraints: list[OutcomeConstraint] | None = None,
    ) -> None:
        """Ensure outcome constraints are valid.

        Either one or two outcome constraints can reference one metric.
        If there are two constraints, they must have different 'ops': one
            LEQ and one GEQ.
        If there are two constraints, the bound of the GEQ op must be less
            than the bound of the LEQ op.

        Args:
            objective: Metric+direction to use for the optimization.
            outcome_constraints: Constraints to validate.
        """
        if type(objective) is MultiObjective:
            # Raise error on exact equality; `ScalarizedObjective` is OK
            raise ValueError(
                "OptimizationConfig does not support MultiObjective. "
                "Use MultiObjectiveOptimizationConfig instead."
            )
        outcome_constraints = outcome_constraints or []
        unconstrainable_metrics = objective.get_unconstrainable_metrics()
        OptimizationConfig._validate_outcome_constraints(
            unconstrainable_metrics=unconstrainable_metrics,
            outcome_constraints=outcome_constraints,
        )

    @staticmethod
    def _validate_outcome_constraints(
        unconstrainable_metrics: list[Metric],
        outcome_constraints: list[OutcomeConstraint],
    ) -> None:
        constraint_metric_map = {}
        for oc in outcome_constraints:
            if isinstance(oc, ScalarizedOutcomeConstraint):
                for m in oc.metrics:
                    constraint_metric_map[m.signature] = m.name
            else:
                constraint_metric_map[oc.metric.signature] = oc.metric.name

        for metric in unconstrainable_metrics:
            if metric.signature in constraint_metric_map:
                raise ValueError("Cannot constrain on objective metric.")

        def constraint_key(oc: OutcomeConstraint) -> str:
            return (
                str(oc)
                if isinstance(oc, ScalarizedOutcomeConstraint)
                else oc.metric.signature
            )

        sorted_constraints = sorted(outcome_constraints, key=constraint_key)
        for metric_signature, constraints_itr in groupby(
            sorted_constraints, constraint_key
        ):
            constraints: list[OutcomeConstraint] = list(constraints_itr)
            constraints_len = len(constraints)
            if constraints_len == 2:
                if constraints[0].op == constraints[1].op:
                    raise ValueError(
                        f"Duplicate outcome constraints "
                        f"{constraint_metric_map[metric_signature]}"
                    )
                lower_bound_idx = 0 if constraints[0].op == ComparisonOp.GEQ else 1
                upper_bound_idx = 1 - lower_bound_idx
                lower_bound = constraints[lower_bound_idx].bound
                upper_bound = constraints[upper_bound_idx].bound
                if lower_bound >= upper_bound:
                    raise ValueError(
                        f"Lower bound {lower_bound} is >= upper bound "
                        f"{upper_bound} for "
                        f"{constraint_metric_map[metric_signature]}"
                    )
            elif constraints_len > 2:
                raise ValueError(
                    "Duplicate outcome constraints "
                    f"{constraint_metric_map[metric_signature]}"
                )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            "objective=" + repr(self.objective) + ", "
            "outcome_constraints=" + repr(self.outcome_constraints) + ")"
        )

    def __hash__(self) -> int:
        """Make the class hashable to support grouping of GeneratorRuns."""
        return hash(repr(self))


class MultiObjectiveOptimizationConfig(OptimizationConfig):
    """An optimization configuration for multi-objective optimization,
    which comprises multiple objective, outcome constraints, and objective
    thresholds.

    There is no minimum or maximum number of outcome constraints, but an
    individual metric can have at most two constraints--which is how we
    represent metrics with both upper and lower bounds.

    ObjectiveThresholds should be present for every objective. A good
    rule of thumb is to set them 10% below the minimum acceptable value
    for each metric.
    """

    def __init__(
        self,
        objective: MultiObjective | ScalarizedObjective,
        outcome_constraints: list[OutcomeConstraint] | None = None,
        objective_thresholds: list[ObjectiveThreshold] | None = None,
        pruning_target_parameterization: Arm | None = None,
    ) -> None:
        """Inits MultiObjectiveOptimizationConfig.

        Args:
            objective: Metric+direction to use for the optimization. Should be either a
                MultiObjective or a ScalarizedObjective.
            outcome_constraints: Constraints on metrics.
            objective_thesholds: Thresholds objectives must exceed. Used for
                multi-objective optimization and for calculating frontiers
                and hypervolumes.
            pruning_target_parameterization: Arm containing the target values for
                irrelevant parameters. The target values are used to prune irrelevant
                parameters from candidates generated via Bayesian optimization: when
                Ax considers arms to suggest for the next trial, it will seek to have
                the proposed arms differ from the target arm as little as possible,
                without a loss in optimization performance. I.e. when suggested arms
                include parameter values that differ from the corresponding value in
                the target arm, the pruning methodology will check if that difference
                is expected to be meaningful w.r.t. the performance of the arm in
                consideration, and if not, the parameter value will be replaced with
                the corresponding value in the target arm.
        """
        constraints: list[OutcomeConstraint] = (
            [] if outcome_constraints is None else outcome_constraints
        )
        objective_thresholds = objective_thresholds or []
        self._validate_transformed_optimization_config(
            objective=objective,
            outcome_constraints=constraints,
            objective_thresholds=objective_thresholds,
        )
        self._objective: MultiObjective | ScalarizedObjective = objective
        self._outcome_constraints: list[OutcomeConstraint] = constraints
        self._objective_thresholds: list[ObjectiveThreshold] = objective_thresholds
        self.pruning_target_parameterization = pruning_target_parameterization

    # pyre-fixme[14]: Inconsistent override.
    def clone_with_args(
        self,
        objective: MultiObjective | ScalarizedObjective | None = None,
        outcome_constraints: None | (list[OutcomeConstraint]) = _NO_OUTCOME_CONSTRAINTS,
        objective_thresholds: None
        | (list[ObjectiveThreshold]) = _NO_OBJECTIVE_THRESHOLDS,
        pruning_target_parameterization: Arm
        | None = _NO_PRUNING_TARGET_PARAMETERIZATION,
    ) -> "MultiObjectiveOptimizationConfig":
        """Make a copy of this optimization config."""
        objective = self.objective.clone() if objective is None else objective
        outcome_constraints = (
            [constraint.clone() for constraint in self.outcome_constraints]
            if outcome_constraints is _NO_OUTCOME_CONSTRAINTS
            else outcome_constraints
        )
        objective_thresholds = (
            [ot.clone() for ot in self.objective_thresholds]
            if objective_thresholds is _NO_OBJECTIVE_THRESHOLDS
            else objective_thresholds
        )
        pruning_target_parameterization = (
            self.pruning_target_parameterization
            if pruning_target_parameterization is _NO_PRUNING_TARGET_PARAMETERIZATION
            else pruning_target_parameterization
        )
        return MultiObjectiveOptimizationConfig(
            objective=objective,
            outcome_constraints=outcome_constraints,
            objective_thresholds=objective_thresholds,
            pruning_target_parameterization=pruning_target_parameterization,
        )

    @property
    def objective(self) -> MultiObjective | ScalarizedObjective:
        """Get objective."""
        return self._objective

    @objective.setter
    def objective(self, objective: MultiObjective | ScalarizedObjective) -> None:
        """Set objective if not present in outcome constraints."""
        self._validate_transformed_optimization_config(
            objective=objective,
            outcome_constraints=self.outcome_constraints,
            objective_thresholds=self.objective_thresholds,
        )
        self._objective = objective

    @property
    def all_constraints(self) -> list[OutcomeConstraint]:
        """Get all constraints and thresholds."""
        return self.outcome_constraints + self.objective_thresholds

    @property
    def objective_thresholds(self) -> list[ObjectiveThreshold]:
        """Get objective thresholds."""
        return self._objective_thresholds

    @objective_thresholds.setter
    def objective_thresholds(
        self, objective_thresholds: list[ObjectiveThreshold]
    ) -> None:
        """Set outcome constraints if valid, else raise."""
        self._validate_transformed_optimization_config(
            objective=self.objective,
            objective_thresholds=objective_thresholds,
        )
        self._objective_thresholds = objective_thresholds

    @property
    def objective_thresholds_dict(self) -> dict[str, ObjectiveThreshold]:
        """Get a mapping from objective metric name to the corresponding
        threshold.
        """
        return {ot.metric.name: ot for ot in self._objective_thresholds}

    @staticmethod
    def _validate_transformed_optimization_config(
        objective: Objective,
        outcome_constraints: list[OutcomeConstraint] | None = None,
        objective_thresholds: list[ObjectiveThreshold] | None = None,
    ) -> None:
        """Ensure outcome constraints are valid.

        Either one or two outcome constraints can reference one metric.
        If there are two constraints, they must have different 'ops': one
            LEQ and one GEQ.
        If there are two constraints, the bound of the GEQ op must be less
            than the bound of the LEQ op.

        Args:
            objective: Metric+direction to use for the optimization.
            outcome_constraints: Constraints to validate.
            objective_thresholds: Thresholds objectives must exceed.
        """
        if not isinstance(objective, (MultiObjective, ScalarizedObjective)):
            raise TypeError(
                "`MultiObjectiveOptimizationConfig` requires an objective "
                "of type `MultiObjective` or `ScalarizedObjective`. "
                "Use `OptimizationConfig` instead if using a "
                "single-metric objective."
            )
        outcome_constraints = outcome_constraints or []
        objective_thresholds = objective_thresholds or []
        if isinstance(objective, MultiObjective):
            objectives_by_signature = {
                obj.metric.signature: obj for obj in objective.objectives
            }
            check_objective_thresholds_match_objectives(
                objectives_by_signature=objectives_by_signature,
                objective_thresholds=objective_thresholds,
            )

        unconstrainable_metrics = objective.get_unconstrainable_metrics()
        OptimizationConfig._validate_outcome_constraints(
            unconstrainable_metrics=unconstrainable_metrics,
            outcome_constraints=outcome_constraints,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            "objective=" + repr(self.objective) + ", "
            "outcome_constraints=" + repr(self.outcome_constraints) + ", "
            "objective_thresholds=" + repr(self.objective_thresholds) + ")"
        )


def check_objective_thresholds_match_objectives(
    objectives_by_signature: dict[str, Objective],
    objective_thresholds: list[ObjectiveThreshold],
) -> None:
    """Error if thresholds on objective_metrics bound from the wrong direction or
    if there is a mismatch between objective thresholds and objectives.
    """
    obj_thresh_metrics = set()
    for threshold in objective_thresholds:
        th_metric_signature = threshold.metric.signature
        th_metric_name = threshold.metric.name
        if th_metric_signature not in objectives_by_signature:
            raise UserInputError(
                f"Objective threshold {threshold} is on metric '{th_metric_name}', "
                f"but that metric is not among the objectives."
            )
        if th_metric_signature in obj_thresh_metrics:
            raise UserInputError(
                "More than one objective threshold specified for metric "
                f"{th_metric_name}."
            )
        obj_thresh_metrics.add(th_metric_signature)

        minimize = objectives_by_signature[th_metric_signature].minimize
        bounded_above = threshold.op == ComparisonOp.LEQ
        is_aligned = minimize == bounded_above
        if not is_aligned:
            raise UserInputError(
                f"Objective threshold on {th_metric_name} bounds from "
                f"{'above' if bounded_above else 'below'} "
                f"but {th_metric_name} is being "
                f"{'minimized' if minimize else 'maximized'}."
            )


class PreferenceOptimizationConfig(MultiObjectiveOptimizationConfig):
    def __init__(
        self,
        objective: MultiObjective,
        preference_profile_name: str,
        outcome_constraints: list[OutcomeConstraint] | None = None,
        expect_relativized_outcomes: bool = False,
        pruning_target_parameterization: Arm | None = None,
    ) -> None:
        """Inits PreferenceOptimizationConfig.

        Args:
            objective: Metric+direction to use for the optimization. Should be a
                MultiObjective.
            preference_profile_name: The name of the auxiliary experiment to use as the
                preference profile for the experiment. An auxiliary experiment with
                this name and purpose PE_EXPERIMENT should be attached to
                the experiment.
            outcome_constraints: Constraints on metrics. Not yet supported.
            expect_relativized_outcomes: Whether the learned objective expects outcomes
                in relative percentage scale (e.g., -1 means -1% change vs. status quo)
                after transforms. When True, TorchAdapter validates that a
                RelativizeWithConstantControl transform exists in the pipeline
                to convert absolute outcomes to relative scale before reaching the
                preference model.
            pruning_target_parameterization: Arm containing the target values for
                irrelevant parameters. The target values are used to prune irrelevant
                parameters from candidates generated via Bayesian optimization: when
                Ax considers arms to suggest for the next trial, it will seek to have
                the proposed arms differ from the target arm as little as possible,
                without a loss in optimization performance. I.e. when suggested arms
                include parameter values that differ from the corresponding value in
                the target arm, the pruning methodology will check if that difference
                is expected to be meaningful w.r.t. the performance of the arm in
                consideration, and if not, the parameter value will be replaced with
                the corresponding value in the target arm.
        """
        if outcome_constraints:
            raise NotImplementedError(
                "Outcome constraints are not yet supported in "
                "PreferenceOptimizationConfig"
            )

        # Call parent's __init__ with objective_thresholds=None
        super().__init__(
            objective=objective,
            outcome_constraints=outcome_constraints,
            objective_thresholds=None,
            pruning_target_parameterization=pruning_target_parameterization,
        )
        self.preference_profile_name = preference_profile_name
        self.expect_relativized_outcomes = expect_relativized_outcomes

    @property
    def is_bope_problem(self) -> bool:
        """Whether this is a preference optimization config for BO with
        Preference Exploration (BOPE) problems.

        Returns:
            True for PreferenceOptimizationConfig.
        """
        return True

    # pyre-ignore[14]: Inconsistent override.
    def clone_with_args(
        self,
        objective: MultiObjective | None = None,
        preference_profile_name: str | None = None,
        outcome_constraints: list[OutcomeConstraint] | None = _NO_OUTCOME_CONSTRAINTS,
        expect_relativized_outcomes: bool | None = None,
        pruning_target_parameterization: Arm
        | None = _NO_PRUNING_TARGET_PARAMETERIZATION,
    ) -> PreferenceOptimizationConfig:
        """Make a copy of this optimization config."""
        objective = (
            assert_is_instance(self.objective.clone(), MultiObjective)
            if objective is None
            else objective
        )

        preference_profile_name = (
            self.preference_profile_name
            if preference_profile_name is None
            else preference_profile_name
        )
        outcome_constraints = (
            [constraint.clone() for constraint in self.outcome_constraints]
            if outcome_constraints is _NO_OUTCOME_CONSTRAINTS
            else outcome_constraints
        )
        expect_relativized_outcomes = (
            self.expect_relativized_outcomes
            if expect_relativized_outcomes is None
            else expect_relativized_outcomes
        )
        pruning_target_parameterization = (
            self.pruning_target_parameterization
            if pruning_target_parameterization is _NO_PRUNING_TARGET_PARAMETERIZATION
            else pruning_target_parameterization
        )

        return PreferenceOptimizationConfig(
            objective=objective,
            preference_profile_name=preference_profile_name,
            outcome_constraints=outcome_constraints,
            expect_relativized_outcomes=expect_relativized_outcomes,
            pruning_target_parameterization=pruning_target_parameterization,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            "objective=" + repr(self.objective) + ", "
            "preference_profile_name=" + repr(self.preference_profile_name) + ", "
            "outcome_constraints=" + repr(self.outcome_constraints) + ", "
            "expect_relativized_outcomes="
            + repr(self.expect_relativized_outcomes)
            + ")"
        )

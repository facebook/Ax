#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import warnings
from collections.abc import Mapping
from itertools import groupby
from typing import Self

from ax.core.arm import Arm
from ax.core.objective import Objective
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.utils.common.base import Base
from pyre_extensions import none_throws


TRefPoint = list[OutcomeConstraint]

# Sentinels for default arguments when None is a valid input
_NO_OUTCOME_CONSTRAINTS: list[OutcomeConstraint] = [
    OutcomeConstraint(
        expression="__placeholder__ >= 0",
        metric_name_to_signature={"__placeholder__": "__placeholder__"},
    )
]
_NO_OBJECTIVE_THRESHOLDS: list[OutcomeConstraint] = [
    OutcomeConstraint(
        expression="__placeholder__ >= 0",
        metric_name_to_signature={"__placeholder__": "__placeholder__"},
    )
]

_NO_PRUNING_TARGET_PARAMETERIZATION = Arm(parameters={})


class OptimizationConfig(Base):
    """An optimization configuration, which comprises one or more objectives
    and outcome constraints.

    For single-objective optimization, pass a single ``objective``.
    For multi-objective optimization, pass a list of ``objectives``.

    There is no minimum or maximum number of outcome constraints, but an
    individual metric can have at most two constraints--which is how we
    represent metrics with both upper and lower bounds.
    """

    def __init__(
        self,
        *,
        objective: Objective | None = None,
        objectives: list[Objective] | None = None,
        outcome_constraints: list[OutcomeConstraint] | None = None,
        pruning_target_parameterization: Arm | None = None,
    ) -> None:
        """Inits OptimizationConfig.

        Args:
            objective: Metric+direction to use for the optimization. Mutually
                exclusive with ``objectives``.
            objectives: List of objectives for multi-objective optimization.
                Mutually exclusive with ``objective``. Each element must be a
                single or scalarized Objective (not multi-objective).
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
        if objective is not None:
            warnings.warn(
                "Passing `objective` to OptimizationConfig is deprecated. "
                "Use `objectives=[objective]` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if objective is not None and objectives is not None:
            raise UserInputError(
                "Cannot specify both `objective` and `objectives`. "
                "Use `objectives=[objective]` instead."
            )
        if objective is None and objectives is None:
            raise UserInputError("Must specify either `objective` or `objectives`.")

        if objectives is not None:
            if len(objectives) == 0:
                raise UserInputError("`objectives` must not be empty.")
            self._objectives: list[Objective] = objectives
        else:
            self._objectives = [none_throws(objective)]

        constraints: list[OutcomeConstraint] = (
            [] if outcome_constraints is None else outcome_constraints
        )
        self._validate_transformed_optimization_config(
            objectives=self._objectives,
            outcome_constraints=constraints,
        )
        self._outcome_constraints: list[OutcomeConstraint] = constraints
        self.pruning_target_parameterization = pruning_target_parameterization

    def clone(self) -> Self:
        """Make a copy of this optimization config."""
        return self.clone_with_args()

    def clone_with_args(
        self,
        *,
        objectives: list[Objective] | None = None,
        outcome_constraints: None | (list[OutcomeConstraint]) = _NO_OUTCOME_CONSTRAINTS,
        pruning_target_parameterization: Arm
        | None = _NO_PRUNING_TARGET_PARAMETERIZATION,
    ) -> Self:
        """Make a copy of this optimization config.

        Args:
            objectives: Replace with a list of objectives.
            outcome_constraints: Replace outcome constraints. Pass ``None``
                to clear them.
            pruning_target_parameterization: Replace the pruning target.
        """
        if objectives is not None:
            cloned_objectives = objectives
        else:
            cloned_objectives = [obj.clone() for obj in self._objectives]
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

        return self.__class__(
            objectives=cloned_objectives,
            outcome_constraints=outcome_constraints,
            pruning_target_parameterization=pruning_target_parameterization,
        )

    @property
    def objectives(self) -> list[Objective]:
        """Get the list of objectives.

        For single-objective optimization, this returns a single-element list.
        For multi-objective optimization, this returns all objectives.
        """
        return self._objectives

    @property
    def objective(self) -> Objective:
        """Get the single objective.

        For single-objective or scalarized-objective configs, returns the
        objective. For multi-objective configs (multiple objectives in the
        list), raises ``UserInputError`` -- use ``objectives`` instead.
        """
        if len(self._objectives) > 1:
            raise UnsupportedError(
                "This OptimizationConfig has multiple objectives. "
                "Use `objectives` to access the list of objectives, or "
                "iterate over individual objectives."
            )
        return self._objectives[0]

    @property
    def all_constraints(self) -> list[OutcomeConstraint]:
        """Get outcome constraints."""
        return self.outcome_constraints

    @property
    def outcome_constraints(self) -> list[OutcomeConstraint]:
        """Get outcome constraints."""
        return self._outcome_constraints

    @property
    def objective_thresholds(self) -> list[OutcomeConstraint]:
        """Get objective thresholds.

        Returns outcome constraints whose primary metric is an objective
        metric.
        """
        all_obj_metric_names: set[str] = set()
        for obj in self._objectives:
            all_obj_metric_names.update(obj.metric_names)
        return [
            threshold
            for threshold in self.outcome_constraints
            if threshold.metric_names[0] in all_obj_metric_names
        ]

    @property
    def metric_names(self) -> set[str]:
        """All metric names referenced by the objectives and constraints."""
        names: set[str] = set()
        for obj in self._objectives:
            names.update(obj.metric_names)
        for oc in self.all_constraints:
            names.update(oc.metric_names)
        return names

    @property
    def metric_name_to_signature(self) -> dict[str, str]:
        """Aggregated mapping from all metric names to their canonical
        signatures, across all objectives and all constraints.
        """
        mapping: dict[str, str] = {}
        for obj in self._objectives:
            mapping.update(obj.metric_name_to_signature)
        for constraint in self.all_constraints:
            mapping.update(constraint.metric_name_to_signature)
        return mapping

    def update_metric_name_to_signature_mapping(
        self, mapping: Mapping[str, str]
    ) -> None:
        """Set the metric name to signature mapping on all objectives and
        constraints.
        """
        for obj in self._objectives:
            obj.update_metric_name_to_signature_mapping(mapping)
        for constraint in self.all_constraints:
            constraint.update_metric_name_to_signature_mapping(mapping)

    @property
    def metric_signatures(self) -> set[str]:
        """All metric signatures referenced by the objectives and constraints."""
        mapping = self.metric_name_to_signature
        return {mapping[name] for name in self.metric_names}

    @property
    def is_moo_problem(self) -> bool:
        """Whether this is a multi-objective optimization problem.

        True when there are multiple objectives in the list, or when a single
        objective is a (legacy) multi-objective expression.
        """
        if len(self._objectives) > 1:
            return True
        return self._objectives[0].is_multi_objective

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
        unconstrainable: list[str] = []
        for obj in self._objectives:
            unconstrainable.extend(obj.get_unconstrainable_metric_names())
        self._validate_outcome_constraints(
            unconstrainable_metric_names=unconstrainable,
            outcome_constraints=outcome_constraints,
        )
        self._outcome_constraints = outcome_constraints

    @staticmethod
    def _validate_transformed_optimization_config(
        objectives: list[Objective],
        outcome_constraints: list[OutcomeConstraint] | None = None,
    ) -> None:
        """Validate objectives and outcome constraints.

        Ensures no multi-objective expressions in individual objectives,
        no duplicate metrics across objectives, outcome constraints don't
        constrain objective metrics, and that constraint pairs on the
        same metric are valid.

        Subclasses (e.g. ``MultiObjectiveOptimizationConfig``) override
        this to allow multi-objective expressions.

        Args:
            objectives: List of objectives to validate.
            outcome_constraints: Constraints to validate.
        """
        all_metric_names: list[str] = []
        for obj in objectives:
            if obj.is_multi_objective:
                raise ValueError(
                    "Each objective in `objectives` must be a single or "
                    "scalarized objective, not a multi-objective. "
                    "Pass each sub-objective as a separate list element."
                )
            for name in obj.metric_names:
                if name in all_metric_names:
                    raise UserInputError(
                        f"Metric '{name}' appears in multiple objectives. "
                        "Each metric can only appear in one objective."
                    )
                all_metric_names.append(name)
        outcome_constraints = outcome_constraints or []
        unconstrainable_metric_names: list[str] = []
        for obj in objectives:
            unconstrainable_metric_names.extend(obj.get_unconstrainable_metric_names())
        OptimizationConfig._validate_outcome_constraints(
            unconstrainable_metric_names=unconstrainable_metric_names,
            outcome_constraints=outcome_constraints,
        )

    @staticmethod
    def _validate_outcome_constraints(
        unconstrainable_metric_names: list[str],
        outcome_constraints: list[OutcomeConstraint],
    ) -> None:
        # Build a set of all metric names referenced in constraints
        constraint_metric_names: set[str] = set()
        for oc in outcome_constraints:
            constraint_metric_names.update(oc.metric_names)

        for name in unconstrainable_metric_names:
            if name in constraint_metric_names:
                raise ValueError("Cannot constrain on objective metric.")

        def constraint_key(oc: OutcomeConstraint) -> str:
            # For multi-metric constraints, use the full string representation
            # For single-metric, use the metric name
            if len(oc.metric_names) > 1:
                return str(oc)
            return oc.metric_names[0]

        sorted_constraints = sorted(outcome_constraints, key=constraint_key)
        for key, constraints_itr in groupby(sorted_constraints, constraint_key):
            constraints: list[OutcomeConstraint] = list(constraints_itr)
            constraints_len = len(constraints)
            if constraints_len == 2:
                if constraints[0].op == constraints[1].op:
                    raise ValueError(f"Duplicate outcome constraints {key}")
                lower_bound_idx = 0 if constraints[0].op == ComparisonOp.GEQ else 1
                upper_bound_idx = 1 - lower_bound_idx
                lower_bound = constraints[lower_bound_idx].bound
                upper_bound = constraints[upper_bound_idx].bound
                if lower_bound >= upper_bound:
                    raise ValueError(
                        f"Lower bound {lower_bound} is >= upper bound "
                        f"{upper_bound} for {key}"
                    )
            elif constraints_len > 2:
                raise ValueError(f"Duplicate outcome constraints {key}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            "objectives=" + repr(self._objectives) + ", "
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
        *,
        objective: Objective | None = None,
        objectives: list[Objective] | None = None,
        outcome_constraints: list[OutcomeConstraint] | None = None,
        objective_thresholds: list[OutcomeConstraint] | None = None,
        pruning_target_parameterization: Arm | None = None,
    ) -> None:
        """Inits MultiObjectiveOptimizationConfig.

        Args:
            objective: Metric+direction to use for the optimization. Should be either a
                MultiObjective or a ScalarizedObjective. Mutually exclusive with
                ``objectives``.
            objectives: List containing the objective. Mutually exclusive
                with ``objective``.
            outcome_constraints: Constraints on metrics.
            objective_thresholds: Thresholds objectives must exceed. Used for
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
        super().__init__(
            objective=objective,
            objectives=objectives,
            outcome_constraints=outcome_constraints,
            pruning_target_parameterization=pruning_target_parameterization,
        )
        # Validate and set objective thresholds (MOOC-specific).
        self._objective_thresholds: list[OutcomeConstraint] = objective_thresholds or []
        self._validate_transformed_optimization_config(
            objectives=self._objectives,
            outcome_constraints=self._outcome_constraints,
            objective_thresholds=self._objective_thresholds,
        )

    # pyre-fixme[14]: Inconsistent override.
    def clone_with_args(
        self,
        *,
        objectives: list[Objective] | None = None,
        outcome_constraints: None | (list[OutcomeConstraint]) = _NO_OUTCOME_CONSTRAINTS,
        objective_thresholds: None
        | (list[OutcomeConstraint]) = _NO_OBJECTIVE_THRESHOLDS,
        pruning_target_parameterization: Arm
        | None = _NO_PRUNING_TARGET_PARAMETERIZATION,
    ) -> "MultiObjectiveOptimizationConfig":
        """Make a copy of this optimization config."""
        if objectives is not None:
            cloned_objectives = objectives
        else:
            cloned_objectives = [obj.clone() for obj in self._objectives]
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
            objectives=cloned_objectives,
            outcome_constraints=outcome_constraints,
            objective_thresholds=objective_thresholds,
            pruning_target_parameterization=pruning_target_parameterization,
        )

    @property
    def objective(self) -> Objective:
        """Get the (multi-)objective."""
        return self._objectives[0]

    @objective.setter
    def objective(self, objective: Objective) -> None:
        """Set objective if not present in outcome constraints."""
        self._validate_transformed_optimization_config(
            objectives=[objective],
            outcome_constraints=self.outcome_constraints,
            objective_thresholds=self.objective_thresholds,
        )
        self._objectives = [objective]

    @property
    def all_constraints(self) -> list[OutcomeConstraint]:
        """Get all constraints and thresholds."""
        return self.outcome_constraints + self.objective_thresholds

    @property
    def objective_thresholds(self) -> list[OutcomeConstraint]:
        """Get objective thresholds."""
        return self._objective_thresholds

    @objective_thresholds.setter
    def objective_thresholds(
        self, objective_thresholds: list[OutcomeConstraint]
    ) -> None:
        """Set outcome constraints if valid, else raise."""
        self._validate_transformed_optimization_config(
            objectives=self._objectives,
            objective_thresholds=objective_thresholds,
        )
        self._objective_thresholds = objective_thresholds

    @property
    def objective_thresholds_dict(self) -> dict[str, OutcomeConstraint]:
        """Get a mapping from objective metric name to the corresponding
        threshold.
        """
        return {ot.metric_names[0]: ot for ot in self._objective_thresholds}

    @staticmethod
    def _validate_transformed_optimization_config(
        objectives: list[Objective],
        outcome_constraints: list[OutcomeConstraint] | None = None,
        objective_thresholds: list[OutcomeConstraint] | None = None,
    ) -> None:
        """Ensure outcome constraints are valid.

        Either one or two outcome constraints can reference one metric.
        If there are two constraints, they must have different 'ops': one
            LEQ and one GEQ.
        If there are two constraints, the bound of the GEQ op must be less
            than the bound of the LEQ op.

        Args:
            objectives: List of objectives to validate.
            outcome_constraints: Constraints to validate.
            objective_thresholds: Thresholds objectives must exceed.
        """
        objective = objectives[0]
        if not (objective.is_multi_objective or objective.is_scalarized_objective):
            raise TypeError(
                "`MultiObjectiveOptimizationConfig` requires an objective "
                "of type `MultiObjective` or `ScalarizedObjective`. "
                "Use `OptimizationConfig` instead if using a "
                "single-metric objective."
            )
        outcome_constraints = outcome_constraints or []
        objective_thresholds = objective_thresholds or []
        if objective.is_multi_objective:
            # Build objectives_by_name by decomposing the multi-objective
            # expression into sub-objectives
            parts = [p.strip() for p in objective.expression.split(",")]
            objectives_by_name: dict[str, Objective] = {}
            for part in parts:
                sub_obj = Objective(
                    expression=part,
                    metric_name_to_signature=objective.metric_name_to_signature,
                )
                for name in sub_obj.metric_names:
                    objectives_by_name[name] = sub_obj

            check_objective_thresholds_match_objectives(
                objectives_by_name=objectives_by_name,
                objective_thresholds=objective_thresholds,
            )

        unconstrainable_metric_names = objective.get_unconstrainable_metric_names()
        OptimizationConfig._validate_outcome_constraints(
            unconstrainable_metric_names=unconstrainable_metric_names,
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
    objectives_by_name: dict[str, Objective],
    objective_thresholds: list[OutcomeConstraint],
) -> None:
    """Error if thresholds on objective_metrics bound from the wrong direction or
    if there is a mismatch between objective thresholds and objectives.
    """
    obj_thresh_metrics: set[str] = set()
    for threshold in objective_thresholds:
        th_metric_name = threshold.metric_names[0]
        if th_metric_name not in objectives_by_name:
            raise UserInputError(
                f"Objective threshold {threshold} is on metric '{th_metric_name}', "
                f"but that metric is not among the objectives."
            )
        if th_metric_name in obj_thresh_metrics:
            raise UserInputError(
                "More than one objective threshold specified for metric "
                f"{th_metric_name}."
            )
        obj_thresh_metrics.add(th_metric_name)

        minimize = objectives_by_name[th_metric_name].minimize
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
        *,
        objective: Objective | None = None,
        objectives: list[Objective] | None = None,
        preference_profile_name: str,
        outcome_constraints: list[OutcomeConstraint] | None = None,
        expect_relativized_outcomes: bool = False,
        pruning_target_parameterization: Arm | None = None,
    ) -> None:
        """Inits PreferenceOptimizationConfig.

        Args:
            objective: Metric+direction to use for the optimization. Should be a
                MultiObjective. Mutually exclusive with ``objectives``.
            objectives: List containing the objective. Mutually exclusive
                with ``objective``.
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
            objectives=objectives,
            outcome_constraints=outcome_constraints,
            objective_thresholds=None,
            pruning_target_parameterization=pruning_target_parameterization,
        )
        # Validate that the objective is multi-objective (after super sets
        # self._objectives).
        if not self._objectives[0].is_multi_objective:
            raise TypeError(
                "`PreferenceOptimizationConfig` requires a multi-objective. "
                "Use `OptimizationConfig` instead if using a "
                "single-metric objective."
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
        *,
        objectives: list[Objective] | None = None,
        preference_profile_name: str | None = None,
        outcome_constraints: list[OutcomeConstraint] | None = _NO_OUTCOME_CONSTRAINTS,
        expect_relativized_outcomes: bool | None = None,
        pruning_target_parameterization: Arm
        | None = _NO_PRUNING_TARGET_PARAMETERIZATION,
    ) -> PreferenceOptimizationConfig:
        """Make a copy of this optimization config."""
        cloned_objectives = (
            [obj.clone() for obj in self._objectives]
            if objectives is None
            else objectives
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
            objectives=cloned_objectives,
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

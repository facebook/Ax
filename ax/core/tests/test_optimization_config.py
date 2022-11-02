#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import (
    _NO_RISK_MEASURE,
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import (
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.core.risk_measures import RiskMeasure
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase


OC_STR = (
    'OptimizationConfig(objective=Objective(metric_name="m1", minimize=False), '
    "outcome_constraints=[OutcomeConstraint(m3 >= -0.25%), "
    "OutcomeConstraint(m4 <= 0.25%), "
    "ScalarizedOutcomeConstraint(metric_names=['m3', 'm4'], "
    "weights=[0.5, 0.5], >= -0.25%)])"
)

MOOC_STR = (
    "MultiObjectiveOptimizationConfig(objective=MultiObjective(objectives="
    '[Objective(metric_name="m1", minimize=True), '
    'Objective(metric_name="m2", minimize=False)]), '
    "outcome_constraints=[OutcomeConstraint(m3 >= -0.25%), "
    "OutcomeConstraint(m3 <= 0.25%)], objective_thresholds=[])"
)


class OptimizationConfigTest(TestCase):
    def setUp(self) -> None:
        self.metrics = {
            "m1": Metric(name="m1"),
            "m2": Metric(name="m2"),
            "m3": Metric(name="m3"),
            "m4": Metric(name="m4"),
        }
        self.objective = Objective(metric=self.metrics["m1"], minimize=False)
        self.alt_objective = Objective(metric=self.metrics["m3"], minimize=False)
        self.multi_objective = MultiObjective(
            objectives=[self.objective, self.alt_objective],
        )
        self.m2_objective = ScalarizedObjective(
            metrics=[self.metrics["m1"], self.metrics["m2"]]
        )
        self.outcome_constraint = OutcomeConstraint(
            metric=self.metrics["m3"], op=ComparisonOp.GEQ, bound=-0.25
        )
        self.additional_outcome_constraint = OutcomeConstraint(
            metric=self.metrics["m4"], op=ComparisonOp.LEQ, bound=0.25
        )
        self.scalarized_outcome_constraint = ScalarizedOutcomeConstraint(
            metrics=[self.metrics["m3"], self.metrics["m4"]],
            weights=[0.5, 0.5],
            op=ComparisonOp.GEQ,
            bound=-0.25,
        )
        self.outcome_constraints = [
            self.outcome_constraint,
            self.additional_outcome_constraint,
            self.scalarized_outcome_constraint,
        ]
        self.single_output_risk_measure = RiskMeasure(
            risk_measure="Expectation",
            options={"n_w": 2},
        )
        self.multi_output_risk_measure = RiskMeasure(
            risk_measure="MultiOutputExpectation",
            options={"n_w": 2},
        )

    def testInit(self) -> None:
        config1 = OptimizationConfig(
            objective=self.objective, outcome_constraints=self.outcome_constraints
        )
        self.assertEqual(str(config1), OC_STR)
        with self.assertRaises(ValueError):
            config1.objective = self.alt_objective  # constrained Objective.
        # updating constraints is fine.
        config1.outcome_constraints = [self.outcome_constraint]
        self.assertEqual(len(config1.metrics), 2)

        # objective without outcome_constraints is also supported
        config2 = OptimizationConfig(objective=self.objective)
        self.assertEqual(config2.outcome_constraints, [])

        # setting objective is fine too, if it's compatible with constraints..
        config2.objective = self.m2_objective
        # setting constraints on objectives is fine for MultiObjective components.

        config2.outcome_constraints = self.outcome_constraints
        self.assertEqual(config2.outcome_constraints, self.outcome_constraints)

        # Risk measure is correctly registered.
        self.assertIsNone(config2.risk_measure)
        config3 = OptimizationConfig(
            objective=self.objective,
            outcome_constraints=self.outcome_constraints,
            risk_measure=self.single_output_risk_measure,
        )
        expected_str = (
            OC_STR[:-1] + ", risk_measure=RiskMeasure(risk_measure=Expectation, "
            "options={'n_w': 2}))"
        )
        self.assertEqual(str(config3), expected_str)
        self.assertIs(config3.risk_measure, self.single_output_risk_measure)

    def testEq(self) -> None:
        config1 = OptimizationConfig(
            objective=self.objective,
            outcome_constraints=self.outcome_constraints,
            risk_measure=self.single_output_risk_measure,
        )
        config2 = OptimizationConfig(
            objective=self.objective,
            outcome_constraints=self.outcome_constraints,
            risk_measure=self.single_output_risk_measure,
        )
        self.assertEqual(config1, config2)

        new_outcome_constraint = OutcomeConstraint(
            metric=self.metrics["m2"], op=ComparisonOp.LEQ, bound=0.5
        )
        config3 = OptimizationConfig(
            objective=self.objective,
            outcome_constraints=[self.outcome_constraint, new_outcome_constraint],
        )
        self.assertNotEqual(config1, config3)

    def testConstraintValidation(self) -> None:
        # Can build OptimizationConfig with MultiObjective
        with self.assertRaises(ValueError):
            OptimizationConfig(objective=self.multi_objective)

        # Can't constrain on objective metric.
        objective_constraint = OutcomeConstraint(
            metric=self.objective.metric, op=ComparisonOp.GEQ, bound=0
        )
        with self.assertRaises(ValueError):
            OptimizationConfig(
                objective=self.objective, outcome_constraints=[objective_constraint]
            )
        # Using an outcome constraint for ScalarizedObjective should also raise
        with self.assertRaisesRegex(
            ValueError, "Cannot constrain on objective metric."
        ):
            OptimizationConfig(
                objective=self.m2_objective,
                outcome_constraints=[objective_constraint],
            )
        # Two outcome_constraints on the same metric with the same op
        # should raise.
        duplicate_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            op=self.outcome_constraint.op,
            bound=self.outcome_constraint.bound + 1,
        )
        with self.assertRaises(ValueError):
            OptimizationConfig(
                objective=self.objective,
                outcome_constraints=[self.outcome_constraint, duplicate_constraint],
            )

        # Three outcome_constraints on the same metric should raise.
        opposing_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            # pyre-fixme[6]: For 2nd param expected `ComparisonOp` but got `bool`.
            op=not self.outcome_constraint.op,
            bound=self.outcome_constraint.bound,
        )
        with self.assertRaises(ValueError):
            OptimizationConfig(
                objective=self.objective,
                outcome_constraints=self.outcome_constraints + [opposing_constraint],
            )

        # Two outcome_constraints on the same metric with different ops and
        # flipped bounds (lower < upper) should raise.
        add_bound = 1 if self.outcome_constraint.op == ComparisonOp.LEQ else -1
        opposing_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            # pyre-fixme[6]: For 2nd param expected `ComparisonOp` but got `bool`.
            op=not self.outcome_constraint.op,
            bound=self.outcome_constraint.bound + add_bound,
        )
        with self.assertRaises(ValueError):
            OptimizationConfig(
                objective=self.objective,
                outcome_constraints=([self.outcome_constraint, opposing_constraint]),
            )

        # Two outcome_constraints on the same metric with different ops and
        # bounds should not raise.
        opposing_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            # pyre-fixme[6]: For 2nd param expected `ComparisonOp` but got `bool`.
            op=not self.outcome_constraint.op,
            bound=self.outcome_constraint.bound + 1,
        )
        config = OptimizationConfig(
            objective=self.objective,
            outcome_constraints=([self.outcome_constraint, opposing_constraint]),
        )
        self.assertEqual(
            config.outcome_constraints, [self.outcome_constraint, opposing_constraint]
        )

    def testClone(self) -> None:
        config1 = OptimizationConfig(
            objective=self.objective,
            outcome_constraints=self.outcome_constraints,
            risk_measure=self.single_output_risk_measure,
        )
        self.assertEqual(config1, config1.clone())

    def testCloneWithArgs(self) -> None:
        config1 = OptimizationConfig(
            objective=self.objective,
            outcome_constraints=self.outcome_constraints,
            risk_measure=self.single_output_risk_measure,
        )
        config2 = OptimizationConfig(
            objective=self.objective,
        )
        config3 = OptimizationConfig(
            objective=self.objective, risk_measure=_NO_RISK_MEASURE.clone()
        )

        # Empty args produce exact clone
        self.assertEqual(
            config1.clone_with_args(),
            config1,
        )

        # None args not treated as default
        self.assertEqual(
            config1.clone_with_args(
                outcome_constraints=None,
                risk_measure=None,
            ),
            config2,
        )

        # Arguments that has same value with default won't be treated as default
        self.assertEqual(
            config1.clone_with_args(
                outcome_constraints=None,
                risk_measure=config3.risk_measure,
            ),
            config3,
        )


class MultiObjectiveOptimizationConfigTest(TestCase):
    def setUp(self) -> None:
        self.metrics = {
            "m1": Metric(name="m1", lower_is_better=True),
            "m2": Metric(name="m2", lower_is_better=False),
            "m3": Metric(name="m3", lower_is_better=False),
        }
        self.objectives = {
            "o1": Objective(metric=self.metrics["m1"]),
            "o2": Objective(metric=self.metrics["m2"], minimize=False),
            "o3": Objective(metric=self.metrics["m3"], minimize=False),
        }
        self.objective = Objective(metric=self.metrics["m1"], minimize=False)
        self.multi_objective = MultiObjective(
            objectives=[self.objectives["o1"], self.objectives["o2"]]
        )
        self.multi_objective_just_m2 = MultiObjective(
            objectives=[self.objectives["o2"]]
        )
        self.scalarized_objective = ScalarizedObjective(
            metrics=list(self.metrics.values()), weights=[1.0, 1.0, 1.0]
        )
        self.outcome_constraint = OutcomeConstraint(
            metric=self.metrics["m3"], op=ComparisonOp.GEQ, bound=-0.25
        )
        self.additional_outcome_constraint = OutcomeConstraint(
            metric=self.metrics["m3"], op=ComparisonOp.LEQ, bound=0.25
        )
        self.outcome_constraints = [
            self.outcome_constraint,
            self.additional_outcome_constraint,
        ]
        self.objective_thresholds = [
            ObjectiveThreshold(metric=self.metrics["m1"], bound=-1.0, relative=False),
            ObjectiveThreshold(metric=self.metrics["m2"], bound=-1.0, relative=False),
        ]
        self.relative_objective_thresholds = [
            ObjectiveThreshold(metric=self.metrics["m1"], bound=-1.0, relative=True),
            ObjectiveThreshold(
                metric=self.metrics["m2"],
                op=ComparisonOp.GEQ,
                bound=-1.0,
                relative=True,
            ),
        ]
        self.m1_constraint = OutcomeConstraint(
            metric=self.metrics["m1"], op=ComparisonOp.LEQ, bound=0.1, relative=True
        )
        self.m3_constraint = OutcomeConstraint(
            metric=self.metrics["m3"], op=ComparisonOp.GEQ, bound=0.1, relative=True
        )
        self.single_output_risk_measure = RiskMeasure(
            risk_measure="Expectation",
            options={"n_w": 2},
        )
        self.multi_output_risk_measure = RiskMeasure(
            risk_measure="MultiOutputExpectation",
            options={"n_w": 2},
        )

    def testInit(self) -> None:
        config1 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective, outcome_constraints=self.outcome_constraints
        )
        self.assertEqual(str(config1), MOOC_STR)
        with self.assertRaises(TypeError):
            config1.objective = self.objective  # Wrong objective type
        # updating constraints is fine.
        config1.outcome_constraints = [self.outcome_constraint]
        self.assertEqual(len(config1.metrics), 3)

        # objective without outcome_constraints is also supported
        config2 = MultiObjectiveOptimizationConfig(objective=self.multi_objective)

        # setting objective is fine too, if it's compatible with constraints.
        config2.objective = self.multi_objective

        # setting constraints on objectives is fine for MultiObjective components.
        config2.outcome_constraints = [self.outcome_constraint]
        self.assertEqual(config2.outcome_constraints, [self.outcome_constraint])

        # construct constraints with objective_thresholds:
        config3 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            objective_thresholds=self.objective_thresholds,
        )
        self.assertEqual(config3.all_constraints, self.objective_thresholds)

        # objective_thresholds and outcome constraints together.
        config4 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            objective_thresholds=self.objective_thresholds,
            outcome_constraints=[self.m3_constraint],
        )
        self.assertEqual(
            config4.all_constraints, [self.m3_constraint] + self.objective_thresholds
        )
        self.assertEqual(config4.outcome_constraints, [self.m3_constraint])
        self.assertEqual(config4.objective_thresholds, self.objective_thresholds)

        # verify relative_objective_thresholds works:
        config5 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            objective_thresholds=self.relative_objective_thresholds,
        )
        threshold = config5.objective_thresholds[0]
        self.assertTrue(threshold.relative)
        self.assertEqual(threshold.bound, -1.0)

        # ValueError on wrong direction constraints
        with self.assertRaises(UserInputError):
            MultiObjectiveOptimizationConfig(
                objective=self.multi_objective,
                # pyre-fixme[6]: For 2nd param expected
                #  `Optional[List[ObjectiveThreshold]]` but got
                #  `List[OutcomeConstraint]`.
                objective_thresholds=[self.additional_outcome_constraint],
            )

        # Test with risk measures.
        self.assertIsNone(config5.risk_measure)
        config6 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            outcome_constraints=self.outcome_constraints,
            risk_measure=self.multi_output_risk_measure,
        )
        self.assertIs(config6.risk_measure, self.multi_output_risk_measure)
        expected_str = (
            MOOC_STR[:-1] + ", risk_measure=RiskMeasure(risk_measure="
            "MultiOutputExpectation, options={'n_w': 2}))"
        )
        self.assertEqual(str(config6), expected_str)
        # With scalarized objective.
        config7 = MultiObjectiveOptimizationConfig(
            objective=self.scalarized_objective,
            risk_measure=self.single_output_risk_measure,
        )
        self.assertIs(config7.risk_measure, self.single_output_risk_measure)

    def testEq(self) -> None:
        config1 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective, outcome_constraints=self.outcome_constraints
        )
        config2 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective, outcome_constraints=self.outcome_constraints
        )
        self.assertEqual(config1, config2)

        new_outcome_constraint = OutcomeConstraint(
            metric=self.metrics["m3"], op=ComparisonOp.LEQ, bound=0.5
        )
        config3 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            outcome_constraints=[self.outcome_constraint, new_outcome_constraint],
        )
        self.assertNotEqual(config1, config3)

    def testConstraintValidation(self) -> None:
        # Cannot build with non-MultiObjective
        with self.assertRaises(TypeError):
            MultiObjectiveOptimizationConfig(objective=self.objective)

        # Using an outcome constraint for an objective should raise
        outcome_constraint_m1 = OutcomeConstraint(
            metric=self.metrics["m1"], op=ComparisonOp.LEQ, bound=1234, relative=False
        )
        with self.assertRaisesRegex(
            ValueError, "Cannot constrain on objective metric."
        ):
            MultiObjectiveOptimizationConfig(
                objective=self.multi_objective,
                outcome_constraints=[outcome_constraint_m1],
            )
        # Two outcome_constraints on the same metric with the same op
        # should raise.
        duplicate_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            op=self.outcome_constraint.op,
            bound=self.outcome_constraint.bound + 1,
        )
        with self.assertRaises(ValueError):
            MultiObjectiveOptimizationConfig(
                objective=self.multi_objective,
                outcome_constraints=[self.outcome_constraint, duplicate_constraint],
            )

        # Three outcome_constraints on the same metric should raise.
        opposing_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            # pyre-fixme[6]: For 2nd param expected `ComparisonOp` but got `bool`.
            op=not self.outcome_constraint.op,
            bound=self.outcome_constraint.bound,
        )
        with self.assertRaises(ValueError):
            MultiObjectiveOptimizationConfig(
                objective=self.multi_objective,
                outcome_constraints=self.outcome_constraints + [opposing_constraint],
            )

        # Two outcome_constraints on the same metric with different ops and
        # flipped bounds (lower < upper) should raise.
        add_bound = 1 if self.outcome_constraint.op == ComparisonOp.LEQ else -1
        opposing_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            # pyre-fixme[6]: For 2nd param expected `ComparisonOp` but got `bool`.
            op=not self.outcome_constraint.op,
            bound=self.outcome_constraint.bound + add_bound,
        )
        with self.assertRaises(ValueError):
            MultiObjectiveOptimizationConfig(
                objective=self.multi_objective,
                outcome_constraints=([self.outcome_constraint, opposing_constraint]),
            )

        # Two outcome_constraints on the same metric with different ops and
        # bounds should not raise.
        opposing_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            # pyre-fixme[6]: For 2nd param expected `ComparisonOp` but got `bool`.
            op=not self.outcome_constraint.op,
            bound=self.outcome_constraint.bound + 1,
        )
        config = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            outcome_constraints=([self.outcome_constraint, opposing_constraint]),
        )
        self.assertEqual(
            config.outcome_constraints, [self.outcome_constraint, opposing_constraint]
        )

    def testClone(self) -> None:
        config1 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective, outcome_constraints=self.outcome_constraints
        )
        self.assertEqual(config1, config1.clone())

        config2 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            objective_thresholds=self.objective_thresholds,
        )
        self.assertEqual(config2, config2.clone())

    def testCloneWithArgs(self) -> None:
        config1 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            objective_thresholds=self.objective_thresholds,
            outcome_constraints=self.outcome_constraints,
            risk_measure=self.multi_output_risk_measure,
        )
        config2 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
        )
        config3 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective, risk_measure=_NO_RISK_MEASURE.clone()
        )

        # Empty args produce exact clone
        self.assertEqual(
            config1.clone_with_args(),
            config1,
        )

        # None args not treated as default
        self.assertEqual(
            config1.clone_with_args(
                outcome_constraints=None,
                objective_thresholds=None,
                risk_measure=None,
            ),
            config2,
        )

        # Arguments that has same value with default won't be treated as default
        self.assertEqual(
            config1.clone_with_args(
                outcome_constraints=None,
                objective_thresholds=None,
                risk_measure=config3.risk_measure,
            ),
            config3,
        )

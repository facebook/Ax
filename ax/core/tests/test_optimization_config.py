#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.metric import Metric
from ax.core.objective import Objective, ScalarizedObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.utils.common.testutils import TestCase


CONFIG_STR = (
    'OptimizationConfig(objective=Objective(metric_name="m1", minimize=False), '
    "outcome_constraints=[OutcomeConstraint(m2 >= -0.25%), "
    "OutcomeConstraint(m2 <= 0.25%)])"
)


class OptimizationConfigTest(TestCase):
    def setUp(self):
        self.metrics = {"m1": Metric(name="m1"), "m2": Metric(name="m2")}
        self.objective = Objective(metric=self.metrics["m1"], minimize=False)
        self.m2_objective = ScalarizedObjective(
            metrics=[self.metrics["m1"], self.metrics["m2"]]
        )
        self.outcome_constraint = OutcomeConstraint(
            metric=self.metrics["m2"], op=ComparisonOp.GEQ, bound=-0.25
        )
        self.additional_outcome_constraint = OutcomeConstraint(
            metric=self.metrics["m2"], op=ComparisonOp.LEQ, bound=0.25
        )
        self.outcome_constraints = [
            self.outcome_constraint,
            self.additional_outcome_constraint,
        ]

    def testInit(self):
        config1 = OptimizationConfig(
            objective=self.objective, outcome_constraints=self.outcome_constraints
        )
        self.assertEqual(str(config1), CONFIG_STR)
        with self.assertRaises(ValueError):
            config1.objective = self.m2_objective
        # updating constraints is fine.
        config1.outcome_constraints = [self.outcome_constraint]
        self.assertEqual(len(config1.metrics), 2)

        # objective without outcome_constraints is also supported
        config2 = OptimizationConfig(objective=self.objective)
        self.assertEqual(config2.outcome_constraints, [])

        # setting objective is fine too, if it's compatible with constraints..
        config2.objective = self.m2_objective
        # setting incompatible constraints is not fine.
        with self.assertRaises(ValueError):
            config2.outcome_constraints = self.outcome_constraints

    def testEq(self):
        config1 = OptimizationConfig(
            objective=self.objective, outcome_constraints=self.outcome_constraints
        )
        config2 = OptimizationConfig(
            objective=self.objective, outcome_constraints=self.outcome_constraints
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

    def testConstraintValidation(self):
        # Can't constrain on objective metric.
        objective_constraint = OutcomeConstraint(
            metric=self.objective.metric, op=ComparisonOp.GEQ, bound=0
        )
        with self.assertRaises(ValueError):
            OptimizationConfig(
                objective=self.objective, outcome_constraints=[objective_constraint]
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

    def testClone(self):
        config1 = OptimizationConfig(
            objective=self.objective, outcome_constraints=self.outcome_constraints
        )
        self.assertEqual(config1, config1.clone())

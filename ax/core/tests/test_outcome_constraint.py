#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

from ax.core.metric import Metric
from ax.core.outcome_constraint import (
    CONSTRAINT_WARNING_MESSAGE,
    LOWER_BOUND_MISMATCH,
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
    UPPER_BOUND_MISMATCH,
)
from ax.core.types import ComparisonOp
from ax.utils.common.testutils import TestCase


OUTCOME_CONSTRAINT_PATH = "ax.core.outcome_constraint"


class OutcomeConstraintTest(TestCase):
    def setUp(self) -> None:
        self.minimize_metric = Metric(name="bar", lower_is_better=True)
        self.maximize_metric = Metric(name="baz", lower_is_better=False)
        self.bound = 0
        simple_metric = Metric(name="foo")
        self.constraint = OutcomeConstraint(
            metric=simple_metric, op=ComparisonOp.GEQ, bound=self.bound
        )

    def testEq(self) -> None:
        constraint1 = OutcomeConstraint(
            metric=Metric(name="foo"), op=ComparisonOp.GEQ, bound=self.bound
        )
        constraint2 = OutcomeConstraint(
            metric=Metric(name="foo"), op=ComparisonOp.GEQ, bound=self.bound
        )
        self.assertEqual(constraint1, constraint2)

        constraint3 = OutcomeConstraint(
            metric=Metric(name="foo"), op=ComparisonOp.LEQ, bound=self.bound
        )
        self.assertNotEqual(constraint1, constraint3)

    def testValidMutations(self) -> None:
        # updating constraint metric is ok as long as lower_is_better is compatible.
        self.constraint.metric = self.maximize_metric
        self.constraint.op = ComparisonOp.LEQ
        self.assertEqual(self.constraint.metric.name, "baz")

    def testOutcomeConstraintFail(self) -> None:
        logger_name = OUTCOME_CONSTRAINT_PATH + ".logger"
        with mock.patch(logger_name) as mock_warning:
            OutcomeConstraint(
                metric=self.minimize_metric, op=ComparisonOp.GEQ, bound=self.bound
            )
            mock_warning.debug.assert_called_once_with(
                CONSTRAINT_WARNING_MESSAGE.format(**LOWER_BOUND_MISMATCH)
            )
        with mock.patch(logger_name) as mock_warning:
            OutcomeConstraint(
                metric=self.maximize_metric, op=ComparisonOp.LEQ, bound=self.bound
            )
            mock_warning.debug.assert_called_once_with(
                CONSTRAINT_WARNING_MESSAGE.format(**UPPER_BOUND_MISMATCH)
            )

    def testSortable(self) -> None:
        constraint1 = OutcomeConstraint(
            metric=Metric(name="foo"), op=ComparisonOp.LEQ, bound=self.bound
        )
        constraint2 = OutcomeConstraint(
            metric=Metric(name="foo"), op=ComparisonOp.GEQ, bound=self.bound
        )
        self.assertTrue(constraint1 < constraint2)


class ObjectiveThresholdTest(TestCase):
    def setUp(self) -> None:
        self.minimize_metric = Metric(name="bar", lower_is_better=True)
        self.maximize_metric = Metric(name="baz", lower_is_better=False)
        self.ambiguous_metric = Metric(name="buz")
        self.bound = 0
        self.threshold = ObjectiveThreshold(
            metric=self.maximize_metric, op=ComparisonOp.GEQ, bound=self.bound
        )

    def testEq(self) -> None:
        threshold1 = ObjectiveThreshold(metric=self.minimize_metric, bound=self.bound)
        threshold2 = ObjectiveThreshold(metric=self.minimize_metric, bound=self.bound)
        self.assertEqual(threshold1, threshold2)

        threshold3 = ObjectiveThreshold(
            metric=self.minimize_metric, bound=self.bound, relative=False
        )
        self.assertNotEqual(threshold1, threshold3)

        constraint3 = OutcomeConstraint(
            metric=self.minimize_metric, op=ComparisonOp.LEQ, bound=self.bound
        )
        self.assertNotEqual(threshold1, constraint3)

    def testValidMutations(self) -> None:
        # updating constraint metric is ok as long as lower_is_better is compatible.
        self.threshold.metric = self.ambiguous_metric
        self.threshold.op = ComparisonOp.LEQ
        self.assertEqual(self.threshold.metric.name, "buz")

    def testObjectiveThresholdFail(self) -> None:
        logger_name = OUTCOME_CONSTRAINT_PATH + ".logger"
        with mock.patch(logger_name) as mock_warning:
            ObjectiveThreshold(
                metric=self.minimize_metric, op=ComparisonOp.GEQ, bound=self.bound
            )
            mock_warning.debug.assert_called_once_with(
                CONSTRAINT_WARNING_MESSAGE.format(**LOWER_BOUND_MISMATCH)
            )
        with mock.patch(logger_name) as mock_warning:
            ObjectiveThreshold(
                metric=self.maximize_metric, op=ComparisonOp.LEQ, bound=self.bound
            )
            mock_warning.debug.assert_called_once_with(
                CONSTRAINT_WARNING_MESSAGE.format(**UPPER_BOUND_MISMATCH)
            )

    def testRelativize(self) -> None:
        self.assertTrue(
            ObjectiveThreshold(
                metric=self.maximize_metric, op=ComparisonOp.LEQ, bound=self.bound
            ).relative
        )
        self.assertTrue(
            ObjectiveThreshold(
                metric=self.maximize_metric,
                op=ComparisonOp.LEQ,
                bound=self.bound,
                relative=True,
            ).relative
        )
        self.assertFalse(
            ObjectiveThreshold(
                metric=self.maximize_metric,
                op=ComparisonOp.LEQ,
                bound=self.bound,
                relative=False,
            ).relative
        )


class ScalarizedOutcomeConstraintTest(TestCase):
    def setUp(self) -> None:
        self.metrics = [
            Metric(name="m1", lower_is_better=True),
            Metric(name="m2", lower_is_better=True),
            Metric(name="m3", lower_is_better=True),
        ]
        self.weights = [0.1, 0.3, 0.6]
        self.bound = 0
        self.constraint = ScalarizedOutcomeConstraint(
            metrics=self.metrics,
            weights=self.weights,
            op=ComparisonOp.GEQ,
            bound=self.bound,
        )

    def testInit(self) -> None:
        self.assertListEqual(self.constraint.metrics, self.metrics)
        self.assertListEqual(self.constraint.weights, self.weights)
        self.assertEqual(len(list(self.constraint.metric_weights)), len(self.metrics))
        self.assertEqual(
            str(self.constraint),
            (
                "ScalarizedOutcomeConstraint(metric_names=['m1', 'm2', 'm3'], "
                "weights=[0.1, 0.3, 0.6], >= 0%)"
            ),
        )
        # check that weights are set uniformly by default
        con = ScalarizedOutcomeConstraint(
            metrics=[
                Metric(name="m1", lower_is_better=True),
                Metric(name="m2", lower_is_better=True),
            ],
            op=ComparisonOp.LEQ,
            bound=self.bound,
        )
        self.assertListEqual(con.weights, [0.5, 0.5])

    def testEq(self) -> None:
        constraint1 = ScalarizedOutcomeConstraint(
            metrics=self.metrics,
            weights=self.weights,
            op=ComparisonOp.GEQ,
            bound=self.bound,
        )
        self.assertEqual(constraint1, self.constraint)

        constraint2 = ScalarizedOutcomeConstraint(
            metrics=self.metrics,
            weights=[0.2, 0.2, 0.6],
            op=ComparisonOp.LEQ,
            bound=self.bound,
        )
        self.assertNotEqual(constraint2, self.constraint)

    def testClone(self) -> None:
        self.assertEqual(self.constraint, self.constraint.clone())

    def testValidMutations(self) -> None:
        # updating constraint metric is ok as long as lower_is_better is compatible.
        self.constraint.metrics = [
            Metric(name="m2"),
            Metric(name="m4"),
        ]
        self.constraint.op = ComparisonOp.LEQ

    def testRaiseError(self) -> None:
        # set a wrong weights
        with self.assertRaises(ValueError):
            ScalarizedOutcomeConstraint(
                metrics=self.metrics,
                weights=[0.2, 0.8],
                op=ComparisonOp.LEQ,
                bound=self.bound,
            )

        with self.assertRaises(NotImplementedError):
            # pyre-fixme[7]: Expected `None` but got `Metric`.
            return self.constraint.metric

        with self.assertRaises(NotImplementedError):
            self.constraint.metric = self.metrics[0]

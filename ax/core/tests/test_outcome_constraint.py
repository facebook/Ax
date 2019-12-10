#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

from ax.core.metric import Metric
from ax.core.outcome_constraint import (
    CONSTRAINT_WARNING_MESSAGE,
    LOWER_BOUND_MISMATCH,
    UPPER_BOUND_MISMATCH,
    OutcomeConstraint,
)
from ax.core.types import ComparisonOp
from ax.utils.common.testutils import TestCase


OUTCOME_CONSTRAINT_PATH = "ax.core.outcome_constraint"


class OutcomeConstraintTest(TestCase):
    def setUp(self):
        self.minimize_metric = Metric(name="bar", lower_is_better=True)
        self.maximize_metric = Metric(name="baz", lower_is_better=False)
        self.bound = 0
        simple_metric = Metric(name="foo")
        self.constraint = OutcomeConstraint(
            metric=simple_metric, op=ComparisonOp.GEQ, bound=self.bound
        )

    def testEq(self):
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

    def testValidMutations(self):
        # updating constraint metric is ok as long as lower_is_better is compatible.
        self.constraint.metric = self.maximize_metric
        self.constraint.op = ComparisonOp.LEQ
        self.assertEqual(self.constraint.metric.name, "baz")

    def testOutcomeConstraintFail(self):
        logger_name = OUTCOME_CONSTRAINT_PATH + ".logger"
        with mock.patch(logger_name) as mock_warning:
            OutcomeConstraint(
                metric=self.minimize_metric, op=ComparisonOp.GEQ, bound=self.bound
            )
            mock_warning.warning.assert_called_once_with(
                CONSTRAINT_WARNING_MESSAGE.format(**LOWER_BOUND_MISMATCH)
            )
        with mock.patch(logger_name) as mock_warning:
            OutcomeConstraint(
                metric=self.maximize_metric, op=ComparisonOp.LEQ, bound=self.bound
            )
            mock_warning.warning.assert_called_once_with(
                CONSTRAINT_WARNING_MESSAGE.format(**UPPER_BOUND_MISMATCH)
            )

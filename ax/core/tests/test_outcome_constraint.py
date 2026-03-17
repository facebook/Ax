#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings

from ax.core.metric import Metric
from ax.core.outcome_constraint import (
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase


class OutcomeConstraintTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.constraint = OutcomeConstraint(expression="foo >= 0", relative=False)

    def test_ExpressionInit(self) -> None:
        """Test creating constraints with expression strings."""
        # GEQ constraint
        oc = OutcomeConstraint(expression="qps >= 700", relative=False)
        self.assertEqual(oc.expression, "qps >= 700")
        self.assertEqual(oc.metric_names, ["qps"])
        self.assertEqual(oc.op, ComparisonOp.GEQ)
        self.assertEqual(oc.bound, 700.0)
        self.assertFalse(oc.relative)

        # LEQ constraint
        oc2 = OutcomeConstraint(expression="loss <= 0.5", relative=False)
        self.assertEqual(oc2.metric_names, ["loss"])
        self.assertEqual(oc2.op, ComparisonOp.LEQ)
        self.assertEqual(oc2.bound, 0.5)
        self.assertFalse(oc2.relative)

        # Relative constraint via * baseline
        oc3 = OutcomeConstraint(expression="latency <= 1.05 * baseline")
        self.assertEqual(oc3.metric_names, ["latency"])
        self.assertEqual(oc3.bound, 5.0)  # (1.05 - 1) * 100
        self.assertTrue(oc3.relative)

        # relative kwarg without baseline in expression has no effect
        oc4 = OutcomeConstraint(expression="qps >= 700", relative=True)
        self.assertFalse(oc4.relative)  # expression has no baseline

        # No expression should error
        with self.assertRaisesRegex(UserInputError, "expression string is required"):
            OutcomeConstraint()
        with self.assertRaisesRegex(UserInputError, "expression string is required"):
            OutcomeConstraint(expression=None)

        # Missing operator should error
        with self.assertRaisesRegex(UserInputError, "Expected an inequality"):
            OutcomeConstraint(expression="qps 700")

    def test_DeprecatedMetricInit(self) -> None:
        """Test the deprecated metric-based init path."""
        metric = Metric(name="foo")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            oc = OutcomeConstraint(
                metric=metric, op=ComparisonOp.GEQ, bound=0, relative=True
            )
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            self.assertTrue(len(deprecation_warnings) > 0)

        self.assertEqual(oc.metric_names, ["foo"])
        self.assertEqual(oc.op, ComparisonOp.GEQ)
        self.assertEqual(oc.bound, 0.0)
        self.assertTrue(oc.relative)

        # Cannot specify both expression and metric
        with self.assertRaisesRegex(UserInputError, "Cannot specify both"):
            OutcomeConstraint(
                expression="foo >= 0", metric=metric, op=ComparisonOp.GEQ, bound=0
            )

        # Must specify op and bound with metric
        with self.assertRaisesRegex(UserInputError, "Must specify `op`"):
            OutcomeConstraint(metric=metric, bound=0)
        with self.assertRaisesRegex(UserInputError, "Must specify `bound`"):
            OutcomeConstraint(metric=metric, op=ComparisonOp.GEQ)

    def test_Eq(self) -> None:
        constraint1 = OutcomeConstraint(expression="foo >= 0", relative=False)
        constraint2 = OutcomeConstraint(expression="foo >= 0", relative=False)
        self.assertEqual(constraint1, constraint2)

        constraint3 = OutcomeConstraint(expression="foo <= 0", relative=False)
        self.assertNotEqual(constraint1, constraint3)

    def test_ValidMutations(self) -> None:
        # updating constraint metric is ok as long as lower_is_better is compatible.
        self.constraint.metric = self.maximize_metric
        self.constraint.op = ComparisonOp.LEQ
        self.assertEqual(self.constraint.metric.name, "baz")
        self.assertEqual(self.constraint.metric.signature, "baz")

    def test_OutcomeConstraintFail(self) -> None:
        logger_name = OUTCOME_CONSTRAINT_PATH + ".logger"
        cases = [
            (self.minimize_metric, ComparisonOp.GEQ, LOWER_BOUND_MISMATCH, "bar"),
            (self.maximize_metric, ComparisonOp.LEQ, UPPER_BOUND_MISMATCH, "baz"),
        ]
        for metric, op, mismatch, name in cases:
            with self.subTest(metric=name):
                with mock.patch(logger_name) as mock_warning:
                    OutcomeConstraint(metric=metric, op=op, bound=self.bound)
                mock_warning.debug.assert_called_once_with(
                    CONSTRAINT_WARNING_MESSAGE.format(**mismatch, name=name)
                )

    def test_Sortable(self) -> None:
        constraint1 = OutcomeConstraint(expression="foo <= 0", relative=False)
        constraint2 = OutcomeConstraint(expression="foo >= 0", relative=False)
        self.assertTrue(constraint1 < constraint2)

    def test_validate_constraint(self) -> None:
        # Validate constraint logic for (lower_is_better, bound, op) combinations.
        # Only (lower_is_better=False, bound<0, GEQ) and (lower_is_better=True,
        # bound>0, LEQ) are considered valid/sensible constraints.
        cases = [
            (False, -3, ComparisonOp.GEQ, True),
            (False, -3, ComparisonOp.LEQ, False),
            (False, 3, ComparisonOp.GEQ, False),
            (False, 3, ComparisonOp.LEQ, False),
            (True, 3, ComparisonOp.LEQ, True),
            (True, 3, ComparisonOp.GEQ, False),
            (True, -3, ComparisonOp.LEQ, False),
            (True, -3, ComparisonOp.GEQ, False),
        ]
        for lower_is_better, bound, op, expected in cases:
            with self.subTest(lower_is_better=lower_is_better, bound=bound, op=op.name):
                metric = Metric(
                    name=f"metric_lib{'1' if lower_is_better else '0'}",
                    lower_is_better=lower_is_better,
                )
                oc = OutcomeConstraint(metric, bound=bound, relative=True, op=op)
                self.assertEqual(oc._validate_constraint()[0], expected)


class ObjectiveThresholdTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.minimize_metric = Metric(name="bar", lower_is_better=True)
        self.maximize_metric = Metric(name="baz", lower_is_better=False)
        self.ambiguous_metric = Metric(name="buz")

    def test_Init(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ot = ObjectiveThreshold(
                metric=self.maximize_metric,
                bound=0.5,
                relative=False,
            )
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            self.assertTrue(len(deprecation_warnings) > 0)

        # GEQ inferred from lower_is_better=False
        self.assertEqual(ot.op, ComparisonOp.GEQ)
        self.assertEqual(ot.bound, 0.5)
        self.assertFalse(ot.relative)
        self.assertEqual(ot.metric_names, ["baz"])

    def test_OpInference(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ot_min = ObjectiveThreshold(metric=self.minimize_metric, bound=0.0)
        self.assertEqual(ot_min.op, ComparisonOp.LEQ)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ot_max = ObjectiveThreshold(metric=self.maximize_metric, bound=0.0)
        self.assertEqual(ot_max.op, ComparisonOp.GEQ)

    def test_OpManualOverride(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ot = ObjectiveThreshold(
                metric=self.maximize_metric,
                bound=0.0,
                op=ComparisonOp.LEQ,
            )
        self.assertEqual(ot.op, ComparisonOp.LEQ)

    def test_AmbiguousMetricRequiresOp(self) -> None:
        with self.assertRaises(ValueError):
            ObjectiveThreshold(metric=self.ambiguous_metric, bound=0.0)

    def test_Eq(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            threshold1 = ObjectiveThreshold(metric=self.minimize_metric, bound=0.0)
            threshold2 = ObjectiveThreshold(metric=self.minimize_metric, bound=0.0)
        self.assertEqual(threshold1, threshold2)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            threshold3 = ObjectiveThreshold(
                metric=self.minimize_metric, bound=0.0, relative=False
            )
        self.assertNotEqual(threshold1, threshold3)

        constraint3 = OutcomeConstraint(
            metric=self.minimize_metric, op=ComparisonOp.LEQ, bound=self.bound
        )
        self.assertNotEqual(threshold1, constraint3)

    def test_ValidMutations(self) -> None:
        # updating constraint metric is ok as long as lower_is_better is compatible.
        self.threshold.metric = self.ambiguous_metric
        self.threshold.op = ComparisonOp.LEQ
        self.assertEqual(self.threshold.metric.name, "buz")
        self.assertEqual(self.threshold.metric.signature, "buz")

    def test_ObjectiveThresholdFail(self) -> None:
        logger_name = OUTCOME_CONSTRAINT_PATH + ".logger"
        cases = [
            (self.minimize_metric, ComparisonOp.GEQ, LOWER_BOUND_MISMATCH, "bar"),
            (self.maximize_metric, ComparisonOp.LEQ, UPPER_BOUND_MISMATCH, "baz"),
        ]
        for metric, op, mismatch, name in cases:
            with self.subTest(metric=name):
                with mock.patch(logger_name) as mock_warning:
                    ObjectiveThreshold(metric=metric, op=op, bound=self.bound)
                mock_warning.debug.assert_called_once_with(
                    CONSTRAINT_WARNING_MESSAGE.format(**mismatch, name=name)
                )

    def test_Relativize(self) -> None:
        cases = [
            ("default", {}, True),
            ("explicit_true", {"relative": True}, True),
            ("explicit_false", {"relative": False}, False),
        ]
        for label, kwargs, expected in cases:
            with self.subTest(label=label):
                ot = ObjectiveThreshold(
                    metric=self.maximize_metric,
                    op=ComparisonOp.LEQ,
                    bound=self.bound,
                    **kwargs,
                )
                self.assertEqual(ot.relative, expected)


class ScalarizedOutcomeConstraintTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.metrics = [
            Metric(name="m1"),
            Metric(name="m2"),
            Metric(name="m3"),
        ]
        self.weights = [0.1, 0.3, 0.6]

    def test_Init(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            soc = ScalarizedOutcomeConstraint(
                metrics=self.metrics,
                weights=self.weights,
                op=ComparisonOp.GEQ,
                bound=0,
            )
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            self.assertTrue(len(deprecation_warnings) > 0)

        self.assertEqual(soc.weights, self.weights)
        self.assertEqual(sorted(soc.metric_names), ["m1", "m2", "m3"])
        self.assertEqual(soc.op, ComparisonOp.GEQ)
        self.assertEqual(soc.bound, 0.0)

    def test_DefaultWeights(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            soc = ScalarizedOutcomeConstraint(
                metrics=[Metric(name="m1"), Metric(name="m2")],
                op=ComparisonOp.LEQ,
                bound=0,
            )
        self.assertEqual(soc.weights, [0.5, 0.5])

    def test_WeightMismatchRaises(self) -> None:
        with self.assertRaises(ValueError):
            ScalarizedOutcomeConstraint(
                metrics=self.metrics,
                weights=[0.2, 0.8],
                op=ComparisonOp.LEQ,
                bound=0,
            )

    def test_Eq(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            soc1 = ScalarizedOutcomeConstraint(
                metrics=self.metrics,
                weights=self.weights,
                op=ComparisonOp.GEQ,
                bound=0,
            )
            soc2 = ScalarizedOutcomeConstraint(
                metrics=self.metrics,
                weights=self.weights,
                op=ComparisonOp.GEQ,
                bound=0,
            )
        self.assertEqual(soc1, soc2)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            soc3 = ScalarizedOutcomeConstraint(
                metrics=self.metrics,
                weights=[0.2, 0.2, 0.6],
                op=ComparisonOp.LEQ,
                bound=0,
            )
        self.assertNotEqual(soc1, soc3)

    def test_Clone(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            soc = ScalarizedOutcomeConstraint(
                metrics=self.metrics,
                weights=self.weights,
                op=ComparisonOp.GEQ,
                bound=0,
            )
        cloned = soc.clone()
        self.assertEqual(soc, cloned)
        self.assertIsNot(soc, cloned)
        self.assertIsInstance(cloned, ScalarizedOutcomeConstraint)
        self.assertEqual(cloned.weights, self.weights)

    def test_Repr(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            soc = ScalarizedOutcomeConstraint(
                metrics=self.metrics,
                weights=self.weights,
                op=ComparisonOp.GEQ,
                bound=0,
            )
        repr_str = str(soc)
        self.assertTrue(repr_str.startswith("ScalarizedOutcomeConstraint("))
        self.assertIn(">=", repr_str)

    def test_IsInstance(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            soc = ScalarizedOutcomeConstraint(
                metrics=self.metrics,
                weights=self.weights,
                op=ComparisonOp.GEQ,
                bound=0,
            )
        self.assertIsInstance(soc, OutcomeConstraint)
        self.assertIsInstance(soc, ScalarizedOutcomeConstraint)

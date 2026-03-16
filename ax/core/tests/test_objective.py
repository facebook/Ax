#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings

from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase


class ObjectiveTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.metrics = {
            "m1": Metric(name="m1"),
            "m2": Metric(name="m2", lower_is_better=True),
            "m3": Metric(name="m3", lower_is_better=False),
        }
        # Expression-based objectives
        self.objective = Objective(expression="m1")
        self.objective_minimize = Objective(expression="-m1")
        self.multi_objective = Objective(expression="-m1, -m2, m3")
        self.scalarized_objective = Objective(expression="2*m1 + m2")

    def test_ExpressionInit(self) -> None:
        """Test creating objectives with expression strings."""
        # Single metric maximize
        obj = Objective(expression="accuracy")
        self.assertEqual(obj.expression, "accuracy")
        self.assertEqual(obj.metric_names, ["accuracy"])
        self.assertFalse(obj.minimize)
        self.assertTrue(obj.is_single_objective)
        self.assertFalse(obj.is_multi_objective)
        self.assertFalse(obj.is_scalarized_objective)

        # Single metric minimize
        obj_min = Objective(expression="-loss")
        self.assertEqual(obj_min.expression, "-loss")
        self.assertEqual(obj_min.metric_names, ["loss"])
        self.assertTrue(obj_min.minimize)

        # Scalarized
        obj_scalar = Objective(expression="2*acc + recall")
        self.assertEqual(obj_scalar.metric_names, ["acc", "recall"])
        self.assertTrue(obj_scalar.is_scalarized_objective)
        self.assertFalse(obj_scalar.is_multi_objective)
        self.assertFalse(obj_scalar.is_single_objective)
        self.assertEqual(obj_scalar.metric_weights, [("acc", 2.0), ("recall", 1.0)])

        # Multi-objective
        obj_multi = Objective(expression="acc, -loss")
        self.assertEqual(obj_multi.metric_names, ["acc", "loss"])
        self.assertTrue(obj_multi.is_multi_objective)
        self.assertFalse(obj_multi.is_single_objective)
        self.assertFalse(obj_multi.is_scalarized_objective)

        # No expression should error
        with self.assertRaisesRegex(UserInputError, "expression string is required"):
            Objective()
        with self.assertRaisesRegex(UserInputError, "expression string is required"):
            Objective(expression=None)

        # Cannot specify both expression and metric
        with self.assertRaisesRegex(UserInputError, "Cannot specify both"):
            Objective(expression="m1", metric=self.metrics["m1"])

    def test_DeprecatedMetricInit(self) -> None:
        """Test the deprecated metric-based init path."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = Objective(metric=self.metrics["m2"], minimize=True)
            self.assertEqual(len(w), 1)
            self.assertIn("deprecated", str(w[0].message).lower())
        self.assertEqual(obj.expression, "-m2")
        self.assertTrue(obj.minimize)
        self.assertEqual(obj.metric_names, ["m2"])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj2 = Objective(metric=self.metrics["m3"], minimize=False)
            self.assertEqual(len(w), 1)
        self.assertEqual(obj2.expression, "m3")
        self.assertFalse(obj2.minimize)

        # Test lower_is_better validation still works
        with self.assertRaisesRegex(UserInputError, "does not specify"):
            Objective(metric=self.metrics["m1"])

        with self.assertRaisesRegex(
            UserInputError, "doesn't match the specified optimization direction"
        ):
            Objective(metric=self.metrics["m2"], minimize=False)

    def test_MetricWeights(self) -> None:
        """Test metric_weights property."""
        obj = Objective(expression="m1")
        self.assertEqual(obj.metric_weights, [("m1", 1.0)])

        obj_neg = Objective(expression="-m1")
        self.assertEqual(obj_neg.metric_weights, [("m1", -1.0)])

        obj_scalar = Objective(expression="2*m1 + 3*m2")
        self.assertEqual(obj_scalar.metric_weights, [("m1", 2.0), ("m2", 3.0)])

        # metric_weights works for multi-objective too (flat list)
        obj_multi = Objective(expression="m1, m2")
        self.assertEqual(obj_multi.metric_weights, [("m1", 1.0), ("m2", 1.0)])

        obj_multi2 = Objective(expression="2*m1, -m2")
        self.assertEqual(obj_multi2.metric_weights, [("m1", 2.0), ("m2", -1.0)])

    def test_Minimize(self) -> None:
        """Test minimize property."""
        self.assertFalse(Objective(expression="m1").minimize)
        self.assertTrue(Objective(expression="-m1").minimize)

        # Should raise for scalarized
        with self.assertRaisesRegex(UserInputError, "single-metric"):
            Objective(expression="m1 + m2").minimize

        # Should raise for multi
        with self.assertRaisesRegex(UserInputError, "single-metric"):
            Objective(expression="m1, m2").minimize

    def test_Clone(self) -> None:
        """Test clone method."""
        obj = Objective(expression="2*m1 + m2")
        cloned = obj.clone()
        self.assertEqual(obj, cloned)
        self.assertIsNot(obj, cloned)
        self.assertEqual(obj.expression, cloned.expression)

    def test_Repr(self) -> None:
        """Test __repr__."""
        self.assertEqual(str(self.objective), 'Objective(expression="m1")')
        self.assertEqual(str(self.objective_minimize), 'Objective(expression="-m1")')

    def test_GetUnconstrainableMetricNames(self) -> None:
        """Test get_unconstrainable_metric_names."""
        self.assertEqual(self.objective.get_unconstrainable_metric_names(), ["m1"])
        self.assertEqual(
            self.multi_objective.get_unconstrainable_metric_names(),
            ["m1", "m2", "m3"],
        )

    def test_DeprecatedProperties(self) -> None:
        """Test that removed deprecated properties raise AttributeError."""

        # Use a helper to prevent Pyre from resolving the attribute name
        # statically; these attributes were intentionally removed.
        def _getattr(obj: object, name: str) -> object:
            return getattr(obj, name)

        with self.assertRaises(AttributeError):
            _getattr(self.objective, "metric")

        with self.assertRaises(AttributeError):
            _getattr(self.objective, "metrics")

        with self.assertRaises(AttributeError):
            _getattr(self.objective, "metric_signatures")

        with self.assertRaises(AttributeError):
            _getattr(self.objective, "get_unconstrainable_metrics")

    def test_MultiObjective(self) -> None:
        """Test deprecated MultiObjective subclass."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            objectives = [
                Objective(expression="-m1"),
                Objective(expression="-m2"),
                Objective(expression="m3"),
            ]
            mo = MultiObjective(objectives=objectives)
            # Check deprecation warning was emitted
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            self.assertTrue(len(deprecation_warnings) > 0)

        self.assertTrue(mo.is_multi_objective)
        self.assertEqual(mo.metric_names, ["m1", "m2", "m3"])

        # Verify metric_weights reflect the sub-objective directions
        self.assertEqual(mo.metric_weights, [("m1", -1.0), ("m2", -1.0), ("m3", 1.0)])

        # Clone returns a plain Objective with the same expression
        cloned = mo.clone()
        self.assertEqual(mo.expression, cloned.expression)
        self.assertEqual(mo.metric_names, cloned.metric_names)

        # Test repr
        self.assertEqual(str(mo), 'Objective(expression="-m1, -m2, m3")')

        # Test that ScalarizedObjective is not allowed in MultiObjective
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            scalarized = ScalarizedObjective(
                metrics=[self.metrics["m1"], self.metrics["m2"]],
                minimize=True,
            )

        with self.assertRaisesRegex(
            NotImplementedError,
            "Scalarized objectives are not supported for a `MultiObjective`.",
        ):
            MultiObjective(objectives=[scalarized])

        with self.assertRaisesRegex(
            NotImplementedError,
            "Scalarized objectives are not supported for a `MultiObjective`.",
        ):
            MultiObjective(
                objectives=[
                    Objective(expression="-m1"),
                    scalarized,
                    Objective(expression="m3"),
                ]
            )

    def test_ScalarizedObjective(self) -> None:
        """Test deprecated ScalarizedObjective subclass."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            so = ScalarizedObjective(
                metrics=[self.metrics["m1"], self.metrics["m2"]],
                minimize=True,
            )
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            self.assertTrue(len(deprecation_warnings) > 0)

        # Should build a negated expression since minimize=True
        self.assertEqual(so.expression, "-m1 - m2")
        self.assertEqual(so.metric_names, ["m1", "m2"])

        # metric_weights should return (name, weight) tuples
        self.assertEqual(so.metric_weights, [("m1", -1.0), ("m2", -1.0)])

        # Clone returns a plain Objective with the same expression
        cloned = so.clone()
        self.assertEqual(so.expression, cloned.expression)
        self.assertEqual(so.metric_names, cloned.metric_names)

        # Weight validation
        with self.assertRaises(ValueError):
            ScalarizedObjective(
                metrics=[self.metrics["m1"], self.metrics["m2"]], weights=[1.0]
            )

        # lower_is_better validation
        with self.assertRaisesRegex(
            ValueError,
            "Metric with name m2 specifies `lower_is_better` = "
            "True, which doesn't match the specified optimization direction.",
        ):
            ScalarizedObjective(
                metrics=[self.metrics["m1"], self.metrics["m2"]],
                minimize=False,
            )

    def test_ScalarizedObjective_expression(self) -> None:
        """Test the expression property of ScalarizedObjective."""
        test_cases = [
            # (name, metrics, weights, minimize, expected_expression)
            (
                "default_weights_maximize",
                [self.metrics["m1"], self.metrics["m3"]],
                None,
                False,
                "m1 + m3",
            ),
            (
                "custom_weights_maximize",
                [self.metrics["m1"], self.metrics["m3"]],
                [1.0, 2.0],
                False,
                "m1 + 2.0*m3",
            ),
            (
                "negative_weight",
                [Metric(name="accuracy"), Metric(name="latency")],
                [1.0, -0.5],
                False,
                "accuracy - 0.5*latency",
            ),
            (
                "negative_one_weight",
                [Metric(name="reward"), Metric(name="cost")],
                [2.0, -1.0],
                False,
                "2.0*reward - cost",
            ),
        ]

        for name, metrics, weights, minimize, expected in test_cases:
            with self.subTest(name=name, expected=expected):
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    if weights is None:
                        obj = ScalarizedObjective(metrics=metrics, minimize=minimize)
                    else:
                        obj = ScalarizedObjective(
                            metrics=metrics, weights=weights, minimize=minimize
                        )
                self.assertEqual(obj.expression, expected)

    def test_IsInstanceChecks(self) -> None:
        """Test that isinstance checks work for deprecated subclasses."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            mo = MultiObjective(
                objectives=[
                    Objective(expression="m1"),
                    Objective(expression="-m2"),
                ]
            )
            so = ScalarizedObjective(
                metrics=[self.metrics["m1"], self.metrics["m3"]],
                minimize=False,
            )

        self.assertIsInstance(mo, Objective)
        self.assertIsInstance(mo, MultiObjective)
        self.assertIsInstance(so, Objective)
        self.assertIsInstance(so, ScalarizedObjective)

    def test_UniqueId(self) -> None:
        """Test _unique_id used for sorting."""
        obj = Objective(expression="m1")
        self.assertEqual(obj._unique_id, str(obj))

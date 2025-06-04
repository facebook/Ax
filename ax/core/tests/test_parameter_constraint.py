#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import (
    ComparisonOp,
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.utils.common.testutils import TestCase


class ParameterConstraintTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.constraint = ParameterConstraint(
            constraint_dict={"x": 2.0, "y": -3.0}, bound=6.0
        )
        self.constraint_repr = "ParameterConstraint(2.0*x + -3.0*y <= 6.0)"

    def test_Eq(self) -> None:
        constraint1 = ParameterConstraint(
            constraint_dict={"x": 2.0, "y": -3.0}, bound=6.0
        )
        constraint2 = ParameterConstraint(
            constraint_dict={"y": -3.0, "x": 2.0}, bound=6.0
        )
        self.assertEqual(constraint1, constraint2)

        constraint3 = ParameterConstraint(
            constraint_dict={"x": 2.0, "y": -5.0}, bound=6.0
        )
        self.assertNotEqual(constraint1, constraint3)

    def test_Properties(self) -> None:
        self.assertEqual(self.constraint.constraint_dict["x"], 2.0)
        self.assertEqual(self.constraint.bound, 6.0)

    def test_Repr(self) -> None:
        self.assertEqual(str(self.constraint), self.constraint_repr)

    def test_Validate(self) -> None:
        parameters = {"x": 4, "z": 3.0}
        with self.assertRaises(ValueError):
            self.constraint.check(parameters)

        # check slack constraint
        parameters = {"x": 4, "y": 1.0}
        self.assertTrue(self.constraint.check(parameters))

        # check tight constraint (within numerical tolerance)
        parameters = {"x": 4, "y": (2 - 0.5e-8) / 3}
        self.assertTrue(self.constraint.check(parameters))

        # check violated constraint
        parameters = {"x": 4, "y": (2 - 0.5e-6) / 3}
        self.assertFalse(self.constraint.check(parameters))

    def test_Clone(self) -> None:
        constraint_clone = self.constraint.clone()
        self.assertEqual(self.constraint.bound, constraint_clone.bound)

        constraint_clone._bound = 7.0
        self.assertNotEqual(self.constraint.bound, constraint_clone.bound)

    def test_CloneWithTransformedParameters(self) -> None:
        constraint_clone = self.constraint.clone_with_transformed_parameters(
            transformed_parameters={}
        )
        self.assertEqual(self.constraint.bound, constraint_clone.bound)

        constraint_clone._bound = 7.0
        self.assertNotEqual(self.constraint.bound, constraint_clone.bound)

    def test_Sortable(self) -> None:
        constraint1 = ParameterConstraint(
            constraint_dict={"x": 2.0, "y": -3.0}, bound=1.0
        )
        constraint2 = ParameterConstraint(
            constraint_dict={"y": -3.0, "x": 2.0}, bound=6.0
        )
        self.assertTrue(constraint1 < constraint2)


class OrderConstraintTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.x = RangeParameter("x", ParameterType.INT, lower=0, upper=1)
        self.y = RangeParameter("y", ParameterType.INT, lower=0, upper=1)
        self.constraint = OrderConstraint(
            lower_parameter=self.x, upper_parameter=self.y
        )
        self.constraint_repr = "OrderConstraint(x <= y)"

    def test_Properties(self) -> None:
        self.assertEqual(self.constraint.lower_parameter.name, "x")
        self.assertEqual(self.constraint.upper_parameter.name, "y")

    def test_Repr(self) -> None:
        self.assertEqual(str(self.constraint), self.constraint_repr)

    def test_Validate(self) -> None:
        self.assertTrue(self.constraint.check({"x": 0, "y": 1}))
        self.assertTrue(self.constraint.check({"x": 1, "y": 1}))
        self.assertFalse(self.constraint.check({"x": 1, "y": 0}))

    def test_Clone(self) -> None:
        constraint_clone = self.constraint.clone()
        self.assertEqual(
            self.constraint.lower_parameter, constraint_clone.lower_parameter
        )

        constraint_clone._lower_parameter = self.y
        self.assertNotEqual(
            self.constraint.lower_parameter, constraint_clone.lower_parameter
        )

    def test_CloneWithTransformedParameters(self) -> None:
        constraint_clone = self.constraint.clone_with_transformed_parameters(
            transformed_parameters={p.name: p for p in self.constraint.parameters}
        )
        self.assertEqual(
            self.constraint.lower_parameter, constraint_clone.lower_parameter
        )

        constraint_clone._lower_parameter = self.y
        self.assertNotEqual(
            self.constraint.lower_parameter, constraint_clone.lower_parameter
        )

    def test_InvalidSetup(self) -> None:
        z1 = FixedParameter("z1", ParameterType.INT, 0)
        z2 = FixedParameter("z2", ParameterType.INT, 1)
        # Order constraints with one fixed parameter are supported
        OrderConstraint(lower_parameter=self.x, upper_parameter=z1)
        # But not if all parameters are fixed
        with self.assertRaisesRegex(
            ValueError,
            "not supported if all involved parameters are of type ``FixedParameter``",
        ):
            OrderConstraint(lower_parameter=z1, upper_parameter=z2)

        z = ChoiceParameter("z", ParameterType.STRING, ["a", "b", "c"])
        with self.assertRaisesRegex(
            ValueError, "only supported for numeric parameters"
        ):
            OrderConstraint(lower_parameter=self.x, upper_parameter=z)


class SumConstraintTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.x = RangeParameter("x", ParameterType.INT, lower=-5, upper=5)
        self.y = RangeParameter("y", ParameterType.INT, lower=-5, upper=5)
        self.constraint1 = SumConstraint(
            parameters=[self.x, self.y], is_upper_bound=True, bound=5
        )
        self.constraint2 = SumConstraint(
            parameters=[self.x, self.y], is_upper_bound=False, bound=-5
        )

        self.constraint_repr1 = "SumConstraint(x + y <= 5.0)"
        self.constraint_repr2 = "SumConstraint(x + y >= -5.0)"

    def test_BadConstruct(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate parameter in constraint"):
            SumConstraint(parameters=[self.x, self.x], is_upper_bound=False, bound=-5.0)
        z = ChoiceParameter("z", ParameterType.STRING, ["a", "b", "c"])
        with self.assertRaisesRegex(
            ValueError, "only supported for numeric parameters"
        ):
            SumConstraint(parameters=[self.x, z], is_upper_bound=False, bound=-5.0)

    def test_Properties(self) -> None:
        self.assertEqual(self.constraint1.op, ComparisonOp.LEQ)
        self.assertTrue(self.constraint1._is_upper_bound)

        self.assertEqual(self.constraint2.op, ComparisonOp.GEQ)
        self.assertFalse(self.constraint2._is_upper_bound)

    def test_Repr(self) -> None:
        self.assertEqual(str(self.constraint1), self.constraint_repr1)
        self.assertEqual(str(self.constraint2), self.constraint_repr2)

    def test_Validate(self) -> None:
        self.assertTrue(self.constraint1.check({"x": 1, "y": 4}))
        self.assertTrue(self.constraint1.check({"x": 4, "y": 1}))
        self.assertFalse(self.constraint1.check({"x": 1, "y": 5}))

        self.assertTrue(self.constraint2.check({"x": -4, "y": -1}))
        self.assertTrue(self.constraint2.check({"x": -1, "y": -4}))
        self.assertFalse(self.constraint2.check({"x": -5, "y": -1}))

    def test_Clone(self) -> None:
        constraint_clone = self.constraint1.clone()
        self.assertEqual(self.constraint1.bound, constraint_clone.bound)

        constraint_clone._bound = 7.0
        self.assertNotEqual(self.constraint1.bound, constraint_clone.bound)

        constraint_clone_2 = self.constraint2.clone()
        self.assertEqual(self.constraint2.bound, constraint_clone_2.bound)

    def test_CloneWithTransformedParameters(self) -> None:
        constraint_clone = self.constraint1.clone_with_transformed_parameters(
            transformed_parameters={p.name: p for p in self.constraint1.parameters}
        )
        self.assertEqual(self.constraint1.bound, constraint_clone.bound)

        constraint_clone._bound = 7.0
        self.assertNotEqual(self.constraint1.bound, constraint_clone.bound)

    def test_InvalidSetup(self) -> None:
        z1 = FixedParameter("z1", ParameterType.INT, 0)
        z2 = FixedParameter("z2", ParameterType.INT, 1)
        # Sum constraints with one fixed parameter are supported
        SumConstraint(parameters=[self.x, z1], is_upper_bound=False, bound=4.0)
        # But not if all parameters are fixed
        with self.assertRaisesRegex(
            ValueError,
            "not supported if all involved parameters are of type ``FixedParameter``",
        ):
            SumConstraint(parameters=[z1, z2], is_upper_bound=False, bound=4.0)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    def setUp(self):
        self.constraint = ParameterConstraint(
            constraint_dict={"x": 2.0, "y": -3.0}, bound=6.0
        )
        self.constraint_repr = "ParameterConstraint(2.0*x + -3.0*y <= 6.0)"

    def testEq(self):
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

    def testProperties(self):
        self.assertEqual(self.constraint.constraint_dict["x"], 2.0)
        self.assertEqual(self.constraint.bound, 6.0)

    def testRepr(self):
        self.assertEqual(str(self.constraint), self.constraint_repr)

    def testValidate(self):
        parameters = {"x": 4, "z": 3}
        with self.assertRaises(ValueError):
            self.constraint.check(parameters)

        parameters = {"x": 4, "y": 1}
        self.assertTrue(self.constraint.check(parameters))

        self.constraint.bound = 4.0
        self.assertFalse(self.constraint.check(parameters))

    def testClone(self):
        constraint_clone = self.constraint.clone()
        self.assertEqual(self.constraint.bound, constraint_clone.bound)

        constraint_clone._bound = 7.0
        self.assertNotEqual(self.constraint.bound, constraint_clone.bound)

    def testCloneWithTransformedParameters(self):
        constraint_clone = self.constraint.clone_with_transformed_parameters(
            transformed_parameters={}
        )
        self.assertEqual(self.constraint.bound, constraint_clone.bound)

        constraint_clone._bound = 7.0
        self.assertNotEqual(self.constraint.bound, constraint_clone.bound)


class OrderConstraintTest(TestCase):
    def setUp(self):
        self.x = RangeParameter("x", ParameterType.INT, lower=0, upper=1)
        self.y = RangeParameter("y", ParameterType.INT, lower=0, upper=1)
        self.constraint = OrderConstraint(
            lower_parameter=self.x, upper_parameter=self.y
        )
        self.constraint_repr = "OrderConstraint(x <= y)"

    def testProperties(self):
        self.assertEqual(self.constraint.lower_parameter.name, "x")
        self.assertEqual(self.constraint.upper_parameter.name, "y")

    def testRepr(self):
        self.assertEqual(str(self.constraint), self.constraint_repr)

    def testValidate(self):
        self.assertTrue(self.constraint.check({"x": 0, "y": 1}))
        self.assertTrue(self.constraint.check({"x": 1, "y": 1}))
        self.assertFalse(self.constraint.check({"x": 1, "y": 0}))

    def testClone(self):
        constraint_clone = self.constraint.clone()
        self.assertEqual(
            self.constraint.lower_parameter, constraint_clone.lower_parameter
        )

        constraint_clone._lower_parameter = self.y
        self.assertNotEqual(
            self.constraint.lower_parameter, constraint_clone.lower_parameter
        )

    def testCloneWithTransformedParameters(self):
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

    def testInvalidSetup(self):
        z = FixedParameter("z", ParameterType.INT, 0)
        with self.assertRaises(ValueError):
            self.constraint = OrderConstraint(lower_parameter=self.x, upper_parameter=z)

        z = ChoiceParameter("z", ParameterType.STRING, ["a", "b", "c"])
        with self.assertRaises(ValueError):
            self.constraint = OrderConstraint(lower_parameter=self.x, upper_parameter=z)


class SumConstraintTest(TestCase):
    def setUp(self):
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

    def testBadConstruct(self):
        with self.assertRaises(ValueError):
            SumConstraint(parameters=[self.x, self.x], is_upper_bound=False, bound=-5.0)
        z = ChoiceParameter("z", ParameterType.STRING, ["a", "b", "c"])
        with self.assertRaises(ValueError):
            self.constraint = SumConstraint(
                parameters=[self.x, z], is_upper_bound=False, bound=-5.0
            )

    def testProperties(self):
        self.assertEqual(self.constraint1.op, ComparisonOp.LEQ)
        self.assertEqual(self.constraint2.op, ComparisonOp.GEQ)

    def testRepr(self):
        self.assertEqual(str(self.constraint1), self.constraint_repr1)
        self.assertEqual(str(self.constraint2), self.constraint_repr2)

    def testValidate(self):
        self.assertTrue(self.constraint1.check({"x": 1, "y": 4}))
        self.assertTrue(self.constraint1.check({"x": 4, "y": 1}))
        self.assertFalse(self.constraint1.check({"x": 1, "y": 5}))

        self.assertTrue(self.constraint2.check({"x": -4, "y": -1}))
        self.assertTrue(self.constraint2.check({"x": -1, "y": -4}))
        self.assertFalse(self.constraint2.check({"x": -5, "y": -1}))

    def testClone(self):
        constraint_clone = self.constraint1.clone()
        self.assertEqual(self.constraint1.bound, constraint_clone.bound)

        constraint_clone._bound = 7.0
        self.assertNotEqual(self.constraint1.bound, constraint_clone.bound)

        constraint_clone_2 = self.constraint2.clone()
        self.assertEqual(self.constraint2.bound, constraint_clone_2.bound)

    def testCloneWithTransformedParameters(self):
        constraint_clone = self.constraint1.clone_with_transformed_parameters(
            transformed_parameters={p.name: p for p in self.constraint1.parameters}
        )
        self.assertEqual(self.constraint1.bound, constraint_clone.bound)

        constraint_clone._bound = 7.0
        self.assertNotEqual(self.constraint1.bound, constraint_clone.bound)

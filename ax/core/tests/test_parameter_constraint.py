#!/usr/bin/env python3

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


class OrderConstraintTest(TestCase):
    def setUp(self):
        self.constraint = OrderConstraint(lower_name="x", upper_name="y")
        self.constraint_repr = "OrderConstraint(x <= y)"

    def testProperties(self):
        self.assertEqual(self.constraint.lower_name, "x")
        self.assertEqual(self.constraint.upper_name, "y")

    def testRepr(self):
        self.assertEqual(str(self.constraint), self.constraint_repr)

    def testValidate(self):
        self.assertTrue(self.constraint.check({"x": 4, "y": 5}))
        self.assertTrue(self.constraint.check({"x": 5, "y": 5}))
        self.assertFalse(self.constraint.check({"x": 6, "y": 5}))

    def testClone(self):
        constraint_clone = self.constraint.clone()
        self.assertEqual(self.constraint.lower_name, constraint_clone.lower_name)

        constraint_clone._lower_name = "z"
        self.assertNotEqual(self.constraint.lower_name, constraint_clone.lower_name)


class SumConstraintTest(TestCase):
    def setUp(self):
        self.constraint1 = SumConstraint(
            parameter_names=["x", "y"], is_upper_bound=True, bound=6.0
        )
        self.constraint2 = SumConstraint(
            parameter_names=["x", "y"], is_upper_bound=False, bound=-3.0
        )

        self.constraint_repr1 = "SumConstraint(x + y <= 6.0)"
        self.constraint_repr2 = "SumConstraint(x + y >= -3.0)"

    def testBadConstruct(self):
        with self.assertRaises(ValueError):
            SumConstraint(
                parameter_names=["x", "x", "y"], is_upper_bound=False, bound=-3.0
            )

    def testProperties(self):
        self.assertEqual(self.constraint1.op, ComparisonOp.LEQ)
        self.assertEqual(self.constraint2.op, ComparisonOp.GEQ)

    def testRepr(self):
        self.assertEqual(str(self.constraint1), self.constraint_repr1)
        self.assertEqual(str(self.constraint2), self.constraint_repr2)

    def testValidate(self):
        self.assertTrue(self.constraint1.check({"x": 2, "y": 3}))
        self.assertTrue(self.constraint1.check({"x": 2, "y": 4}))
        self.assertFalse(self.constraint1.check({"x": 2, "y": 5}))

        self.assertTrue(self.constraint2.check({"x": -3, "y": 1}))
        self.assertTrue(self.constraint2.check({"x": -4, "y": 1}))
        self.assertFalse(self.constraint2.check({"x": -5, "y": 1}))

    def testClone(self):
        constraint_clone = self.constraint1.clone()
        self.assertEqual(self.constraint1.bound, constraint_clone.bound)

        constraint_clone._bound = 7.0
        self.assertNotEqual(self.constraint1.bound, constraint_clone.bound)

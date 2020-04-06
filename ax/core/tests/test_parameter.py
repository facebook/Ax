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
from ax.utils.common.testutils import TestCase


class RangeParameterTest(TestCase):
    def setUp(self):
        self.param1 = RangeParameter(
            name="x",
            parameter_type=ParameterType.FLOAT,
            lower=1,
            upper=3,
            log_scale=True,
            digits=5,
            is_fidelity=True,
            target_value=2,
        )
        self.param1_repr = (
            "RangeParameter(name='x', parameter_type=FLOAT, "
            "range=[1.0, 3.0], log_scale=True, digits=5, fidelity=True, target_"
            "value=2.0)"
        )

        self.param2 = RangeParameter(
            name="y", parameter_type=ParameterType.INT, lower=10, upper=15
        )
        self.param2_repr = (
            "RangeParameter(name='y', parameter_type=INT, range=[10, 15])"
        )

    def testEq(self):
        param2 = RangeParameter(
            name="x",
            parameter_type=ParameterType.FLOAT,
            lower=1,
            upper=3,
            log_scale=True,
            digits=5,
            is_fidelity=True,
            target_value=2,
        )
        self.assertEqual(self.param1, param2)
        self.assertNotEqual(self.param1, self.param2)

    def testProperties(self):
        self.assertEqual(self.param1.name, "x")
        self.assertEqual(self.param1.parameter_type, ParameterType.FLOAT)
        self.assertEqual(self.param1.lower, 1)
        self.assertEqual(self.param1.upper, 3)
        self.assertEqual(self.param1.digits, 5)
        self.assertTrue(self.param1.log_scale)
        self.assertFalse(self.param2.log_scale)
        self.assertTrue(self.param1.is_numeric)
        self.assertTrue(self.param1.is_fidelity)
        self.assertIsNotNone(self.param1.target_value)
        self.assertFalse(self.param2.is_fidelity)
        self.assertIsNone(self.param2.target_value)

    def testValidate(self):
        self.assertFalse(self.param1.validate(None))
        self.assertFalse(self.param1.validate("foo"))
        self.assertTrue(self.param1.validate(1))
        self.assertTrue(self.param1.validate(1.3))

    def testRepr(self):
        self.assertEqual(str(self.param1), self.param1_repr)
        self.assertEqual(str(self.param2), self.param2_repr)

    def testBadCreations(self):
        with self.assertRaises(ValueError):
            RangeParameter("x", ParameterType.STRING, 1, 3)

        with self.assertRaises(ValueError):
            RangeParameter("x", ParameterType.FLOAT, 3, 1)

        with self.assertRaises(ValueError):
            RangeParameter("x", ParameterType.INT, 0, 1, log_scale=True)

        with self.assertRaises(ValueError):
            RangeParameter("x", ParameterType.INT, 0.5, 1)

        with self.assertRaises(ValueError):
            RangeParameter("x", ParameterType.INT, 0.5, 1, is_fidelity=True)

    def testBadSetter(self):
        with self.assertRaises(ValueError):
            self.param1.update_range(upper="foo")

        with self.assertRaises(ValueError):
            self.param1.update_range(lower="foo")

        with self.assertRaises(ValueError):
            self.param1.update_range(lower=4)

        with self.assertRaises(ValueError):
            self.param1.update_range(upper=0.5)

        with self.assertRaises(ValueError):
            self.param1.update_range(lower=1.0, upper=0.9)

    def testGoodSetter(self):
        self.param1.update_range(lower=1.0)
        self.param1.update_range(upper=1.0011)
        self.param1.set_log_scale(False)
        self.param1.set_digits(3)
        self.assertEqual(self.param1.digits, 3)
        self.assertEqual(self.param1.upper, 1.001)

        # This would cast Upper = Lower = 1, which is not allowed
        with self.assertRaises(ValueError):
            self.param1.set_digits(1)

        self.param1.update_range(lower=2.0, upper=3.0)
        self.assertEqual(self.param1.lower, 2.0)
        self.assertEqual(self.param1.upper, 3.0)

    def testCast(self):
        self.assertEqual(self.param2.cast(2.5), 2)
        self.assertEqual(self.param2.cast(3), 3)
        self.assertEqual(self.param2.cast(None), None)

    def testClone(self):
        param_clone = self.param1.clone()
        self.assertEqual(self.param1.lower, param_clone.lower)

        param_clone._lower = 2.0
        self.assertNotEqual(self.param1.lower, param_clone.lower)


class ChoiceParameterTest(TestCase):
    def setUp(self):
        self.param1 = ChoiceParameter(
            name="x", parameter_type=ParameterType.STRING, values=["foo", "bar", "baz"]
        )
        self.param1_repr = (
            "ChoiceParameter(name='x', parameter_type=STRING, "
            "values=['foo', 'bar', 'baz'])"
        )
        self.param2 = ChoiceParameter(
            name="x",
            parameter_type=ParameterType.STRING,
            values=["foo", "bar", "baz"],
            is_ordered=True,
            is_task=True,
        )
        self.param3 = ChoiceParameter(
            name="x",
            parameter_type=ParameterType.STRING,
            values=["foo", "bar"],
            is_fidelity=True,
            target_value="bar",
        )
        self.param3_repr = (
            "ChoiceParameter(name='x', parameter_type=STRING, "
            "values=['foo', 'bar'], fidelity=True, target_value='bar')"
        )

    def testBadCreations(self):
        with self.assertRaises(ValueError):
            ChoiceParameter(
                name="x",
                parameter_type=ParameterType.STRING,
                values=["foo", "foo2"],
                is_fidelity=True,
            )

    def testEq(self):
        param4 = ChoiceParameter(
            name="x", parameter_type=ParameterType.STRING, values=["foo", "bar", "baz"]
        )
        self.assertEqual(self.param1, param4)
        self.assertNotEqual(self.param1, self.param2)

        param5 = ChoiceParameter(
            name="x", parameter_type=ParameterType.STRING, values=["foo", "foobar"]
        )
        self.assertNotEqual(self.param1, param5)

    def testProperties(self):
        self.assertEqual(self.param1.name, "x")
        self.assertEqual(self.param1.parameter_type, ParameterType.STRING)
        self.assertEqual(len(self.param1.values), 3)
        self.assertFalse(self.param1.is_numeric)
        self.assertFalse(self.param1.is_ordered)
        self.assertFalse(self.param1.is_task)
        self.assertTrue(self.param2.is_ordered)
        self.assertTrue(self.param2.is_task)

    def testRepr(self):
        self.assertEqual(str(self.param1), self.param1_repr)
        self.assertEqual(str(self.param3), self.param3_repr)

    def testValidate(self):
        self.assertFalse(self.param1.validate(None))
        self.assertFalse(self.param1.validate(3))
        for value in ["foo", "bar", "baz"]:
            self.assertTrue(self.param1.validate(value))

    def testSetter(self):
        self.param1.add_values(["bin"])
        self.assertTrue(self.param1.validate("bin"))

        self.param1.set_values(["bar", "biz"])
        self.assertTrue(self.param1.validate("biz"))
        self.assertTrue(self.param1.validate("bar"))
        self.assertFalse(self.param1.validate("foo"))

    def testSingleValue(self):
        with self.assertRaises(ValueError):
            ChoiceParameter(
                name="x", parameter_type=ParameterType.STRING, values=["foo"]
            )
        with self.assertRaises(ValueError):
            self.param1.set_values(["foo"])

    def testClone(self):
        param_clone = self.param1.clone()
        self.assertEqual(len(self.param1.values), len(param_clone.values))

        param_clone._values.append("boo")
        self.assertNotEqual(len(self.param1.values), len(param_clone.values))


class FixedParameterTest(TestCase):
    def setUp(self):
        self.param1 = FixedParameter(
            name="x", parameter_type=ParameterType.BOOL, value=True
        )
        self.param1_repr = "FixedParameter(name='x', parameter_type=BOOL, value=True)"

    def testBadCreations(self):
        with self.assertRaises(ValueError):
            FixedParameter(
                name="x",
                parameter_type=ParameterType.BOOL,
                value=True,
                is_fidelity=True,
            )

    def testEq(self):
        param2 = FixedParameter(name="x", parameter_type=ParameterType.BOOL, value=True)
        self.assertEqual(self.param1, param2)

        param3 = FixedParameter(
            name="x", parameter_type=ParameterType.BOOL, value=False
        )
        self.assertNotEqual(self.param1, param3)

    def testProperties(self):
        self.assertEqual(self.param1.name, "x")
        self.assertEqual(self.param1.parameter_type, ParameterType.BOOL)
        self.assertEqual(self.param1.value, True)
        self.assertFalse(self.param1.is_numeric)

    def testRepr(self):
        self.assertEqual(str(self.param1), self.param1_repr)
        self.param1._is_fidelity = True
        self.assertNotEqual(str(self.param1), self.param1_repr)

    def testValidate(self):
        self.assertFalse(self.param1.validate(None))
        self.assertFalse(self.param1.validate("foo"))
        self.assertFalse(self.param1.validate(False))
        self.assertTrue(self.param1.validate(True))

    def testSetter(self):
        self.param1.set_value(False)
        self.assertEqual(self.param1.value, False)

    def testClone(self):
        param_clone = self.param1.clone()
        self.assertEqual(self.param1.value, param_clone.value)

        param_clone._value = False
        self.assertNotEqual(self.param1.value, param_clone.value)

    def testCast(self):
        self.assertEqual(self.param1.cast(1), True)
        self.assertEqual(self.param1.cast(False), False)
        self.assertEqual(self.param1.cast(None), None)


class ParameterEqualityTest(TestCase):
    def setUp(self):
        self.fixed_parameter = FixedParameter(
            name="x", parameter_type=ParameterType.BOOL, value=True
        )
        self.choice_parameter = ChoiceParameter(
            name="x", parameter_type=ParameterType.STRING, values=["foo", "bar", "baz"]
        )

    def testNotEqual(self):
        self.assertNotEqual(self.fixed_parameter, self.choice_parameter)

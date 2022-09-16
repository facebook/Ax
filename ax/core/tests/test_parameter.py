#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.parameter import (
    _get_parameter_type,
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase


class RangeParameterTest(TestCase):
    def setUp(self) -> None:
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

    def testEq(self) -> None:
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

    def testProperties(self) -> None:
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

    def testValidate(self) -> None:
        self.assertFalse(self.param1.validate(None))
        self.assertFalse(self.param1.validate("foo"))
        self.assertTrue(self.param1.validate(1))
        self.assertTrue(self.param1.validate(1.3))

    def testRepr(self) -> None:
        self.assertEqual(str(self.param1), self.param1_repr)
        self.assertEqual(str(self.param2), self.param2_repr)

    def testBadCreations(self) -> None:
        with self.assertRaises(UserInputError):
            RangeParameter("x", ParameterType.STRING, 1, 3)

        with self.assertRaises(UserInputError):
            RangeParameter("x", ParameterType.FLOAT, 3, 1)

        with self.assertRaises(UserInputError):
            RangeParameter("x", ParameterType.INT, 0, 1, log_scale=True)

        with self.assertRaises(UserInputError):
            RangeParameter("x", ParameterType.INT, 0.5, 1)

        with self.assertRaises(UserInputError):
            RangeParameter("x", ParameterType.INT, 0.5, 1, is_fidelity=True)

    def testBadSetter(self) -> None:
        with self.assertRaises(ValueError):
            # pyre-fixme[6]: For 1st param expected `Optional[float]` but got `str`.
            self.param1.update_range(upper="foo")

        with self.assertRaises(ValueError):
            # pyre-fixme[6]: For 1st param expected `Optional[float]` but got `str`.
            self.param1.update_range(lower="foo")

        with self.assertRaises(UserInputError):
            self.param1.update_range(lower=4)

        with self.assertRaises(UserInputError):
            self.param1.update_range(upper=0.5)

        with self.assertRaises(UserInputError):
            self.param1.update_range(lower=1.0, upper=0.9)

    def testGoodSetter(self) -> None:
        self.param1.update_range(lower=1.0)
        self.param1.update_range(upper=1.0011)
        self.param1.set_log_scale(False)
        self.param1.set_digits(3)
        self.assertEqual(self.param1.digits, 3)
        self.assertEqual(self.param1.upper, 1.001)

        # This would cast Upper = Lower = 1, which is not allowed
        with self.assertRaises(UserInputError):
            self.param1.set_digits(1)

        self.param1.update_range(lower=2.0, upper=3.0)
        self.assertEqual(self.param1.lower, 2.0)
        self.assertEqual(self.param1.upper, 3.0)

    def testCast(self) -> None:
        self.assertEqual(self.param2.cast(2.5), 2)
        self.assertEqual(self.param2.cast(3), 3)
        self.assertEqual(self.param2.cast(None), None)

    def testClone(self) -> None:
        param_clone = self.param1.clone()
        self.assertEqual(self.param1.lower, param_clone.lower)

        param_clone._lower = 2.0
        self.assertNotEqual(self.param1.lower, param_clone.lower)

    def test_get_parameter_type(self) -> None:
        self.assertEqual(_get_parameter_type(float), ParameterType.FLOAT)
        self.assertEqual(_get_parameter_type(int), ParameterType.INT)
        self.assertEqual(_get_parameter_type(bool), ParameterType.BOOL)
        self.assertEqual(_get_parameter_type(str), ParameterType.STRING)
        with self.assertRaises(ValueError):
            _get_parameter_type(dict)

    def testSortable(self) -> None:
        param2 = RangeParameter(
            name="z",
            parameter_type=ParameterType.FLOAT,
            lower=0,
            upper=1,
        )
        self.assertTrue(self.param1 < param2)

    def testHierarchicalValidation(self) -> None:
        self.assertFalse(self.param1.is_hierarchical)
        with self.assertRaises(NotImplementedError):
            self.param1.dependents


class ChoiceParameterTest(TestCase):
    def setUp(self) -> None:
        self.param1 = ChoiceParameter(
            name="x", parameter_type=ParameterType.STRING, values=["foo", "bar", "baz"]
        )
        self.param1_repr = (
            "ChoiceParameter(name='x', parameter_type=STRING, "
            "values=['foo', 'bar', 'baz'], is_ordered=False, sort_values=False)"
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
            "values=['foo', 'bar'], is_ordered=False, sort_values=False, "
            "is_fidelity=True, target_value='bar')"
        )
        self.param4 = ChoiceParameter(
            name="x",
            parameter_type=ParameterType.INT,
            values=[1, 2],
        )
        self.param4_repr = (
            "ChoiceParameter(name='x', parameter_type=INT, "
            "values=[1, 2], is_ordered=True, sort_values=True)"
        )

    def testBadCreations(self) -> None:
        with self.assertRaises(UserInputError):
            ChoiceParameter(
                name="x",
                parameter_type=ParameterType.STRING,
                values=["foo", "foo2"],
                is_fidelity=True,
            )

    def testEq(self) -> None:
        param4 = ChoiceParameter(
            name="x", parameter_type=ParameterType.STRING, values=["foo", "bar", "baz"]
        )
        self.assertEqual(self.param1, param4)
        self.assertNotEqual(self.param1, self.param2)

        param5 = ChoiceParameter(
            name="x", parameter_type=ParameterType.STRING, values=["foo", "foobar"]
        )
        self.assertNotEqual(self.param1, param5)

    def testProperties(self) -> None:
        self.assertEqual(self.param1.name, "x")
        self.assertEqual(self.param1.parameter_type, ParameterType.STRING)
        self.assertEqual(len(self.param1.values), 3)
        self.assertFalse(self.param1.is_numeric)
        self.assertFalse(self.param1.is_ordered)
        self.assertFalse(self.param1.is_task)
        self.assertTrue(self.param2.is_ordered)
        self.assertTrue(self.param2.is_task)
        # check is_ordered defaults
        bool_param = ChoiceParameter(
            name="x", parameter_type=ParameterType.BOOL, values=[True, False]
        )
        self.assertTrue(bool_param.is_ordered)
        int_param = ChoiceParameter(
            name="x", parameter_type=ParameterType.INT, values=[2, 1, 3]
        )
        self.assertTrue(int_param.is_ordered)
        # pyre-fixme[6]: For 1st param expected
        #  `Iterable[Variable[SupportsRichComparisonT (bound to
        #  Union[SupportsDunderGT[typing.Any], SupportsDunderLT[typing.Any]])]]` but
        #  got `List[Union[None, bool, float, int, str]]`.
        self.assertListEqual(int_param.values, sorted(int_param.values))
        float_param = ChoiceParameter(
            name="x", parameter_type=ParameterType.FLOAT, values=[1.5, 2.5, 3.5]
        )
        self.assertTrue(float_param.is_ordered)
        string_param = ChoiceParameter(
            name="x", parameter_type=ParameterType.STRING, values=["foo", "bar", "baz"]
        )
        self.assertFalse(string_param.is_ordered)

    def testRepr(self) -> None:
        self.assertEqual(str(self.param1), self.param1_repr)
        self.assertEqual(str(self.param3), self.param3_repr)
        self.assertEqual(str(self.param4), self.param4_repr)

    def testValidate(self) -> None:
        self.assertFalse(self.param1.validate(None))
        self.assertFalse(self.param1.validate(3))
        for value in ["foo", "bar", "baz"]:
            self.assertTrue(self.param1.validate(value))

    def testSetter(self) -> None:
        self.param1.add_values(["bin"])
        self.assertTrue(self.param1.validate("bin"))

        self.param1.set_values(["bar", "biz"])
        self.assertTrue(self.param1.validate("biz"))
        self.assertTrue(self.param1.validate("bar"))
        self.assertFalse(self.param1.validate("foo"))

    def testSingleValue(self) -> None:
        with self.assertRaises(UserInputError):
            ChoiceParameter(
                name="x", parameter_type=ParameterType.STRING, values=["foo"]
            )
        with self.assertRaises(UserInputError):
            self.param1.set_values(["foo"])

    def testClone(self) -> None:
        param_clone = self.param1.clone()
        self.assertEqual(len(self.param1.values), len(param_clone.values))
        self.assertEqual(self.param1._is_ordered, param_clone._is_ordered)

        param_clone._values.append("boo")
        self.assertNotEqual(len(self.param1.values), len(param_clone.values))

    def testHierarchicalValidation(self) -> None:
        self.assertFalse(self.param1.is_hierarchical)
        with self.assertRaises(NotImplementedError):
            self.param1.dependents
        with self.assertRaises(UserInputError):
            ChoiceParameter(
                name="x",
                parameter_type=ParameterType.BOOL,
                values=[True, False],
                # pyre-fixme[6]: For 4th param expected `Optional[Dict[Union[None,
                #  bool, float, int, str], List[str]]]` but got `Dict[str, str]`.
                dependents={"not_a_value": "other_param"},
            )

    def testHierarchical(self) -> None:
        # Test case where only some of the values entail dependents.
        hierarchical_param = ChoiceParameter(
            name="x",
            parameter_type=ParameterType.BOOL,
            values=[True, False],
            # pyre-fixme[6]: For 4th param expected `Optional[Dict[Union[None, bool,
            #  float, int, str], List[str]]]` but got `Dict[bool, str]`.
            dependents={True: "other_param"},
        )
        self.assertTrue(hierarchical_param.is_hierarchical)
        self.assertEqual(hierarchical_param.dependents, {True: "other_param"})

        # Test case where all of the values entail dependents.
        hierarchical_param_2 = ChoiceParameter(
            name="x",
            parameter_type=ParameterType.STRING,
            values=["a", "b"],
            # pyre-fixme[6]: For 4th param expected `Optional[Dict[Union[None, bool,
            #  float, int, str], List[str]]]` but got `Dict[str, str]`.
            dependents={"a": "other_param", "b": "third_param"},
        )
        self.assertTrue(hierarchical_param_2.is_hierarchical)
        self.assertEqual(
            hierarchical_param_2.dependents, {"a": "other_param", "b": "third_param"}
        )

        # Test case where nonexisted value entails dependents.
        with self.assertRaises(UserInputError):
            ChoiceParameter(
                name="x",
                parameter_type=ParameterType.STRING,
                values=["a", "b"],
                # pyre-fixme[6]: For 4th param expected `Optional[Dict[Union[None,
                #  bool, float, int, str], List[str]]]` but got `Dict[str, str]`.
                dependents={"c": "other_param"},
            )


class FixedParameterTest(TestCase):
    def setUp(self) -> None:
        self.param1 = FixedParameter(
            name="x", parameter_type=ParameterType.BOOL, value=True
        )
        self.param1_repr = "FixedParameter(name='x', parameter_type=BOOL, value=True)"

    def testBadCreations(self) -> None:
        with self.assertRaises(UserInputError):
            FixedParameter(
                name="x",
                parameter_type=ParameterType.BOOL,
                value=True,
                is_fidelity=True,
            )

    def testEq(self) -> None:
        param2 = FixedParameter(name="x", parameter_type=ParameterType.BOOL, value=True)
        self.assertEqual(self.param1, param2)

        param3 = FixedParameter(
            name="x", parameter_type=ParameterType.BOOL, value=False
        )
        self.assertNotEqual(self.param1, param3)

    def testProperties(self) -> None:
        self.assertEqual(self.param1.name, "x")
        self.assertEqual(self.param1.parameter_type, ParameterType.BOOL)
        self.assertEqual(self.param1.value, True)
        self.assertFalse(self.param1.is_numeric)

    def testRepr(self) -> None:
        self.assertEqual(str(self.param1), self.param1_repr)
        self.param1._is_fidelity = True
        self.assertNotEqual(str(self.param1), self.param1_repr)

    def testValidate(self) -> None:
        self.assertFalse(self.param1.validate(None))
        self.assertFalse(self.param1.validate("foo"))
        self.assertFalse(self.param1.validate(False))
        self.assertTrue(self.param1.validate(True))

    def testSetter(self) -> None:
        self.param1.set_value(False)
        self.assertEqual(self.param1.value, False)

    def testClone(self) -> None:
        param_clone = self.param1.clone()
        self.assertEqual(self.param1.value, param_clone.value)

        param_clone._value = False
        self.assertNotEqual(self.param1.value, param_clone.value)

    def testCast(self) -> None:
        self.assertEqual(self.param1.cast(1), True)
        self.assertEqual(self.param1.cast(False), False)
        self.assertEqual(self.param1.cast(None), None)

    def testHierarchicalValidation(self) -> None:
        self.assertFalse(self.param1.is_hierarchical)
        with self.assertRaises(NotImplementedError):
            self.param1.dependents

    def testHierarchical(self) -> None:
        # Test case where only some of the values entail dependents.
        hierarchical_param = FixedParameter(
            name="x",
            parameter_type=ParameterType.BOOL,
            value=True,
            # pyre-fixme[6]: For 4th param expected `Optional[Dict[Union[None, bool,
            #  float, int, str], List[str]]]` but got `Dict[bool, str]`.
            dependents={True: "other_param"},
        )
        self.assertTrue(hierarchical_param.is_hierarchical)
        self.assertEqual(hierarchical_param.dependents, {True: "other_param"})

        # Test case where nonexistent value entails dependents.
        with self.assertRaises(UserInputError):
            FixedParameter(
                name="x",
                parameter_type=ParameterType.BOOL,
                value=True,
                # pyre-fixme[6]: For 4th param expected `Optional[Dict[Union[None,
                #  bool, float, int, str], List[str]]]` but got `Dict[bool, str]`.
                dependents={False: "other_param"},
            )


class ParameterEqualityTest(TestCase):
    def setUp(self) -> None:
        self.fixed_parameter = FixedParameter(
            name="x", parameter_type=ParameterType.BOOL, value=True
        )
        self.choice_parameter = ChoiceParameter(
            name="x", parameter_type=ParameterType.STRING, values=["foo", "bar", "baz"]
        )

    def testNotEqual(self) -> None:
        self.assertNotEqual(self.fixed_parameter, self.choice_parameter)

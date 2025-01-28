#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import cast

from ax.core.parameter import (
    _get_parameter_type,
    ChoiceParameter,
    EPS,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.exceptions.core import AxParameterWarning, AxWarning, UserInputError
from ax.utils.common.testutils import TestCase
from pyre_extensions import none_throws


class RangeParameterTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
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
            "RangeParameter(name='x', parameter_type=FLOAT, range=[1.0, 3.0], "
            "is_fidelity=True, log_scale=True, target_value=2.0, digits=5)"
        )

        self.param2 = RangeParameter(
            name="y", parameter_type=ParameterType.INT, lower=10, upper=15
        )
        self.param2_repr = (
            "RangeParameter(name='y', parameter_type=INT, range=[10, 15])"
        )

    def test_Eq(self) -> None:
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

    def test_Properties(self) -> None:
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

    def test_Validate(self) -> None:
        self.assertFalse(self.param1.validate(None))
        self.assertFalse(self.param1.validate("foo"))
        self.assertTrue(self.param1.validate(1))
        self.assertTrue(self.param1.validate(1.3))
        self.assertFalse(self.param1.validate(3.5))
        # Check with tolerances
        self.assertTrue(self.param1.validate(1 - 0.5 * EPS))
        self.assertTrue(self.param1.validate(3 + 0.5 * EPS))

    def test_Repr(self) -> None:
        self.assertEqual(str(self.param1), self.param1_repr)
        self.assertEqual(str(self.param2), self.param2_repr)

    def test_BadCreations(self) -> None:
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

        with self.assertRaisesRegex(
            UserInputError,
            "likely to cause numerical errors. Consider reparameterizing",
        ):
            RangeParameter("x", ParameterType.FLOAT, EPS, 2 * EPS)

    def test_BadSetter(self) -> None:
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

    def test_GoodSetter(self) -> None:
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

    def test_Cast(self) -> None:
        self.assertEqual(self.param2.cast(2.5), 2)
        self.assertEqual(self.param2.cast(3), 3)
        self.assertEqual(self.param2.cast(None), None)

    def test_Clone(self) -> None:
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

    def test_Sortable(self) -> None:
        param2 = RangeParameter(
            name="z",
            parameter_type=ParameterType.FLOAT,
            lower=0,
            upper=1,
        )
        self.assertTrue(self.param1 < param2)

    def test_HierarchicalValidation(self) -> None:
        self.assertFalse(self.param1.is_hierarchical)
        with self.assertRaises(NotImplementedError):
            self.param1.dependents

    def test_available_flags(self) -> None:
        range_flags = ["is_fidelity", "log_scale", "logit_scale"]
        self.assertListEqual(self.param1.available_flags, range_flags)
        self.assertListEqual(self.param2.available_flags, range_flags)

    def test_domain_repr(self) -> None:
        self.assertEqual(self.param1.domain_repr, "range=[1.0, 3.0]")
        self.assertEqual(self.param2.domain_repr, "range=[10, 15]")

    def test_summary_dict(self) -> None:
        self.assertDictEqual(
            self.param1.summary_dict,
            {
                "name": "x",
                "type": "Range",
                "domain": "range=[1.0, 3.0]",
                "parameter_type": "float",
                "flags": "fidelity, log_scale",
                "target_value": 2.0,
            },
        )
        self.assertDictEqual(
            self.param2.summary_dict,
            {
                "name": "y",
                "type": "Range",
                "domain": "range=[10, 15]",
                "parameter_type": "int",
            },
        )


class ChoiceParameterTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
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
            target_value="baz",
        )
        self.param2_repr = (
            "ChoiceParameter(name='x', parameter_type=STRING, "
            "values=['foo', 'bar', 'baz'], is_ordered=False, is_task=True, "
            "sort_values=False, target_value='baz')"
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
            "values=['foo', 'bar'], is_fidelity=True, is_ordered=True, "
            "sort_values=False, target_value='bar')"
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

    def test_BadCreations(self) -> None:
        with self.assertRaises(UserInputError):
            ChoiceParameter(
                name="x",
                parameter_type=ParameterType.STRING,
                values=["foo", "foo2"],
                is_fidelity=True,
            )
        with self.assertRaises(UserInputError):
            ChoiceParameter(
                name="x",
                parameter_type=ParameterType.STRING,
                values=["foo", "foo2"],
                is_task=True,
            )

    def test_Eq(self) -> None:
        param4 = ChoiceParameter(
            name="x", parameter_type=ParameterType.STRING, values=["foo", "bar", "baz"]
        )
        self.assertEqual(self.param1, param4)
        self.assertNotEqual(self.param1, self.param2)

        param5 = ChoiceParameter(
            name="x", parameter_type=ParameterType.STRING, values=["foo", "foobar"]
        )
        self.assertNotEqual(self.param1, param5)

    def test_Properties(self) -> None:
        self.assertEqual(self.param1.name, "x")
        self.assertEqual(self.param1.parameter_type, ParameterType.STRING)
        self.assertEqual(len(self.param1.values), 3)
        self.assertFalse(self.param1.is_numeric)
        self.assertFalse(self.param1.is_ordered)
        self.assertFalse(self.param1.is_task)
        self.assertTrue(self.param2.is_ordered)
        self.assertTrue(self.param2.is_task)
        self.assertEqual(self.param2.target_value, "baz")
        # check is_ordered defaults
        bool_param = ChoiceParameter(
            name="x", parameter_type=ParameterType.BOOL, values=[True, False]
        )
        self.assertTrue(bool_param.is_ordered)
        int_param = ChoiceParameter(
            name="x", parameter_type=ParameterType.INT, values=[2, 1, 3]
        )
        self.assertTrue(int_param.is_ordered)
        self.assertListEqual(
            int_param.values, sorted(cast(list[int], int_param.values))
        )
        float_param = ChoiceParameter(
            name="x", parameter_type=ParameterType.FLOAT, values=[1.5, 2.5, 3.5]
        )
        self.assertTrue(float_param.is_ordered)
        string_param = ChoiceParameter(
            name="x", parameter_type=ParameterType.STRING, values=["foo", "bar", "baz"]
        )
        self.assertFalse(string_param.is_ordered)

    def test_Repr(self) -> None:
        self.assertEqual(str(self.param1), self.param1_repr)
        self.assertEqual(str(self.param3), self.param3_repr)
        self.assertEqual(str(self.param4), self.param4_repr)

    def test_Validate(self) -> None:
        self.assertFalse(self.param1.validate(None))
        self.assertFalse(self.param1.validate(3))
        for value in ["foo", "bar", "baz"]:
            self.assertTrue(self.param1.validate(value))

    def test_Setter(self) -> None:
        self.param1.add_values(["bin"])
        self.assertTrue(self.param1.validate("bin"))

        self.param1.set_values(["bar", "biz"])
        self.assertTrue(self.param1.validate("biz"))
        self.assertTrue(self.param1.validate("bar"))
        self.assertFalse(self.param1.validate("foo"))

    def test_SingleValue(self) -> None:
        with self.assertRaises(UserInputError):
            ChoiceParameter(
                name="x", parameter_type=ParameterType.STRING, values=["foo"]
            )
        with self.assertRaises(UserInputError):
            self.param1.set_values(["foo"])

    def test_Clone(self) -> None:
        param_clone = self.param1.clone()
        self.assertEqual(len(self.param1.values), len(param_clone.values))
        self.assertEqual(self.param1._is_ordered, param_clone._is_ordered)

        param_clone._values.append("boo")
        self.assertNotEqual(len(self.param1.values), len(param_clone.values))

        # With dependents.
        param = ChoiceParameter(
            name="x",
            parameter_type=ParameterType.STRING,
            values=["foo", "bar", "baz"],
            dependents={"foo": ["y", "z"], "bar": ["w"]},
        )
        param_clone = param.clone()
        none_throws(param_clone._dependents)["foo"] = ["y"]
        self.assertEqual(param.dependents, {"foo": ["y", "z"], "bar": ["w"]})
        self.assertEqual(param_clone.dependents, {"foo": ["y"], "bar": ["w"]})

    def test_HierarchicalValidation(self) -> None:
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
        # Check that empty dependents doesn't flag as hierarchical.
        self.param4._dependents = {}
        self.assertFalse(self.param4.is_hierarchical)
        # Check that valid dependents are detected.
        self.param4._dependents = {1: ["other_param"]}
        self.assertTrue(self.param4.is_hierarchical)

    def test_MaxValuesValidation(self) -> None:
        ChoiceParameter(
            name="x",
            parameter_type=ParameterType.INT,
            values=list(range(999)),
        )
        with self.assertRaisesRegex(
            UserInputError,
            "`ChoiceParameter` with more than 1000 values is not supported! Use a "
            "`RangeParameter` instead.",
        ):
            ChoiceParameter(
                name="x",
                parameter_type=ParameterType.INT,
                values=list(range(1001)),
            )

    def test_Hierarchical(self) -> None:
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

    def test_available_flags(self) -> None:
        choice_flags = [
            "is_fidelity",
            "is_ordered",
            "is_hierarchical",
            "is_task",
            "sort_values",
        ]
        self.assertListEqual(self.param1.available_flags, choice_flags)
        self.assertListEqual(self.param2.available_flags, choice_flags)
        self.assertListEqual(self.param3.available_flags, choice_flags)
        self.assertListEqual(self.param4.available_flags, choice_flags)

    def test_domain_repr(self) -> None:
        self.assertEqual(self.param1.domain_repr, "values=['foo', 'bar', 'baz']")
        self.assertEqual(self.param2.domain_repr, "values=['foo', 'bar', 'baz']")
        self.assertEqual(self.param3.domain_repr, "values=['foo', 'bar']")
        self.assertEqual(self.param4.domain_repr, "values=[1, 2]")

    def test_summary_dict(self) -> None:
        self.assertDictEqual(
            self.param1.summary_dict,
            {
                "name": "x",
                "type": "Choice",
                "domain": "values=['foo', 'bar', 'baz']",
                "parameter_type": "string",
                "flags": "unordered, unsorted",
            },
        )
        self.assertDictEqual(
            self.param2.summary_dict,
            {
                "name": "x",
                "type": "Choice",
                "domain": "values=['foo', 'bar', 'baz']",
                "parameter_type": "string",
                "flags": "ordered, task, unsorted",
                "target_value": "baz",
            },
        )
        self.assertDictEqual(
            self.param3.summary_dict,
            {
                "name": "x",
                "type": "Choice",
                "domain": "values=['foo', 'bar']",
                "parameter_type": "string",
                "flags": "fidelity, ordered, unsorted",
                "target_value": "bar",
            },
        )
        self.assertDictEqual(
            self.param4.summary_dict,
            {
                "name": "x",
                "type": "Choice",
                "domain": "values=[1, 2]",
                "parameter_type": "int",
                "flags": "ordered, sorted",
            },
        )

    def test_duplicate_values(self) -> None:
        with self.assertWarnsRegex(AxWarning, "Duplicate values found"):
            p = ChoiceParameter(
                name="x",
                parameter_type=ParameterType.STRING,
                values=["foo", "bar", "foo"],
            )
        self.assertEqual(p.values, ["foo", "bar"])

    def test_two_values_is_ordered(self) -> None:
        parameter_types = (
            ParameterType.INT,
            ParameterType.FLOAT,
            ParameterType.BOOL,
            ParameterType.STRING,
        )
        parameter_values = ([0, 4], [0, 1.234], [False, True], ["foo", "bar"])
        for parameter_type, values in zip(parameter_types, parameter_values):
            p = ChoiceParameter(
                name="x",
                parameter_type=parameter_type,
                values=values,  # pyre-ignore
            )
            self.assertEqual(p._is_ordered, True)

            # Change `is_ordered` to True
            p = ChoiceParameter(
                name="x",
                parameter_type=parameter_type,
                values=values,  # pyre-ignore
                is_ordered=False,
            )
            self.assertEqual(p._is_ordered, True)

            # Set to True if `is_ordered` is not specified
            with self.assertWarnsRegex(
                AxParameterWarning, "since there are exactly two choices"
            ):
                p = ChoiceParameter(
                    name="x",
                    parameter_type=parameter_type,
                    values=values,  # pyre-ignore
                    sort_values=False,
                )
                self.assertEqual(p._is_ordered, True)


class FixedParameterTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.param1 = FixedParameter(
            name="x", parameter_type=ParameterType.BOOL, value=True
        )
        self.param1_repr = "FixedParameter(name='x', parameter_type=BOOL, value=True)"
        self.param2 = FixedParameter(
            name="y", parameter_type=ParameterType.STRING, value="foo"
        )
        self.param2_repr = (
            "FixedParameter(name='y', parameter_type=STRING, value='foo')"
        )

    def test_BadCreations(self) -> None:
        with self.assertRaises(UserInputError):
            FixedParameter(
                name="x",
                parameter_type=ParameterType.BOOL,
                value=True,
                is_fidelity=True,
            )

    def test_Eq(self) -> None:
        param2 = FixedParameter(name="x", parameter_type=ParameterType.BOOL, value=True)
        self.assertEqual(self.param1, param2)

        param3 = FixedParameter(
            name="x", parameter_type=ParameterType.BOOL, value=False
        )
        self.assertNotEqual(self.param1, param3)

    def test_Properties(self) -> None:
        self.assertEqual(self.param1.name, "x")
        self.assertEqual(self.param1.parameter_type, ParameterType.BOOL)
        self.assertEqual(self.param1.value, True)
        self.assertFalse(self.param1.is_numeric)

    def test_Repr(self) -> None:
        self.assertEqual(str(self.param1), self.param1_repr)
        self.param1._is_fidelity = True
        self.assertNotEqual(str(self.param1), self.param1_repr)

    def test_Validate(self) -> None:
        self.assertFalse(self.param1.validate(None))
        self.assertFalse(self.param1.validate("foo"))
        self.assertFalse(self.param1.validate(False))
        self.assertTrue(self.param1.validate(True))

    def test_Setter(self) -> None:
        self.param1.set_value(False)
        self.assertEqual(self.param1.value, False)

    def test_Clone(self) -> None:
        param_clone = self.param1.clone()
        self.assertEqual(self.param1.value, param_clone.value)

        param_clone._value = False
        self.assertNotEqual(self.param1.value, param_clone.value)

    def test_Cast(self) -> None:
        self.assertEqual(self.param1.cast(1), True)
        self.assertEqual(self.param1.cast(False), False)
        self.assertEqual(self.param1.cast(None), None)

    def test_HierarchicalValidation(self) -> None:
        self.assertFalse(self.param1.is_hierarchical)
        with self.assertRaises(NotImplementedError):
            self.param1.dependents

    def test_Hierarchical(self) -> None:
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

    def test_available_flags(self) -> None:
        fixed_flags = ["is_fidelity", "is_hierarchical"]
        self.assertListEqual(self.param1.available_flags, fixed_flags)
        self.assertListEqual(self.param2.available_flags, fixed_flags)

    def test_domain_repr(self) -> None:
        self.assertEqual(self.param1.domain_repr, "value=True")
        self.assertEqual(self.param2.domain_repr, "value='foo'")

    def test_summary_dict(self) -> None:
        self.assertDictEqual(
            self.param1.summary_dict,
            {
                "name": "x",
                "type": "Fixed",
                "domain": "value=True",
                "parameter_type": "bool",
            },
        )
        self.assertDictEqual(
            self.param2.summary_dict,
            {
                "name": "y",
                "type": "Fixed",
                "domain": "value='foo'",
                "parameter_type": "string",
            },
        )


class ParameterEqualityTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.fixed_parameter = FixedParameter(
            name="x", parameter_type=ParameterType.BOOL, value=True
        )
        self.choice_parameter = ChoiceParameter(
            name="x", parameter_type=ParameterType.STRING, values=["foo", "bar", "baz"]
        )

    def test_NotEqual(self) -> None:
        self.assertNotEqual(self.fixed_parameter, self.choice_parameter)

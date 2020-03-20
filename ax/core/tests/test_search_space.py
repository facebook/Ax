#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.arm import Arm
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.core.search_space import SearchSpace
from ax.utils.common.testutils import TestCase


TOTAL_PARAMS = 6
TUNABLE_PARAMS = 4


class SearchSpaceTest(TestCase):
    def setUp(self):
        self.a = RangeParameter(
            name="a", parameter_type=ParameterType.FLOAT, lower=0.5, upper=5.5
        )
        self.b = RangeParameter(
            name="b", parameter_type=ParameterType.INT, lower=2, upper=10
        )
        self.c = ChoiceParameter(
            name="c", parameter_type=ParameterType.STRING, values=["foo", "bar", "baz"]
        )
        self.d = FixedParameter(name="d", parameter_type=ParameterType.BOOL, value=True)
        self.e = ChoiceParameter(
            name="e", parameter_type=ParameterType.FLOAT, values=[0.0, 0.1, 0.2, 0.5]
        )
        self.f = RangeParameter(
            name="f",
            parameter_type=ParameterType.INT,
            lower=2,
            upper=10,
            log_scale=True,
        )
        self.g = RangeParameter(
            name="g", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        )
        self.parameters = [self.a, self.b, self.c, self.d, self.e, self.f]
        self.ss1 = SearchSpace(parameters=self.parameters)
        self.ss2 = SearchSpace(
            parameters=self.parameters,
            parameter_constraints=[
                OrderConstraint(lower_parameter=self.a, upper_parameter=self.b)
            ],
        )
        self.ss1_repr = (
            "SearchSpace("
            "parameters=["
            "RangeParameter(name='a', parameter_type=FLOAT, range=[0.5, 5.5]), "
            "RangeParameter(name='b', parameter_type=INT, range=[2, 10]), "
            "ChoiceParameter(name='c', parameter_type=STRING, "
            "values=['foo', 'bar', 'baz']), "
            "FixedParameter(name='d', parameter_type=BOOL, value=True), "
            "ChoiceParameter(name='e', parameter_type=FLOAT, "
            "values=[0.0, 0.1, 0.2, 0.5]), "
            "RangeParameter(name='f', parameter_type=INT, range=[2, 10], "
            "log_scale=True)], "
            "parameter_constraints=[])"
        )
        self.ss2_repr = (
            "SearchSpace("
            "parameters=["
            "RangeParameter(name='a', parameter_type=FLOAT, range=[0.5, 5.5]), "
            "RangeParameter(name='b', parameter_type=INT, range=[2, 10]), "
            "ChoiceParameter(name='c', parameter_type=STRING, "
            "values=['foo', 'bar', 'baz']), "
            "FixedParameter(name='d', parameter_type=BOOL, value=True), "
            "ChoiceParameter(name='e', parameter_type=FLOAT, "
            "values=[0.0, 0.1, 0.2, 0.5]), "
            "RangeParameter(name='f', parameter_type=INT, range=[2, 10], "
            "log_scale=True)], "
            "parameter_constraints=[OrderConstraint(a <= b)])"
        )

    def testEq(self):
        ss2 = SearchSpace(
            parameters=self.parameters,
            parameter_constraints=[
                OrderConstraint(lower_parameter=self.a, upper_parameter=self.b)
            ],
        )
        self.assertEqual(self.ss2, ss2)
        self.assertNotEqual(self.ss1, self.ss2)

    def testProperties(self):
        self.assertEqual(len(self.ss1.parameters), TOTAL_PARAMS)
        self.assertTrue("a" in self.ss1.parameters)
        self.assertTrue(len(self.ss1.tunable_parameters), TUNABLE_PARAMS)
        self.assertFalse("d" in self.ss1.tunable_parameters)
        self.assertTrue(len(self.ss1.parameter_constraints) == 0)
        self.assertTrue(len(self.ss2.parameter_constraints) == 1)

    def testRepr(self):
        self.assertEqual(str(self.ss2), self.ss2_repr)
        self.assertEqual(str(self.ss1), self.ss1_repr)

    def testSetter(self):
        new_c = SumConstraint(
            parameters=[self.a, self.b], is_upper_bound=True, bound=10
        )
        self.ss2.add_parameter_constraints([new_c])
        self.assertEqual(len(self.ss2.parameter_constraints), 2)

        self.ss2.set_parameter_constraints([])
        self.assertEqual(len(self.ss2.parameter_constraints), 0)

        update_p = RangeParameter(
            name="b", parameter_type=ParameterType.INT, lower=10, upper=20
        )
        self.ss2.add_parameter(self.g)
        self.assertEqual(len(self.ss2.parameters), TOTAL_PARAMS + 1)

        self.ss2.update_parameter(update_p)
        self.assertEqual(self.ss2.parameters["b"].lower, 10)

    def testBadConstruction(self):
        # Duplicate parameter
        with self.assertRaises(ValueError):
            p1 = self.parameters + [self.parameters[0]]
            SearchSpace(parameters=p1, parameter_constraints=[])

        # Constraint on non-existent parameter
        with self.assertRaises(ValueError):
            SearchSpace(
                parameters=self.parameters,
                parameter_constraints=[
                    OrderConstraint(lower_parameter=self.a, upper_parameter=self.g)
                ],
            )

        # Vanilla Constraint on non-existent parameter
        with self.assertRaises(ValueError):
            SearchSpace(
                parameters=self.parameters,
                parameter_constraints=[
                    ParameterConstraint(constraint_dict={"g": 1}, bound=0)
                ],
            )

        # Constraint on non-numeric parameter
        with self.assertRaises(ValueError):
            SearchSpace(
                parameters=self.parameters,
                parameter_constraints=[
                    OrderConstraint(lower_parameter=self.a, upper_parameter=self.d)
                ],
            )

        # Constraint on choice parameter
        with self.assertRaises(ValueError):
            SearchSpace(
                parameters=self.parameters,
                parameter_constraints=[
                    OrderConstraint(lower_parameter=self.a, upper_parameter=self.e)
                ],
            )

        # Constraint on logscale parameter
        with self.assertRaises(ValueError):
            SearchSpace(
                parameters=self.parameters,
                parameter_constraints=[
                    OrderConstraint(lower_parameter=self.a, upper_parameter=self.f)
                ],
            )

        # Constraint on mismatched parameter
        with self.assertRaises(ValueError):
            wrong_a = self.a.clone()
            wrong_a.update_range(upper=10)
            SearchSpace(
                parameters=self.parameters,
                parameter_constraints=[
                    OrderConstraint(lower_parameter=wrong_a, upper_parameter=self.b)
                ],
            )

    def testBadSetter(self):
        new_p = RangeParameter(
            name="b", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        )

        # Add duplicate parameter
        with self.assertRaises(ValueError):
            self.ss1.add_parameter(new_p)

        # Update parameter to different type
        with self.assertRaises(ValueError):
            self.ss1.update_parameter(new_p)

        # Update non-existent parameter
        new_p = RangeParameter(
            name="g", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        )
        with self.assertRaises(ValueError):
            self.ss1.update_parameter(new_p)

    def testCheckMembership(self):
        p_dict = {"a": 1.0, "b": 5, "c": "foo", "d": True, "e": 0.2, "f": 5}

        # Valid
        self.assertTrue(self.ss2.check_membership(p_dict))

        # Value out of range
        p_dict["a"] = 20.0
        self.assertFalse(self.ss2.check_membership(p_dict))
        with self.assertRaises(ValueError):
            self.ss2.check_membership(p_dict, raise_error=True)

        # Violate constraints
        p_dict["a"] = 5.3
        self.assertFalse(self.ss2.check_membership(p_dict))
        with self.assertRaises(ValueError):
            self.ss2.check_membership(p_dict, raise_error=True)

        # Incomplete param dict
        p_dict.pop("a")
        self.assertFalse(self.ss2.check_membership(p_dict))
        with self.assertRaises(ValueError):
            self.ss2.check_membership(p_dict, raise_error=True)

        # Unknown parameter
        p_dict["q"] = 40
        self.assertFalse(self.ss2.check_membership(p_dict))
        with self.assertRaises(ValueError):
            self.ss2.check_membership(p_dict, raise_error=True)

    def testCheckTypes(self):
        p_dict = {"a": 1.0, "b": 5, "c": "foo", "d": True, "e": 0.2, "f": 5}

        # Valid
        self.assertTrue(self.ss2.check_membership(p_dict))

        # Invalid type
        p_dict["b"] = 5.2
        self.assertFalse(self.ss2.check_types(p_dict))
        with self.assertRaises(ValueError):
            self.ss2.check_types(p_dict, raise_error=True)
        p_dict["b"] = 5

        # Incomplete param dict
        p_dict.pop("a")
        self.assertFalse(self.ss2.check_types(p_dict))
        with self.assertRaises(ValueError):
            self.ss2.check_types(p_dict, raise_error=True)

        # Unknown parameter
        p_dict["q"] = 40
        self.assertFalse(self.ss2.check_types(p_dict))
        with self.assertRaises(ValueError):
            self.ss2.check_types(p_dict, raise_error=True)

    def testCastArm(self):
        p_dict = {"a": 1.0, "b": 5.0, "c": "foo", "d": True, "e": 0.2, "f": 5}

        # Check "b" parameter goes from float to int
        self.assertTrue(isinstance(p_dict["b"], float))
        new_arm = self.ss2.cast_arm(Arm(p_dict))
        self.assertTrue(isinstance(new_arm.parameters["b"], int))

        # Unknown parameter should be unchanged
        p_dict["q"] = 40
        new_arm = self.ss2.cast_arm(Arm(p_dict))
        self.assertTrue(isinstance(new_arm.parameters["q"], int))

    def testCopy(self):
        a = RangeParameter("a", ParameterType.FLOAT, 1.0, 5.5)
        b = RangeParameter("b", ParameterType.FLOAT, 2.0, 5.5)
        c = ChoiceParameter("c", ParameterType.INT, [2, 3])
        ss = SearchSpace(
            parameters=[a, b, c],
            parameter_constraints=[
                OrderConstraint(lower_parameter=a, upper_parameter=b)
            ],
        )
        ss_copy = ss.clone()
        self.assertEqual(len(ss_copy.parameters), len(ss_copy.parameters))
        self.assertEqual(
            len(ss_copy.parameter_constraints), len(ss_copy.parameter_constraints)
        )

        ss_copy.add_parameter(FixedParameter("d", ParameterType.STRING, "h"))
        self.assertNotEqual(len(ss_copy.parameters), len(ss.parameters))

    def testOutOfDesignArm(self):
        arm1 = self.ss1.out_of_design_arm()
        arm2 = self.ss2.out_of_design_arm()
        arm1_nones = [p is None for p in arm1.parameters.values()]
        self.assertTrue(all(arm1_nones))
        self.assertTrue(arm1 == arm2)

    def testConstructArm(self):
        # Test constructing an arm of default values
        arm = self.ss1.construct_arm(name="test")
        self.assertEqual(arm.name, "test")
        for p_name in self.ss1.parameters.keys():
            self.assertTrue(p_name in arm.parameters)
            self.assertEqual(arm.parameters[p_name], None)

        # Test constructing an arm with a custom value
        arm = self.ss1.construct_arm({"a": 1.0})
        for p_name in self.ss1.parameters.keys():
            self.assertTrue(p_name in arm.parameters)
            if p_name == "a":
                self.assertEqual(arm.parameters[p_name], 1.0)
            else:
                self.assertEqual(arm.parameters[p_name], None)

        # Test constructing an arm with a bad param name
        with self.assertRaises(ValueError):
            self.ss1.construct_arm({"IDONTEXIST_a": 1.0})

        # Test constructing an arm with a bad param name
        with self.assertRaises(ValueError):
            self.ss1.construct_arm({"a": "notafloat"})

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
import warnings
from random import choice
from typing import cast
from unittest import mock

import pandas as pd
from ax.core.arm import Arm
from ax.core.observation import ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    Parameter,
    ParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.core.parameter_distribution import ParameterDistribution
from ax.core.search_space import (
    HierarchicalSearchSpace,
    RobustSearchSpace,
    RobustSearchSpaceDigest,
    SearchSpace,
    SearchSpaceDigest,
)
from ax.core.types import TParameterization
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_hierarchical_search_space,
    get_l2_reg_weight_parameter,
    get_lr_parameter,
    get_model_parameter,
    get_num_boost_rounds_parameter,
    get_parameter_constraint,
)
from pyre_extensions import assert_is_instance

TOTAL_PARAMS = 6
TUNABLE_PARAMS = 4
RANGE_PARAMS = 3


class SearchSpaceTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
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
        self.parameters: list[Parameter] = [
            self.a,
            self.b,
            self.c,
            self.d,
            self.e,
            self.f,
        ]
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
            "values=['foo', 'bar', 'baz'], is_ordered=False, sort_values=False), "
            "FixedParameter(name='d', parameter_type=BOOL, value=True), "
            "ChoiceParameter(name='e', parameter_type=FLOAT, "
            "values=[0.0, 0.1, 0.2, 0.5], is_ordered=True, sort_values=True), "
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
            "values=['foo', 'bar', 'baz'], is_ordered=False, sort_values=False), "
            "FixedParameter(name='d', parameter_type=BOOL, value=True), "
            "ChoiceParameter(name='e', parameter_type=FLOAT, "
            "values=[0.0, 0.1, 0.2, 0.5], is_ordered=True, sort_values=True), "
            "RangeParameter(name='f', parameter_type=INT, range=[2, 10], "
            "log_scale=True)], "
            "parameter_constraints=[OrderConstraint(a <= b)])"
        )

    def test_Eq(self) -> None:
        ss2 = SearchSpace(
            parameters=self.parameters,
            parameter_constraints=[
                OrderConstraint(lower_parameter=self.a, upper_parameter=self.b)
            ],
        )
        self.assertEqual(self.ss2, ss2)
        self.assertNotEqual(self.ss1, self.ss2)

    def test_Properties(self) -> None:
        self.assertEqual(len(self.ss1.parameters), TOTAL_PARAMS)
        self.assertTrue("a" in self.ss1.parameters)
        self.assertTrue(len(self.ss1.tunable_parameters), TUNABLE_PARAMS)
        self.assertFalse("d" in self.ss1.tunable_parameters)
        self.assertTrue(len(self.ss1.range_parameters), RANGE_PARAMS)
        self.assertFalse("c" in self.ss1.range_parameters)
        self.assertTrue(len(self.ss1.parameter_constraints) == 0)
        self.assertTrue(len(self.ss2.parameter_constraints) == 1)

    def test_Repr(self) -> None:
        self.assertEqual(str(self.ss2), self.ss2_repr)
        self.assertEqual(str(self.ss1), self.ss1_repr)

    def test_Setter(self) -> None:
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
        # pyre-fixme[16]: `Parameter` has no attribute `lower`.
        self.assertEqual(self.ss2.parameters["b"].lower, 10)

    def test_BadConstruction(self) -> None:
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

    def test_BadSetter(self) -> None:
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

    def test_CheckMembership(self) -> None:
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

    def test_CheckTypes(self) -> None:
        p_dict = {"a": 1.0, "b": 5, "c": "foo", "d": True, "e": 0.2, "f": 5}

        # Valid
        # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool, float,
        #  int, str]]` but got `Dict[str, Union[float, str]]`.
        self.assertTrue(self.ss2.check_types(p_dict))

        # Invalid type
        p_dict["b"] = 5.2
        # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool, float,
        #  int, str]]` but got `Dict[str, Union[float, str]]`.
        self.assertFalse(self.ss2.check_types(p_dict))
        with self.assertRaises(ValueError):
            # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Dict[str, Union[float, str]]`.
            self.ss2.check_types(p_dict, raise_error=True)
        p_dict["b"] = 5

        # Unknown parameter
        p_dict["q"] = 40
        self.assertTrue(self.ss2.check_types(p_dict))  # pyre-fixme[6]
        self.assertFalse(
            self.ss2.check_types(p_dict, allow_extra_params=False)  # pyre-fixme[6]
        )
        with self.assertRaises(ValueError):
            self.ss2.check_types(
                p_dict,  # pyre-fixme[6]
                allow_extra_params=False,
                raise_error=True,
            )

    def test_CastArm(self) -> None:
        p_dict = {"a": 1.0, "b": 5.0, "c": "foo", "d": True, "e": 0.2, "f": 5}

        # Check "b" parameter goes from float to int
        self.assertTrue(isinstance(p_dict["b"], float))
        new_arm = self.ss2.cast_arm(Arm(p_dict))
        self.assertTrue(isinstance(new_arm.parameters["b"], int))

        # Unknown parameter should be unchanged
        p_dict["q"] = 40
        new_arm = self.ss2.cast_arm(Arm(p_dict))
        self.assertTrue(isinstance(new_arm.parameters["q"], int))

    def test_Copy(self) -> None:
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

    def test_OutOfDesignArm(self) -> None:
        arm1 = self.ss1.out_of_design_arm()
        arm2 = self.ss2.out_of_design_arm()
        arm1_nones = [p is None for p in arm1.parameters.values()]
        self.assertTrue(all(arm1_nones))
        self.assertTrue(arm1 == arm2)

    def test_ConstructArm(self) -> None:
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

    def test_search_space_summary_df(self) -> None:
        min_search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x1", parameter_type=ParameterType.INT, lower=0, upper=2
                ),
                RangeParameter(
                    "x2",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.1,
                    upper=10,
                ),
            ]
        )
        max_search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x1", parameter_type=ParameterType.INT, lower=0, upper=2
                ),
                RangeParameter(
                    "x2",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.1,
                    upper=10,
                    log_scale=True,
                    is_fidelity=True,
                    target_value=10,
                ),
                FixedParameter("x3", parameter_type=ParameterType.BOOL, value=True),
                ChoiceParameter(
                    "x4",
                    parameter_type=ParameterType.STRING,
                    values=["a", "b", "c"],
                    is_ordered=False,
                    dependents={"a": ["x1", "x2"], "b": ["x4", "x5"], "c": ["x6"]},
                ),
                ChoiceParameter(
                    "x5",
                    parameter_type=ParameterType.STRING,
                    values=["d", "e", "f"],
                    is_ordered=True,
                ),
                RangeParameter(
                    name="x6",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=1.0,
                ),
            ]
        )
        df = max_search_space.summary_df
        expected_df = pd.DataFrame(
            data={
                "Name": ["x1", "x2", "x3", "x4", "x5", "x6"],
                "Type": ["Range", "Range", "Fixed", "Choice", "Choice", "Range"],
                "Domain": [
                    "range=[0, 2]",
                    "range=[0.1, 10.0]",
                    "value=True",
                    "values=['a', 'b', 'c']",
                    "values=['d', 'e', 'f']",
                    "range=[0.0, 1.0]",
                ],
                "Datatype": ["int", "float", "bool", "string", "string", "float"],
                "Flags": [
                    "None",
                    "fidelity, log_scale",
                    "None",
                    "unordered, hierarchical, unsorted",
                    "ordered, unsorted",
                    "None",
                ],
                "Target Value": ["None", 10.0, "None", "None", "None", "None"],
                "Dependent Parameters": [
                    "None",
                    "None",
                    "None",
                    {"a": ["x1", "x2"], "b": ["x4", "x5"], "c": ["x6"]},
                    "None",
                    "None",
                ],
            }
        )

        pd.testing.assert_frame_equal(df, expected_df)

        df = min_search_space.summary_df
        expected_df = pd.DataFrame(
            data={
                "Name": ["x1", "x2"],
                "Type": ["Range", "Range"],
                "Domain": ["range=[0, 2]", "range=[0.1, 10.0]"],
                "Datatype": ["int", "float"],
            }
        )
        pd.testing.assert_frame_equal(df, expected_df)


class SearchSpaceDigestTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.kwargs = {
            "feature_names": ["a", "b", "c"],
            "bounds": [(0.0, 1.0), (0, 2), (0, 4)],
            "ordinal_features": [1],
            "categorical_features": [2],
            "discrete_choices": {1: [0, 1, 2], 2: [0, 0.25, 4.0]},
            "task_features": [3],
            "fidelity_features": [0],
            "target_values": {0: 1.0},
            "robust_digest": None,
        }

    def test_SearchSpaceDigest(self) -> None:
        # test required fields
        with self.assertRaises(TypeError):
            # pyre-fixme[20]: Argument `feature_names` expected.
            SearchSpaceDigest(bounds=[])
        with self.assertRaises(TypeError):
            # pyre-fixme[20]: Argument `bounds` expected.
            SearchSpaceDigest(feature_names=[])
        # test instantiation
        ssd = SearchSpaceDigest(**self.kwargs)
        self.assertEqual(dataclasses.asdict(ssd), self.kwargs)
        # test default instatiation
        for arg in self.kwargs:
            if arg in {"feature_names", "bounds"}:
                continue
            ssd = SearchSpaceDigest(
                **{k: v for k, v in self.kwargs.items() if k != arg}
            )


class RobustSearchSpaceDigestTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.kwargs = {
            "sample_param_perturbations": lambda: 1,
            "sample_environmental": lambda: 2,
            "environmental_variables": ["a"],
            "multiplicative": False,
        }

    def test_robust_search_space_digest(self) -> None:
        # test post init
        with self.assertRaises(UserInputError):
            RobustSearchSpaceDigest()
        # test instantiation
        rssd = RobustSearchSpaceDigest(**self.kwargs)
        self.assertEqual(dataclasses.asdict(rssd), self.kwargs)
        # test default instantiation
        for arg in self.kwargs:
            rssd = RobustSearchSpaceDigest(
                **{k: v for k, v in self.kwargs.items() if k != arg}
            )


class HierarchicalSearchSpaceTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.model_parameter = get_model_parameter()
        self.lr_parameter = get_lr_parameter()
        self.l2_reg_weight_parameter = get_l2_reg_weight_parameter()
        self.num_boost_rounds_parameter = get_num_boost_rounds_parameter()
        self.hss_1 = get_hierarchical_search_space()
        self.use_linear_parameter = ChoiceParameter(
            name="use_linear",  # Contrived!
            parameter_type=ParameterType.BOOL,
            values=[True, False],
            dependents={
                True: ["model"],
            },
        )
        self.hss_2 = HierarchicalSearchSpace(
            parameters=[
                self.use_linear_parameter,
                self.model_parameter,
                self.lr_parameter,
                self.l2_reg_weight_parameter,
                self.num_boost_rounds_parameter,
            ]
        )
        self.hss_with_fixed = HierarchicalSearchSpace(
            parameters=[
                self.use_linear_parameter,
                FixedParameter(
                    name="model",
                    value="fixed_model",
                    parameter_type=ParameterType.STRING,
                ),
            ]
        )
        self.model_2_parameter = ChoiceParameter(
            name="model_2",
            parameter_type=ParameterType.STRING,
            values=["Linear", "XGBoost"],
            dependents={
                "Linear": ["learning_rate", "l2_reg_weight"],
                "XGBoost": ["num_boost_rounds"],
            },
        )
        self.hss_with_constraints = HierarchicalSearchSpace(
            parameters=[
                self.model_parameter,
                self.lr_parameter,
                self.l2_reg_weight_parameter,
                self.num_boost_rounds_parameter,
            ],
            parameter_constraints=[
                get_parameter_constraint(
                    param_x=self.lr_parameter.name,
                    param_y=self.l2_reg_weight_parameter.name,
                ),
            ],
        )
        self.hss_1_arm_1_flat = Arm(
            parameters={
                "model": "Linear",
                "learning_rate": 0.01,
                "l2_reg_weight": 0.0001,
                "num_boost_rounds": 12,
            }
        )
        self.hss_1_arm_2_flat = Arm(
            parameters={
                "model": "XGBoost",
                "learning_rate": 0.01,
                "l2_reg_weight": 0.0001,
                "num_boost_rounds": 12,
            }
        )
        self.hss_1_missing_params: TParameterization = {
            "model": "Linear",
            "l2_reg_weight": 0.0001,
            "num_boost_rounds": 12,
        }
        self.hss_1_arm_missing_param = Arm(parameters=self.hss_1_missing_params)
        self.hss_1_arm_1_cast = Arm(
            parameters={
                "model": "Linear",
                "learning_rate": 0.01,
                "l2_reg_weight": 0.0001,
            }
        )
        self.hss_1_arm_2_cast = Arm(
            parameters={
                "model": "XGBoost",
                "num_boost_rounds": 12,
            }
        )

    def test_init(self) -> None:
        self.assertEqual(self.hss_1._root, self.model_parameter)
        self.assertEqual(
            self.hss_1._all_parameter_names,
            {"l2_reg_weight", "learning_rate", "num_boost_rounds", "model"},
        )
        self.assertEqual(self.hss_2._root, self.use_linear_parameter)
        self.assertEqual(
            self.hss_2._all_parameter_names,
            {
                "l2_reg_weight",
                "learning_rate",
                "num_boost_rounds",
                "model",
                "use_linear",
            },
        )

    def test_validation(self) -> None:
        # Case where dependent parameter is not in the search space.
        with self.assertRaisesRegex(ValueError, ".* 'l2_reg_weight' is not part"):
            HierarchicalSearchSpace(
                parameters=[
                    ChoiceParameter(
                        name="model",
                        parameter_type=ParameterType.STRING,
                        values=["Linear", "XGBoost"],
                        dependents={
                            "Linear": ["learning_rate", "l2_reg_weight"],
                            "XGBoost": ["num_boost_rounds"],
                        },
                    ),
                    self.lr_parameter,
                    self.num_boost_rounds_parameter,
                ]
            )

        # Case where there are two root-parameter candidates.
        with self.assertRaisesRegex(NotImplementedError, "Could not find the root"):
            HierarchicalSearchSpace(
                parameters=[
                    self.model_parameter,
                    self.model_2_parameter,
                    self.lr_parameter,
                    self.l2_reg_weight_parameter,
                    self.num_boost_rounds_parameter,
                ]
            )

        # TODO: Test case where subtrees are not independent.
        with self.assertRaisesRegex(UserInputError, ".* contain the same parameters"):
            HierarchicalSearchSpace(
                parameters=[
                    ChoiceParameter(
                        name="root",
                        parameter_type=ParameterType.BOOL,
                        values=[True, False],
                        dependents={
                            True: ["model", "model_2"],
                        },
                    ),
                    self.model_parameter,
                    self.model_2_parameter,
                    self.lr_parameter,
                    self.l2_reg_weight_parameter,
                    self.num_boost_rounds_parameter,
                ]
            )

    def test_hierarchical_structure_str(self) -> None:
        self.assertEqual(
            self.hss_1.hierarchical_structure_str(),
            f"{self.hss_1.root}\n\t(Linear)\n\t\t{self.lr_parameter}\n\t\t"
            f"{self.l2_reg_weight_parameter}\n\t(XGBoost)\n\t\t"
            f"{self.num_boost_rounds_parameter}\n",
        )
        self.assertEqual(
            self.hss_1.hierarchical_structure_str(parameter_names_only=True),
            f"{self.hss_1.root.name}\n\t(Linear)\n\t\t{self.lr_parameter.name}"
            f"\n\t\t{self.l2_reg_weight_parameter.name}\n\t(XGBoost)\n\t\t"
            f"{self.num_boost_rounds_parameter.name}\n",
        )

    def test_flatten(self) -> None:
        # Test on basic HSS.
        flattened_hss_1 = self.hss_1.flatten()
        self.assertIsNot(flattened_hss_1, self.hss_1)
        self.assertEqual(type(flattened_hss_1), SearchSpace)
        self.assertFalse(isinstance(flattened_hss_1, HierarchicalSearchSpace))
        self.assertEqual(flattened_hss_1.parameters, self.hss_1.parameters)
        self.assertEqual(
            flattened_hss_1.parameter_constraints, self.hss_1.parameter_constraints
        )
        self.assertTrue(str(self.hss_1).startswith("HierarchicalSearchSpace"))
        self.assertTrue(str(flattened_hss_1).startswith("SearchSpace"))

        # Test on HSS with constraints.
        flattened_hss_with_constraints = self.hss_with_constraints.flatten()
        self.assertIsNot(flattened_hss_with_constraints, self.hss_with_constraints)
        self.assertEqual(type(flattened_hss_with_constraints), SearchSpace)
        self.assertFalse(
            isinstance(flattened_hss_with_constraints, HierarchicalSearchSpace)
        )
        self.assertEqual(
            flattened_hss_with_constraints.parameters,
            self.hss_with_constraints.parameters,
        )
        self.assertEqual(
            flattened_hss_with_constraints.parameter_constraints,
            self.hss_with_constraints.parameter_constraints,
        )
        self.assertTrue(
            str(self.hss_with_constraints).startswith("HierarchicalSearchSpace")
        )
        self.assertTrue(str(flattened_hss_with_constraints).startswith("SearchSpace"))

    def test_cast_arm(self) -> None:
        # This uses _cast_parameterization with check_all_parameters_present=True.
        self.assertEqual(  # Check one subtree.
            self.hss_1._cast_arm(arm=self.hss_1_arm_1_flat),
            self.hss_1_arm_1_cast,
        )
        self.assertEqual(  # Check other subtree.
            self.hss_1._cast_arm(arm=self.hss_1_arm_2_flat),
            self.hss_1_arm_2_cast,
        )
        self.assertEqual(  # Check already-cast case.
            self.hss_1._cast_arm(arm=self.hss_1_arm_1_cast),
            self.hss_1_arm_1_cast,
        )
        with self.assertRaises(RuntimeError):
            self.hss_1._cast_arm(arm=self.hss_1_arm_missing_param)

    def test_cast_observation_features(self) -> None:
        # This uses _cast_parameterization with check_all_parameters_present=False.
        # Ensure that during casting, full parameterization is saved
        # in metadata and actual parameterization is cast to HSS.
        hss_1_obs_feats_1 = ObservationFeatures.from_arm(arm=self.hss_1_arm_1_flat)
        hss_1_obs_feats_1_cast = self.hss_1.cast_observation_features(
            observation_features=hss_1_obs_feats_1
        )
        self.assertEqual(  # Check one subtree.
            hss_1_obs_feats_1_cast.parameters,
            ObservationFeatures.from_arm(arm=self.hss_1_arm_1_cast).parameters,
        )
        self.assertEqual(  # Check one subtree.
            hss_1_obs_feats_1_cast.metadata.get(Keys.FULL_PARAMETERIZATION),
            hss_1_obs_feats_1.parameters,
        )
        # Check that difference with observation features made from cast arm
        # is only in metadata (to ensure only parameters and metadata are
        # manipulated during casting).
        hss_1_obs_feats_1_cast.metadata = None
        self.assertEqual(
            hss_1_obs_feats_1_cast,
            ObservationFeatures.from_arm(arm=self.hss_1_arm_1_cast),
        )

    def test_cast_parameterization(self) -> None:
        # NOTE: This is also tested in test_cast_arm & test_cast_observation_features.
        with self.assertRaisesRegex(RuntimeError, "not in parameterization to cast"):
            self.hss_1._cast_parameterization(
                parameters=self.hss_1_missing_params,
                check_all_parameters_present=True,
            )
        # An active leaf param is missing, it'll get ignored. There's an inactive
        # leaf param, that'll just get filtered out.
        self.assertEqual(
            self.hss_1._cast_parameterization(
                parameters=self.hss_1_missing_params,
                check_all_parameters_present=False,
            ),
            {"l2_reg_weight": 0.0001, "model": "Linear"},
        )
        # A hierarchical param is missing, all its dependents will be ignored.
        # In this case, it is the root param, so we'll have empty parameterization.
        self.assertEqual(
            self.hss_1._cast_parameterization(
                parameters={
                    "l2_reg_weight": 0.0001,
                    "num_boost_rounds": 12,
                },
                check_all_parameters_present=False,
            ),
            {},
        )

    def test_flatten_observation_features(self) -> None:
        # Ensure that during casting, full parameterization is saved
        # in metadata and actual parameterization is cast to HSS; during
        # flattening, parameterization in metadata is used ot inject back
        # the parameters removed during casting.
        hss_1_obs_feats_1 = ObservationFeatures.from_arm(arm=self.hss_1_arm_1_flat)
        hss_1_obs_feats_1_cast = self.hss_1.cast_observation_features(
            observation_features=hss_1_obs_feats_1
        )
        for inject_dummy in (True, False):
            hss_1_obs_feats_1_flattened = self.hss_1.flatten_observation_features(
                observation_features=hss_1_obs_feats_1_cast,
                inject_dummy_values_to_complete_flat_parameterization=inject_dummy,
            )
            self.assertEqual(  # Cast-flatten roundtrip.
                hss_1_obs_feats_1.parameters,
                hss_1_obs_feats_1_flattened.parameters,
            )
            self.assertEqual(  # Check that both cast and flattened have full params.
                hss_1_obs_feats_1_cast.metadata.get(Keys.FULL_PARAMETERIZATION),
                hss_1_obs_feats_1_flattened.metadata.get(Keys.FULL_PARAMETERIZATION),
            )
        # Check that flattening observation features without metadata does nothing.
        # Does not warn here since it already has all parameters.
        with warnings.catch_warnings(record=True) as ws:
            self.assertEqual(
                self.hss_1.flatten_observation_features(
                    observation_features=hss_1_obs_feats_1
                ),
                hss_1_obs_feats_1,
            )
        self.assertFalse(
            any("Cannot flatten observation features" in str(w.message) for w in ws)
        )
        # This one warns since it is missing some parameters.
        obs_ft_missing = ObservationFeatures.from_arm(arm=self.hss_1_arm_missing_param)
        with warnings.catch_warnings(record=True) as ws:
            self.assertEqual(
                self.hss_1.flatten_observation_features(
                    observation_features=obs_ft_missing
                ),
                obs_ft_missing,
            )
        self.assertTrue(
            any("Cannot flatten observation features" in str(w.message) for w in ws)
        )
        # Check that no warning is raised if the observation feature doesn't
        # have parameterization (for trial-index only features).
        obs_ft = ObservationFeatures(parameters={}, trial_index=0)
        with warnings.catch_warnings(record=True) as ws:
            self.assertEqual(
                self.hss_1.flatten_observation_features(observation_features=obs_ft),
                obs_ft,
            )
        self.assertFalse(
            any("Cannot flatten observation features" in str(w.message) for w in ws)
        )

    @mock.patch(f"{HierarchicalSearchSpace.__module__}.uniform", return_value=0.6)
    def test_flatten_observation_features_inject_dummy_parameter_values_with_random(
        self, mock_uniform: mock.MagicMock
    ) -> None:
        # Case 1: Linear arm
        hss_obs_feats = ObservationFeatures.from_arm(arm=self.hss_1_arm_1_cast)
        hss_obs_feats_flattened = self.hss_1.flatten_observation_features(
            observation_features=hss_obs_feats, use_random_dummy_values=True
        )
        mock_uniform.assert_not_called()
        self.assertNotIn("num_boost_rounds", hss_obs_feats_flattened.parameters)
        flattened_with_dummies = self.hss_1.flatten_observation_features(
            observation_features=hss_obs_feats,
            inject_dummy_values_to_complete_flat_parameterization=True,
            use_random_dummy_values=True,
        ).parameters
        mock_uniform.assert_called()
        self.assertIn("num_boost_rounds", flattened_with_dummies)
        self.assertEqual(
            flattened_with_dummies["num_boost_rounds"],
            1,  # int(0.6 + 0.5) = floor(0.6 + 0.5) = 1
        )
        self.assertIsInstance(  # Ensure we coerced parameter type correctly, too.
            flattened_with_dummies["num_boost_rounds"],
            int,
        )

        # Case 2: XGBoost arm
        mock_uniform.reset_mock()
        hss_obs_feats = ObservationFeatures.from_arm(arm=self.hss_1_arm_2_cast)
        hss_obs_feats_flattened = self.hss_1.flatten_observation_features(
            observation_features=hss_obs_feats, use_random_dummy_values=True
        )
        mock_uniform.assert_not_called()
        self.assertNotIn("learning_rate", hss_obs_feats_flattened.parameters)
        self.assertNotIn("l2_reg_weight", hss_obs_feats_flattened.parameters)
        flattened_with_dummies = (
            self.hss_1.flatten_observation_features(
                observation_features=hss_obs_feats,
                inject_dummy_values_to_complete_flat_parameterization=True,
                use_random_dummy_values=True,
            )
        ).parameters
        mock_uniform.assert_called()
        self.assertIn("learning_rate", flattened_with_dummies)
        self.assertIn("l2_reg_weight", flattened_with_dummies)
        self.assertEqual(
            flattened_with_dummies["learning_rate"],
            mock_uniform.return_value,
        )
        self.assertEqual(
            flattened_with_dummies["l2_reg_weight"],
            mock_uniform.return_value,
        )

        # Case 3: test setting of choice parameters
        with mock.patch(
            f"{HierarchicalSearchSpace.__module__}.choice", wraps=choice
        ) as mock_choice:
            flattened_only_dummies = self.hss_2.flatten_observation_features(
                observation_features=ObservationFeatures(
                    parameters={"num_boost_rounds": 12}
                ),
                inject_dummy_values_to_complete_flat_parameterization=True,
                use_random_dummy_values=True,
            ).parameters
            self.assertEqual(
                mock_choice.call_args_list,
                [mock.call([False, True]), mock.call(["Linear", "XGBoost"])],
            )
        self.assertEqual(
            set(flattened_only_dummies.keys()), set(self.hss_2.parameters.keys())
        )

        # Case 4: test setting of fixed parameters
        flattened_only_dummies = self.hss_with_fixed.flatten_observation_features(
            observation_features=ObservationFeatures(parameters={"use_linear": True}),
            inject_dummy_values_to_complete_flat_parameterization=True,
            use_random_dummy_values=True,
        ).parameters
        self.assertEqual(
            set(flattened_only_dummies.keys()),
            set(self.hss_with_fixed.parameters.keys()),
        )

    def test_flatten_observation_features_inject_dummy_parameter_values_non_random(
        self,
    ) -> None:
        """Check behavior with different parameter cases:
        Choice, Int-Range, Log-Range & Logit-Range.

        HSS2 has enough parameters that we can modify to test all cases.
        - `use_linear` is bool-Choice
        - `model` is string-Choice
        - `learning_rate` is Range, can be made log-scale
        - `l2_reg_weight` is Range, can be made logit-scale
        - `num_boost_rounds` is Int-Range.
        """
        assert_is_instance(
            self.hss_2.parameters["learning_rate"], RangeParameter
        )._log_scale = True
        assert_is_instance(
            self.hss_2.parameters["l2_reg_weight"], RangeParameter
        )._logit_scale = True
        # This has no other parameters on it, so they should all be set to
        # middle value in their respective domains.
        obs_ft = ObservationFeatures(parameters={"use_linear": False})
        flat_obs_ft = self.hss_2.flatten_observation_features(
            observation_features=obs_ft,
            inject_dummy_values_to_complete_flat_parameterization=True,
            use_random_dummy_values=False,
        )
        expected_parameters = {
            "use_linear": False,
            "model": "XGBoost",  # has two values, so it will be index 1.
            "learning_rate": 0.01,  # middle of 0.1 & 0.001 in log-scale.
            "l2_reg_weight": 0.00010004052867652256,  # middle of range in logit-scale.
            "num_boost_rounds": 15,  # mid-point of 10 & 20.
        }
        self.assertDictEqual(flat_obs_ft.parameters, expected_parameters)

    def test_flatten_observation_features_full_and_dummy(self) -> None:
        # Test flattening when both full features & inject dummy values
        # are specified. This is relevant if the full parameterization
        # is from some subset of the search space.
        # Get an obs feat with hss_1 parameterization in the metadata.
        hss_1_obs_feats_1 = ObservationFeatures.from_arm(arm=self.hss_1_arm_1_flat)
        hss_1_obs_feats_1_cast = self.hss_1.cast_observation_features(
            observation_features=hss_1_obs_feats_1
        )
        hss_1_flat_params = hss_1_obs_feats_1.parameters
        # Flatten it using hss_2, which requires an additional parameter.
        # This will work but miss a parameter.
        self.assertEqual(
            self.hss_2.flatten_observation_features(
                observation_features=hss_1_obs_feats_1_cast,
                inject_dummy_values_to_complete_flat_parameterization=False,
            ).parameters,
            hss_1_flat_params,
        )
        # Now try with inject dummy. This will add the mising param.
        hss_2_flat = self.hss_2.flatten_observation_features(
            observation_features=hss_1_obs_feats_1_cast,
            inject_dummy_values_to_complete_flat_parameterization=True,
        ).parameters
        self.assertNotEqual(hss_2_flat, hss_1_flat_params)
        self.assertEqual(
            {k: hss_2_flat[k] for k in hss_1_flat_params}, hss_1_flat_params
        )
        self.assertEqual(set(hss_2_flat.keys()), set(self.hss_2.parameters.keys()))


class TestRobustSearchSpace(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.a = RangeParameter(
            name="a", parameter_type=ParameterType.FLOAT, lower=0.5, upper=5.5
        )
        self.b = RangeParameter(
            name="b", parameter_type=ParameterType.INT, lower=2, upper=10
        )
        self.c = ChoiceParameter(
            name="c", parameter_type=ParameterType.STRING, values=["foo", "bar", "baz"]
        )
        self.parameters = [self.a, self.b, self.c]
        self.ab_dist = ParameterDistribution(
            parameters=["a", "b"],
            distribution_class="multivariate_normal",
            distribution_parameters={},
        )
        self.constraints = [
            OrderConstraint(lower_parameter=self.a, upper_parameter=self.b)
        ]
        self.rss1 = RobustSearchSpace(
            parameters=self.parameters,
            parameter_distributions=[self.ab_dist],
            parameter_constraints=cast(list[ParameterConstraint], self.constraints),
            num_samples=4,
        )

    def test_init_and_properties(self) -> None:
        # Setup some parameters and distributions.
        a_dist = ParameterDistribution(
            parameters=["a"],
            distribution_class="norm",
            distribution_parameters={"loc": 0.0, "scale": 1.0},
        )
        b_dist = ParameterDistribution(
            parameters=["b"],
            distribution_class="binom",
            distribution_parameters={"n": 2, "p": 0.3},
        )
        env1 = RangeParameter(
            name="env1", parameter_type=ParameterType.FLOAT, lower=0.5, upper=5.5
        )
        env2 = RangeParameter(
            name="env2", parameter_type=ParameterType.INT, lower=2.0, upper=10.0
        )
        choice_dist = ParameterDistribution(
            parameters=["c"],
            distribution_class="binom",
            distribution_parameters={"n": 2, "p": 0.3},
        )
        env1_dist = ParameterDistribution(
            parameters=["env1"],
            distribution_class="norm",
            distribution_parameters={"loc": 0.0, "scale": 1.0},
        )
        env1_dist_mul = ParameterDistribution(
            parameters=["env1"],
            distribution_class="norm",
            distribution_parameters={"loc": 0.0, "scale": 1.0},
            multiplicative=True,
        )
        env2_dist = ParameterDistribution(
            parameters=["env2"],
            distribution_class="binom",
            distribution_parameters={"n": 2, "p": 0.3},
        )
        mixed_dist = ParameterDistribution(
            parameters=["a", "env1"],
            distribution_class="multivariate_normal",
            distribution_parameters={"mean": [0.0, 0.0]},
        )
        # Error handling.
        with self.assertRaisesRegex(UserInputError, "Use SearchSpace instead."):
            RobustSearchSpace(
                parameters=self.parameters,
                parameter_distributions=[],
                num_samples=4,
            )
        with self.assertRaisesRegex(UserInputError, "positive integer"):
            RobustSearchSpace(
                parameters=self.parameters,
                parameter_distributions=[self.ab_dist],
                num_samples=-1,
            )
        with self.assertRaisesRegex(UnsupportedError, "all multiplicative"):
            mul_a_dist = a_dist.clone()
            mul_a_dist.multiplicative = True
            RobustSearchSpace(
                parameters=self.parameters,
                parameter_distributions=[mul_a_dist, b_dist],
                num_samples=4,
            )
        with self.assertRaisesRegex(UserInputError, "must be unique"):
            RobustSearchSpace(
                parameters=self.parameters,
                parameter_distributions=[env1_dist],
                num_samples=4,
                environmental_variables=[env1, env1],
            )
        with self.assertRaisesRegex(UserInputError, "must have a distribution"):
            RobustSearchSpace(
                parameters=self.parameters,
                parameter_distributions=[env1_dist],
                environmental_variables=[env1, env2],
                num_samples=4,
            )
        with self.assertRaisesRegex(UserInputError, "should not be repeated"):
            RobustSearchSpace(
                parameters=self.parameters,
                parameter_distributions=[a_dist],
                num_samples=4,
                environmental_variables=[self.a],
            )
        with self.assertRaisesRegex(
            UserInputError, "Distributions of environmental variables"
        ):
            RobustSearchSpace(
                parameters=self.parameters,
                parameter_distributions=[env1_dist_mul],
                num_samples=4,
                environmental_variables=[env1],
            )
        with self.assertRaisesRegex(UserInputError, "multiple parameter distributions"):
            RobustSearchSpace(
                parameters=self.parameters,
                parameter_distributions=[a_dist, a_dist],
                num_samples=4,
            )
        with self.assertRaisesRegex(UserInputError, "distribution must be"):
            RobustSearchSpace(
                parameters=self.parameters,
                parameter_distributions=[a_dist, choice_dist],
                num_samples=4,
                # pyre-fixme[6]: For 4th param expected
                #  `Optional[List[ParameterConstraint]]` but got
                #  `List[OrderConstraint]`.
                parameter_constraints=self.constraints,
            )
        with self.assertRaisesRegex(UnsupportedError, "Mixing the distribution"):
            RobustSearchSpace(
                parameters=self.parameters,
                parameter_distributions=[mixed_dist],
                num_samples=4,
                environmental_variables=[env1],
                # pyre-fixme[6]: For 5th param expected
                #  `Optional[List[ParameterConstraint]]` but got
                #  `List[OrderConstraint]`.
                parameter_constraints=self.constraints,
            )
        # Test with environmental variables.
        rss = RobustSearchSpace(
            parameters=self.parameters,
            parameter_distributions=[env1_dist, env2_dist],
            num_samples=4,
            environmental_variables=[env1, env2],
            # pyre-fixme[6]: For 5th param expected
            #  `Optional[List[ParameterConstraint]]` but got `List[OrderConstraint]`.
            parameter_constraints=self.constraints,
        )
        self.assertEqual(rss.num_samples, 4)
        self.assertTrue(rss.is_robust)
        self.assertEqual(rss.parameter_constraints, self.constraints)
        self.assertEqual(
            rss.parameters,
            {
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "env1": env1,
                "env2": env2,
            },
        )
        self.assertEqual(rss.parameter_distributions, [env1_dist, env2_dist])
        self.assertEqual(rss._environmental_distributions, [env1_dist, env2_dist])
        self.assertEqual(rss._perturbation_distributions, [])
        self.assertFalse(rss.multiplicative)
        self.assertEqual(rss._distributional_parameters, {"env1", "env2"})
        self.assertEqual(rss._environmental_variables, {"env1": env1, "env2": env2})
        self.assertTrue(all(rss.is_environmental_variable(p) for p in ["env1", "env2"]))
        # Test having both types together.
        mul_a_dist = a_dist.clone()
        mul_a_dist.multiplicative = True
        rss = RobustSearchSpace(
            parameters=self.parameters,
            parameter_distributions=[mul_a_dist, env1_dist],
            num_samples=4,
            environmental_variables=[env1],
        )
        self.assertEqual(
            rss.parameters,
            {
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "env1": env1,
            },
        )
        self.assertEqual(rss.parameter_distributions, [mul_a_dist, env1_dist])
        self.assertEqual(rss._environmental_distributions, [env1_dist])
        self.assertEqual(rss._perturbation_distributions, [mul_a_dist])
        self.assertTrue(rss.multiplicative)
        self.assertEqual(rss._distributional_parameters, {"a", "env1"})
        self.assertEqual(rss._environmental_variables, {"env1": env1})
        # Test with input noise.
        rss = RobustSearchSpace(
            parameters=self.parameters,
            parameter_distributions=[a_dist, b_dist],
            num_samples=4,
            # pyre-fixme[6]: For 4th param expected
            #  `Optional[List[ParameterConstraint]]` but got `List[OrderConstraint]`.
            parameter_constraints=self.constraints,
        )
        self.assertTrue(rss.is_robust)
        self.assertEqual(rss.parameter_constraints, self.constraints)
        self.assertEqual(
            rss.parameters,
            {
                "a": self.a,
                "b": self.b,
                "c": self.c,
            },
        )
        self.assertEqual(rss.parameter_distributions, [a_dist, b_dist])
        self.assertEqual(rss._environmental_distributions, [])
        self.assertEqual(rss._perturbation_distributions, [a_dist, b_dist])
        self.assertFalse(rss.multiplicative)
        self.assertEqual(rss._distributional_parameters, {"a", "b"})
        self.assertEqual(rss._environmental_variables, {})
        # Tests with a multivariate distribution.
        rss = self.rss1
        self.assertEqual(rss.parameter_constraints, self.constraints)
        self.assertEqual(
            rss.parameters,
            {
                "a": self.a,
                "b": self.b,
                "c": self.c,
            },
        )
        self.assertEqual(rss.parameter_distributions, [self.ab_dist])
        self.assertEqual(rss._environmental_distributions, [])
        self.assertEqual(rss._perturbation_distributions, [self.ab_dist])
        self.assertEqual(rss._distributional_parameters, {"a", "b"})
        self.assertEqual(rss._environmental_variables, {})
        self.assertFalse(
            any(rss.is_environmental_variable(p) for p in rss.parameters.keys())
        )

    def test_update_parameter(self) -> None:
        rss = self.rss1
        with self.assertRaisesRegex(UnsupportedError, "update_parameter"):
            rss.update_parameter(self.a)

    def test_clone(self) -> None:
        rss_clone = self.rss1.clone()
        self.assertEqual(
            rss_clone._environmental_variables, self.rss1._environmental_variables
        )
        self.assertEqual(rss_clone.parameters, self.rss1.parameters)
        self.assertEqual(rss_clone.num_samples, self.rss1.num_samples)
        self.assertEqual(
            rss_clone.parameter_constraints, self.rss1.parameter_constraints
        )
        self.assertEqual(
            rss_clone.parameter_distributions, self.rss1.parameter_distributions
        )
        self.assertEqual(
            rss_clone._distributional_parameters, self.rss1._distributional_parameters
        )

    def test_repr(self) -> None:
        expected = (
            "RobustSearchSpace("
            "parameters=["
            "RangeParameter(name='a', parameter_type=FLOAT, range=[0.5, 5.5]), "
            "RangeParameter(name='b', parameter_type=INT, range=[2, 10]), "
            "ChoiceParameter(name='c', parameter_type=STRING, "
            "values=['foo', 'bar', 'baz'], is_ordered=False, sort_values=False)], "
            "parameter_distributions=["
            "ParameterDistribution(parameters=['a', 'b'], "
            "distribution_class=multivariate_normal, distribution_parameters={}, "
            "multiplicative=False)], "
            "num_samples=4, "
            "environmental_variables=[], "
            "parameter_constraints=[OrderConstraint(a <= b)])"
        )
        self.assertEqual(str(self.rss1), expected)

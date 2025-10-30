#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.fixed_to_tunable import FixedToTunable
from ax.core.observation import ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import OrderConstraint
from ax.core.search_space import SearchSpace
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from pandas.testing import assert_frame_equal
from pyre_extensions import assert_is_instance


class FixedToTunableTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Search space with fixed parameters that should be converted to tunable
        self.search_space_with_fixed = SearchSpace(
            parameters=[
                RangeParameter(
                    name="a", parameter_type=ParameterType.FLOAT, lower=1, upper=3
                ),
                FixedParameter(name="b", parameter_type=ParameterType.FLOAT, value=2.0),
                ChoiceParameter(
                    "c", parameter_type=ParameterType.STRING, values=["x", "y", "z"]
                ),
                FixedParameter(name="d", parameter_type=ParameterType.INT, value=5),
                FixedParameter(name="e", parameter_type=ParameterType.FLOAT, value=3.0),
            ]
        )

        # Joint search space with range parameters for the fixed parameters
        self.joint_search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name="a", parameter_type=ParameterType.FLOAT, lower=1, upper=3
                ),
                RangeParameter(
                    name="b", parameter_type=ParameterType.FLOAT, lower=1, upper=5
                ),
                ChoiceParameter(
                    "c", parameter_type=ParameterType.STRING, values=["x", "y", "z"]
                ),
                RangeParameter(
                    name="d", parameter_type=ParameterType.INT, lower=1, upper=10
                ),
                FixedParameter(name="e", parameter_type=ParameterType.FLOAT, value=4.0),
            ]
        )

        self.t = FixedToTunable(search_space=self.joint_search_space)

    def test_init(self) -> None:
        self.assertEqual(self.t.search_space, self.joint_search_space)

    def test_init_requires_search_space(self) -> None:
        with self.assertRaisesRegex(
            AssertionError, "FixedToTunable requires search space"
        ):
            FixedToTunable(search_space=None)

    def test_transform_search_space(self) -> None:
        # Test transforming fixed parameters to tunable ones
        transformed_ss = self.t.transform_search_space(self.search_space_with_fixed)

        # Parameter 'a' should remain unchanged (already a RangeParameter)
        self.assertEqual(
            transformed_ss.parameters["a"], self.search_space_with_fixed.parameters["a"]
        )

        # Parameter 'b' should be converted from FixedParameter to RangeParameter
        transformed_b = assert_is_instance(
            transformed_ss.parameters["b"], RangeParameter
        )
        self.assertEqual(transformed_b.lower, 1)
        self.assertEqual(transformed_b.upper, 5)
        self.assertEqual(transformed_b.parameter_type, ParameterType.FLOAT)

        # Parameter 'c' should remain unchanged (ChoiceParameter)
        self.assertEqual(
            transformed_ss.parameters["c"], self.search_space_with_fixed.parameters["c"]
        )

        # Parameter 'd' should be converted from FixedParameter to RangeParameter
        transformed_d = assert_is_instance(
            transformed_ss.parameters["d"], RangeParameter
        )
        self.assertEqual(transformed_d.lower, 1)
        self.assertEqual(transformed_d.upper, 10)
        self.assertEqual(transformed_d.parameter_type, ParameterType.INT)

        # Parameter 'e' should remain as FixedParameter (with the original value from
        # self.search_space_with_fixed) since it's fixed in the joint search space
        transformed_e = assert_is_instance(
            transformed_ss.parameters["e"], FixedParameter
        )
        self.assertEqual(transformed_e.value, 3.0)

    def test_transform_search_space_with_constraints(self) -> None:
        # Create search space with parameter constraints
        parameters = [
            RangeParameter(
                name="x", parameter_type=ParameterType.FLOAT, lower=1, upper=3
            ),
            RangeParameter(
                name="y", parameter_type=ParameterType.FLOAT, lower=1, upper=3
            ),
            FixedParameter(name="z", parameter_type=ParameterType.FLOAT, value=2.0),
        ]
        search_space_with_constraints = SearchSpace(
            parameters=parameters,
            parameter_constraints=[
                OrderConstraint(
                    lower_parameter=parameters[0], upper_parameter=parameters[1]
                )
            ],
        )

        # Joint space with range parameter for 'y'
        joint_space_with_y_range = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x", parameter_type=ParameterType.FLOAT, lower=1, upper=3
                ),
                RangeParameter(
                    name="y", parameter_type=ParameterType.FLOAT, lower=1, upper=5
                ),
                RangeParameter(
                    name="z", parameter_type=ParameterType.FLOAT, lower=1, upper=5
                ),
            ]
        )

        t = FixedToTunable(search_space=joint_space_with_y_range)
        transformed_ss = t.transform_search_space(search_space_with_constraints)

        # Verify parameter 'z' is transformed to RangeParameter
        transformed_z = assert_is_instance(
            transformed_ss.parameters["z"], RangeParameter
        )
        self.assertEqual(transformed_z.lower, 1)
        self.assertEqual(transformed_z.upper, 5)

        # Verify constraints are preserved
        self.assertEqual(len(transformed_ss.parameter_constraints), 1)
        self.assertIs(
            transformed_ss.parameter_constraints[0],
            search_space_with_constraints.parameter_constraints[0],
        )

    def test_transform_experiment_data(self) -> None:
        # Test that experiment data is returned unchanged
        parameterizations = [
            {"a": 1.0, "b": 2.0, "c": "x", "d": 5, "e": 3.0},
            {"a": 2.0, "b": 2.0, "c": "y", "d": 5, "e": 3.0},
            {"a": 3.0, "b": 2.0, "c": "z", "d": 5, "e": 3.0},
        ]
        experiment = get_experiment_with_observations(
            observations=[[1.0], [2.0], [3.0]],
            search_space=self.search_space_with_fixed,
            parameterizations=parameterizations,
        )
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )

        original_data = deepcopy(experiment_data)
        transformed_data = self.t.transform_experiment_data(experiment_data)

        # The transform should return the data unchanged
        self.assertIs(transformed_data, experiment_data)

        # Verify all data is identical
        assert_frame_equal(transformed_data.arm_data, original_data.arm_data)
        assert_frame_equal(
            transformed_data.observation_data, original_data.observation_data
        )

    def test_transform_observation_features(self) -> None:
        # Since FixedToTunable doesn't override transform_observation_features,
        # it should use the base class implementation which returns unchanged data
        observation_features = [
            ObservationFeatures(
                parameters={"a": 2.0, "b": 2.0, "c": "y", "d": 5, "e": 3.0}
            )
        ]

        # Test transform (should be unchanged)
        transformed_obs = self.t.transform_observation_features(
            deepcopy(observation_features)
        )
        self.assertEqual(transformed_obs, observation_features)

        # Test untransform (should also be unchanged)
        untransformed_obs = self.t.untransform_observation_features(
            deepcopy(observation_features)
        )
        self.assertEqual(untransformed_obs, observation_features)

    def test_clone_behavior(self) -> None:
        # Test that transformed parameters are properly cloned
        transformed_ss = self.t.transform_search_space(self.search_space_with_fixed)
        original_b_param = assert_is_instance(
            self.joint_search_space.parameters["b"], RangeParameter
        )
        transformed_b_param = assert_is_instance(
            transformed_ss.parameters["b"], RangeParameter
        )

        # They should have the same values but be different objects
        self.assertEqual(original_b_param.lower, transformed_b_param.lower)
        self.assertEqual(
            original_b_param.upper,
            transformed_b_param.upper,
        )
        self.assertEqual(
            original_b_param.parameter_type, transformed_b_param.parameter_type
        )
        self.assertIsNot(original_b_param, transformed_b_param)

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from copy import deepcopy

import numpy as np
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.torch import TorchAdapter
from ax.adapter.transforms.choice_encode import ChoiceToNumericChoice
from ax.adapter.transforms.log import Log
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from ax.utils.testing.mock import mock_botorch_optimize
from pandas.testing import assert_frame_equal, assert_series_equal
from pyre_extensions import assert_is_instance, none_throws


class LogTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Search space with both float and integer log-scale parameters
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x",
                    lower=1,
                    upper=3,
                    parameter_type=ParameterType.FLOAT,
                    log_scale=True,
                    digits=3,
                ),
                RangeParameter(
                    "y",
                    lower=1,
                    upper=10,
                    parameter_type=ParameterType.INT,
                    log_scale=True,
                ),
                RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
            ]
        )
        self.t = Log(search_space=self.search_space)

        self.search_space_with_target = SearchSpace(
            parameters=[
                RangeParameter(
                    "x",
                    lower=1,
                    upper=3,
                    parameter_type=ParameterType.FLOAT,
                    log_scale=True,
                    is_fidelity=True,
                    target_value=3,
                ),
                RangeParameter(
                    "y",
                    lower=1,
                    upper=10,
                    parameter_type=ParameterType.INT,
                    log_scale=True,
                    is_fidelity=True,
                    target_value=5,
                ),
            ]
        )

    def test_Init(self) -> None:
        # Test identification of both float and int log-scale parameters
        self.assertEqual(
            self.t.transform_parameters,
            {"x": ParameterType.FLOAT, "y": ParameterType.INT},
        )

    def test_TransformObservationFeatures(self) -> None:
        # Test with both float and integer log-scale parameters
        observation_features = [
            ObservationFeatures(parameters={"x": 2.2, "y": 5, "a": 2, "b": "c"})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [
                ObservationFeatures(
                    parameters={
                        "x": math.log10(2.2),
                        "y": math.log10(5),
                        "a": 2,
                        "b": "c",
                    }
                )
            ],
        )
        self.assertTrue(isinstance(obs_ft2[0].parameters["x"], float))
        self.assertTrue(isinstance(obs_ft2[0].parameters["y"], float))

        # Test untransformation - integer parameters should be rounded
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)
        self.assertTrue(isinstance(obs_ft2[0].parameters["y"], int))

    def test_TransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)

        # Test float log-scale parameter transformation
        param_x = assert_is_instance(ss2.parameters["x"], RangeParameter)
        self.assertEqual(param_x.lower, math.log10(1))
        self.assertEqual(param_x.upper, math.log10(3))
        self.assertIsNone(param_x.digits)

        # Test integer log-scale parameter transformation (converted to ChoiceParameter)
        param_y = assert_is_instance(ss2.parameters["y"], ChoiceParameter)
        self.assertEqual(param_y.parameter_type, ParameterType.FLOAT)
        self.assertTrue(param_y.is_ordered)
        expected_values = [math.log10(i) for i in range(1, 11)]
        self.assertEqual(param_y.values, expected_values)

        # Test non-log-scale parameter remains unchanged
        param_a = assert_is_instance(ss2.parameters["a"], RangeParameter)
        self.assertEqual(param_a.parameter_type, ParameterType.INT)
        self.assertEqual(param_a.lower, 1)
        self.assertEqual(param_a.upper, 2)

        # Test target values transformation
        t2 = Log(search_space=self.search_space_with_target)

        t2.transform_search_space(self.search_space_with_target)
        self.assertEqual(
            self.search_space_with_target.parameters["x"].target_value, math.log10(3)
        )
        # Test integer log-scale parameter with target
        param_y_target = assert_is_instance(
            self.search_space_with_target.parameters["y"], ChoiceParameter
        )
        self.assertEqual(param_y_target.target_value, math.log10(5))
        self.assertTrue(param_y_target.is_fidelity)

    def test_transform_experiment_data(self) -> None:
        # Test with both float and integer log-scale parameters
        parameterizations = [
            {"x": 1.0, "y": 2, "a": 1, "b": "a"},
            {"x": 1.5, "y": 5, "a": 2, "b": "b"},
            {"x": 1.7, "y": 8, "a": 3, "b": "c"},
        ]
        experiment = get_experiment_with_observations(
            observations=[[1.0], [2.0], [3.0]],
            search_space=self.search_space,
            parameterizations=parameterizations,
        )
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        transformed_data = self.t.transform_experiment_data(
            experiment_data=deepcopy(experiment_data)
        )

        # Check that both `x` and `y` have been log-transformed.
        assert_series_equal(
            transformed_data.arm_data["x"], np.log10(experiment_data.arm_data["x"])
        )
        assert_series_equal(
            transformed_data.arm_data["y"], np.log10(experiment_data.arm_data["y"])
        )

        # Check that other columns remain unchanged.
        assert_series_equal(
            transformed_data.arm_data["a"], experiment_data.arm_data["a"]
        )
        assert_series_equal(
            transformed_data.arm_data["b"], experiment_data.arm_data["b"]
        )

        # Check that observation data is unchanged.
        assert_frame_equal(
            transformed_data.observation_data, experiment_data.observation_data
        )

    def test_log_scale_choice_parameter(self) -> None:
        """Test log-scale ChoiceParameter support"""
        # Search space with log-scale ChoiceParameter
        search_space = SearchSpace(
            parameters=[
                ChoiceParameter(
                    "z",
                    parameter_type=ParameterType.FLOAT,
                    values=[1.0, 10.0, 100.0, 1000.0],
                    log_scale=True,
                ),
                ChoiceParameter(
                    "w",
                    parameter_type=ParameterType.INT,
                    values=[2, 4, 8, 16, 32],
                    log_scale=True,
                    is_fidelity=True,
                    target_value=32,
                    dependents={4: ["t"]},
                ),
                ChoiceParameter(
                    "t",
                    parameter_type=ParameterType.INT,
                    values=[1, 2, 3],
                ),
            ]
        )
        t = Log(search_space=search_space)

        # Test that log-scale choice parameters are identified
        self.assertEqual(
            t.transform_parameters,
            {"z": ParameterType.FLOAT, "w": ParameterType.INT},
        )
        self.assertEqual(
            t.original_values, {"z": [1.0, 10.0, 100.0, 1000.0], "w": [2, 4, 8, 16, 32]}
        )

        # Test observation features transformation
        observation_features = [ObservationFeatures(parameters={"z": 100.0, "w": 8})]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [
                ObservationFeatures(
                    parameters={
                        "z": math.log10(100.0),
                        "w": math.log10(8),
                    }
                )
            ],
        )

        # Test untransformation - should get exact match for the original values.
        obs_ft2 = t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)
        self.assertTrue(isinstance(obs_ft2[0].parameters["w"], int))

        # Test search space transformation
        ss2 = deepcopy(search_space)
        ss2 = t.transform_search_space(ss2)

        # Test float log-scale choice parameter transformation
        param_z = assert_is_instance(ss2.parameters["z"], ChoiceParameter)
        self.assertEqual(param_z.parameter_type, ParameterType.FLOAT)
        expected_values_z = [math.log10(v) for v in [1.0, 10.0, 100.0, 1000.0]]
        self.assertEqual(param_z.values, expected_values_z)
        self.assertFalse(param_z.log_scale)

        # Test int log-scale choice parameter transformation
        param_w = assert_is_instance(ss2.parameters["w"], ChoiceParameter)
        self.assertEqual(param_w.parameter_type, ParameterType.FLOAT)
        expected_values_w = [math.log10(v) for v in [2, 4, 8, 16, 32]]
        self.assertEqual(param_w.values, expected_values_w)
        self.assertFalse(param_w.log_scale)
        self.assertTrue(param_w.is_fidelity)
        self.assertEqual(param_w.target_value, math.log10(32))
        self.assertEqual(param_w.dependents, {math.log10(4): ["t"]})

        # Verify that `t` is not transformed.
        self.assertEqual(ss2.parameters["t"], search_space.parameters["t"])

    @mock_botorch_optimize
    def test_log_scale_choice_with_adapter(self) -> None:
        search_space = SearchSpace(
            parameters=[
                ChoiceParameter(
                    "z",
                    parameter_type=ParameterType.FLOAT,
                    values=[1.0, 10.0, 100.0, 1000.0],
                ),
                ChoiceParameter(
                    "w",
                    parameter_type=ParameterType.INT,
                    values=[2, 4, 8, 16, 32],
                ),
            ]
        )
        experiment = get_experiment_with_observations(
            observations=[[1.0], [2.0], [3.0]], search_space=search_space
        )
        generator = BoTorchGenerator()
        adapter = TorchAdapter(
            experiment=experiment,
            generator=generator,
            transforms=[ChoiceToNumericChoice, Log],
        )
        gr = adapter.gen(n=1)
        self.assertEqual(len(gr.arms), 1)
        # Check the SSD to see if the parameters are log-transformed correctly.
        ssd = none_throws(generator.surrogate._last_search_space_digest)
        self.assertEqual(ssd.feature_names, ["z", "w"])
        self.assertEqual(
            ssd.discrete_choices,
            {
                0: [
                    math.log10(1.0),
                    math.log10(10.0),
                    math.log10(100.0),
                    math.log10(1000.0),
                ],
                1: [
                    math.log10(2),
                    math.log10(4),
                    math.log10(8),
                    math.log10(16),
                    math.log10(32),
                ],
            },
        )

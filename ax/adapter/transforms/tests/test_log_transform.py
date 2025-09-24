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
from ax.adapter.transforms.log import Log
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment_with_observations,
    get_robust_search_space,
)
from pandas.testing import assert_frame_equal, assert_series_equal
from pyre_extensions import assert_is_instance


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

    def test_w_parameter_distributions(self) -> None:
        rss = get_robust_search_space(lb=1.0, use_discrete=True)
        # pyre-fixme[16]: `Parameter` has no attribute `set_log_scale`.
        rss.parameters["y"].set_log_scale(True)
        # Transform a non-distributional parameter.
        t = Log(search_space=rss)
        t.transform_search_space(rss)
        # pyre-fixme[16]: Optional type has no attribute `log_scale`.
        self.assertFalse(rss.parameters.get("y").log_scale)
        # Error with distributional parameter.
        rss.parameters["x"].set_log_scale(True)
        t = Log(search_space=rss)
        with self.assertRaisesRegex(UnsupportedError, "transform is not supported"):
            t.transform_search_space(rss)

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

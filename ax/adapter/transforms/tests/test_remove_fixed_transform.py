#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.remove_fixed import RemoveFixed
from ax.core.observation import ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment_with_observations,
    get_robust_search_space,
)
from pandas.testing import assert_frame_equal, assert_series_equal


class RemoveFixedTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "a", lower=1, upper=3, parameter_type=ParameterType.FLOAT
                ),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
                FixedParameter("c", parameter_type=ParameterType.STRING, value="a"),
            ]
        )
        self.t = RemoveFixed(search_space=self.search_space)

    def test_Init(self) -> None:
        self.assertEqual(list(self.t.fixed_parameters.keys()), ["c"])

    def test_TransformObservationFeatures(self) -> None:
        observation_features = [
            ObservationFeatures(parameters={"a": 2.2, "b": "b", "c": "a"})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2, [ObservationFeatures(parameters={"a": 2.2, "b": "b"})]
        )
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

        observation_features = [
            ObservationFeatures(parameters={"a": 2.2, "b": "b", "c": "a"})
        ]
        observation_features_different = [
            ObservationFeatures(parameters={"a": 2.2, "b": "b", "c": "b"})
        ]
        # Fixed parameter is out of design. It will still get removed.
        t_obs = self.t.transform_observation_features(observation_features)
        t_obs_different = self.t.transform_observation_features(
            observation_features_different
        )
        self.assertEqual(t_obs, t_obs_different)

    def test_TransformSearchSpace(self) -> None:
        ss2 = self.search_space.clone()
        ss2 = self.t.transform_search_space(ss2)
        self.assertEqual(ss2.parameters.get("c"), None)

    def test_w_parameter_distributions(self) -> None:
        rss = get_robust_search_space()
        rss.add_parameter(
            FixedParameter("d", parameter_type=ParameterType.STRING, value="a"),
        )
        # Transform a non-distributional parameter.
        t = RemoveFixed(
            search_space=rss,
            observations=[],
        )
        rss = t.transform_search_space(rss)
        # Make sure that the return value is still a RobustSearchSpace.
        self.assertIsInstance(rss, RobustSearchSpace)
        self.assertEqual(len(rss.parameters.keys()), 4)
        # pyre-fixme[16]: `SearchSpace` has no attribute `parameter_distributions`.
        self.assertEqual(len(rss.parameter_distributions), 2)
        self.assertNotIn("d", rss.parameters)
        # Test with environmental variables.
        all_params = list(rss.parameters.values())
        rss = RobustSearchSpace(
            parameters=all_params[2:],
            parameter_distributions=rss.parameter_distributions,
            # pyre-fixme[16]: `SearchSpace` has no attribute `num_samples`.
            num_samples=rss.num_samples,
            environmental_variables=all_params[:2],
        )
        rss.add_parameter(
            FixedParameter("d", parameter_type=ParameterType.STRING, value="a"),
        )
        t = RemoveFixed(
            search_space=rss,
            observations=[],
        )
        rss = t.transform_search_space(rss)
        self.assertIsInstance(rss, RobustSearchSpace)
        self.assertEqual(len(rss.parameters.keys()), 4)
        self.assertEqual(len(rss.parameter_distributions), 2)
        # pyre-fixme[16]: `SearchSpace` has no attribute `_environmental_variables`.
        self.assertEqual(len(rss._environmental_variables), 2)
        self.assertNotIn("d", rss.parameters)

    def test_transform_experiment_data(self) -> None:
        parameterizations = [
            {"a": 1, "b": "a", "c": "a"},
            {"a": 2, "b": "b", "c": "a"},
            {"a": 3, "b": "c", "c": "a"},
        ]
        experiment = get_experiment_with_observations(
            observations=[[1.0], [2.0], [3.0]],
            search_space=self.search_space,
            parameterizations=parameterizations,
        )
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        copy_experiment_data = deepcopy(experiment_data)
        transformed_data = self.t.transform_experiment_data(
            experiment_data=copy_experiment_data
        )

        self.assertIn("c", experiment_data.arm_data)
        # Check that `c` has been removed.
        self.assertNotIn("c", transformed_data.arm_data)

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
        self.assertIs(
            transformed_data.observation_data, copy_experiment_data.observation_data
        )

        # Test with no fixed features.
        search_space = self.t.transform_search_space(search_space=self.search_space)
        t = RemoveFixed(search_space=search_space)
        copy_transformed_data = deepcopy(transformed_data)
        transformed_data_2 = t.transform_experiment_data(
            experiment_data=copy_transformed_data
        )
        self.assertEqual(transformed_data_2, transformed_data)
        self.assertIs(
            transformed_data_2.observation_data, copy_transformed_data.observation_data
        )
        self.assertIsNot(transformed_data_2.arm_data, copy_transformed_data.arm_data)
        assert_frame_equal(transformed_data_2.arm_data, copy_transformed_data.arm_data)

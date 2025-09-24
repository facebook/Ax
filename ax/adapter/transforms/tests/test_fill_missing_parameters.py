#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.fill_missing_parameters import FillMissingParameters
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from pandas.testing import assert_frame_equal


class FillMissingParametersTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.search_space_backfill_values = {
            "x": 0.0,
            "y": 1.0,
        }
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=10.0,
                    backfill_value=self.search_space_backfill_values["x"],
                ),
                RangeParameter(
                    name="y",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=10.0,
                    backfill_value=self.search_space_backfill_values["y"],
                ),
            ]
        )
        self.config_values = {"x": 2.0, "y": 3.0}
        self.config = {"fill_values": self.config_values}

    def test_init_with_search_space(self) -> None:
        t = FillMissingParameters(search_space=self.search_space)
        self.assertEqual(t._fill_values, self.search_space_backfill_values)

    def test_init_with_deprecated_config(self) -> None:
        with self.assertLogs(
            "ax.adapter.transforms.fill_missing_parameters", level="ERROR"
        ) as lg:
            t = FillMissingParameters(config=self.config)
        self.assertIn("deprecated", lg.output[0])
        self.assertEqual(t._fill_values, self.config_values)

    def test_init_with_both_config_and_search_space(self) -> None:
        t = FillMissingParameters(search_space=self.search_space, config=self.config)
        # Search space values should override config values
        self.assertEqual(t._fill_values, self.search_space_backfill_values)

    def test_transform_observation_features(self) -> None:
        observation_features = [
            ObservationFeatures(parameters={"x": None}),
            ObservationFeatures(parameters={"x": 2.0}),
            ObservationFeatures(parameters={"x": 0.0, "y": 2.0}),
        ]
        t = FillMissingParameters(search_space=self.search_space)
        expected = [
            ObservationFeatures(parameters={"x": 0.0, "y": 1.0}),
            ObservationFeatures(parameters={"x": 2.0, "y": 1.0}),
            ObservationFeatures(parameters={"x": 0.0, "y": 2.0}),
        ]
        result = t.transform_observation_features(deepcopy(observation_features))
        self.assertEqual(result, expected)

    def test_transform_observation_features_no_fill_values(self) -> None:
        search_space = self.search_space.clone()
        # Remove backfill values from search space
        for parameter in search_space._parameters.values():
            parameter._backfill_value = None
        observation_features = [
            ObservationFeatures(parameters={"x": None}),
            ObservationFeatures(parameters={"x": 0.0}),
        ]
        t = FillMissingParameters(search_space=search_space)
        result = t.transform_observation_features(deepcopy(observation_features))
        # No changes should be made
        self.assertEqual(result, observation_features)

    def test_transform_experiment_data(self) -> None:
        search_space = self.search_space.clone()
        search_space.add_parameters(
            [
                RangeParameter(
                    name="z",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=1.0,
                    backfill_value=0.0,
                )
            ]
        )
        parameterizations = [
            {"x": 0.0},
            {"x": 1.0, "y": 0.0},
            {"x": None, "y": None},
        ]
        experiment = get_experiment_with_observations(
            observations=[[1.0], [2.0], [3.0]],
            parameterizations=parameterizations,
        )
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        # Check that arm_data has NaNs as expected.
        self.assertEqual(experiment_data.arm_data["x"].isna().sum(), 1)
        self.assertEqual(experiment_data.arm_data["y"].isna().sum(), 2)

        t = FillMissingParameters(search_space=search_space)
        transformed_data = t.transform_experiment_data(
            experiment_data=deepcopy(experiment_data)
        )
        self.assertEqual(transformed_data.arm_data["x"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(transformed_data.arm_data["y"].tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(transformed_data.arm_data["z"].tolist(), [0.0, 0.0, 0.0])
        assert_frame_equal(
            transformed_data.observation_data, experiment_data.observation_data
        )

    def test_transform_experiment_data_no_fill_values(self) -> None:
        search_space = self.search_space.clone()
        # Remove backfill values from search space
        for parameter in search_space._parameters.values():
            parameter._backfill_value = None
        parameterizations = [
            {"x": 0.0},
            {"x": None},
        ]
        experiment = get_experiment_with_observations(
            observations=[[1.0], [2.0]],
            parameterizations=parameterizations,
        )
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )

        t = FillMissingParameters(search_space=search_space)
        transformed_data = t.transform_experiment_data(
            experiment_data=deepcopy(experiment_data)
        )
        # No changes should be made
        assert_frame_equal(transformed_data.arm_data, experiment_data.arm_data)
        assert_frame_equal(
            transformed_data.observation_data, experiment_data.observation_data
        )

    def test_deprecated_config_behavior_still_works(self) -> None:
        observation_features = [
            ObservationFeatures(parameters={"x": None}),
            ObservationFeatures(parameters={"x": 0.0}),
        ]
        t = FillMissingParameters(config=self.config)
        expected = [
            ObservationFeatures(parameters={"x": 2.0, "y": 3.0}),
            ObservationFeatures(parameters={"x": 0.0, "y": 3.0}),
        ]
        result = t.transform_observation_features(deepcopy(observation_features))
        self.assertEqual(result, expected)

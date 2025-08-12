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
from ax.exceptions.core import UnsupportedError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from pandas.testing import assert_frame_equal


class FillMissingParametersTransformTest(TestCase):
    def test_Init(self) -> None:
        config = {"fill_values": {"x": 2.0, "y": 1.0}}
        t = FillMissingParameters(config=config)  # pyre-ignore[6]
        self.assertEqual(t.fill_values, config["fill_values"])
        self.assertTrue(t.fill_None)
        config = {"fill_values": {"x": 2.0, "y": 1.0}, "fill_None": False}
        t = FillMissingParameters(config=config)  # pyre-ignore[6]
        self.assertFalse(t.fill_None)

    def test_TransformObservationFeatures(self) -> None:
        observation_features = [
            ObservationFeatures(parameters={"x": None}),
            ObservationFeatures(parameters={"x": 0.0}),
            ObservationFeatures(parameters={"x": 0.0, "y": 2.0}),
        ]
        config = {"fill_values": {"x": 2.0, "y": 1.0}}
        t = FillMissingParameters(config=config)  # pyre-ignore[6]
        true_1 = [
            ObservationFeatures(parameters={"x": 2.0, "y": 1.0}),
            ObservationFeatures(parameters={"x": 0.0, "y": 1.0}),
            ObservationFeatures(parameters={"x": 0.0, "y": 2.0}),
        ]
        obs_ft1 = t.transform_observation_features(deepcopy(observation_features))
        self.assertEqual(obs_ft1, true_1)
        config["fill_None"] = False  # pyre-ignore[6]
        t = FillMissingParameters(config=config)  # pyre-ignore[6]
        true_2 = [
            ObservationFeatures(parameters={"x": None, "y": 1.0}),
            ObservationFeatures(parameters={"x": 0.0, "y": 1.0}),
            ObservationFeatures(parameters={"x": 0.0, "y": 2.0}),
        ]
        obs_ft2 = t.transform_observation_features(deepcopy(observation_features))
        self.assertEqual(obs_ft2, true_2)
        # No transformation if no fill values given
        t = FillMissingParameters(config={})
        obs_ft3 = t.transform_observation_features(deepcopy(observation_features))
        self.assertEqual(obs_ft3, observation_features)

    def test_transform_experiment_data(self) -> None:
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

        fill_x = 2.0
        fill_y = 1.0
        fill_z = 0.0
        # Transform and see that NaNs are filled.
        t = FillMissingParameters(
            config={"fill_values": {"x": fill_x, "y": fill_y, "z": fill_z}}
        )
        transformed_data = t.transform_experiment_data(
            experiment_data=deepcopy(experiment_data)
        )
        self.assertEqual(transformed_data.arm_data["x"].tolist(), [0.0, 1.0, fill_x])
        self.assertEqual(transformed_data.arm_data["y"].tolist(), [fill_y, 0.0, fill_y])
        self.assertEqual(
            transformed_data.arm_data["z"].tolist(), [fill_z, fill_z, fill_z]
        )
        assert_frame_equal(
            transformed_data.observation_data, experiment_data.observation_data
        )

        # Nothing happens if no fill values are given.
        t = FillMissingParameters(config={})
        transformed_data = t.transform_experiment_data(
            experiment_data=deepcopy(experiment_data)
        )
        self.assertEqual(transformed_data, experiment_data)

        # Check for error if fill_None is False.
        t = FillMissingParameters(
            config={"fill_values": {"x": 2.0, "y": 1.0}, "fill_None": False}
        )
        with self.assertRaisesRegex(UnsupportedError, "ExperimentData"):
            t.transform_experiment_data(experiment_data=experiment_data)

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.modelbridge.transforms.fill_missing_parameters import FillMissingParameters
from ax.utils.common.testutils import TestCase


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

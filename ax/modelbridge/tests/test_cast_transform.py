#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.cast import Cast
from ax.utils.common.testutils import TestCase


class CastTransformTest(TestCase):
    def setUp(self):
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "a", lower=1.0, upper=5.0, parameter_type=ParameterType.FLOAT
                ),
                RangeParameter(
                    "b",
                    lower=1.0,
                    upper=5.0,
                    digits=2,
                    parameter_type=ParameterType.FLOAT,
                ),
                ChoiceParameter(
                    "c", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
                FixedParameter(name="d", parameter_type=ParameterType.INT, value=2),
            ],
            parameter_constraints=[],
        )
        self.t = Cast(search_space=self.search_space)

    def testTransformObservationFeatures(self):
        # Verify running the transform on already-casted features does nothing
        observation_features = [
            ObservationFeatures(parameters={"a": 1.2345, "b": 2.34, "c": "a", "d": 2})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

    def testUntransformObservationFeatures(self):
        # Verify running the transform on uncasted values properly converts them
        # (e.g. typing, rounding)
        observation_features = [
            ObservationFeatures(parameters={"a": 1, "b": 2.3466789, "c": "a", "d": 2.0})
        ]
        observation_features = self.t.untransform_observation_features(
            observation_features
        )
        self.assertEqual(
            observation_features,
            [ObservationFeatures(parameters={"a": 1.0, "b": 2.35, "c": "a", "d": 2})],
        )

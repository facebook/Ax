#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.out_of_design import OutOfDesign
from ax.utils.common.testutils import TestCase


class OutOfDesignTransformTest(TestCase):
    def setUp(self):
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
        self.t = OutOfDesign(
            search_space=self.search_space,
            observation_features=None,
            observation_data=None,
        )

    def testTransformObservationFeatures(self):
        # Don't modify points without None.
        observation_features = [
            ObservationFeatures(parameters={"a": 2.2, "b": "b", "c": "a"})
        ]
        obs_ft = deepcopy(observation_features)
        obs_ft = self.t.transform_observation_features(obs_ft)
        self.assertEqual(obs_ft, observation_features)

        # Strip params from points with any Nones.
        observation_features = [
            ObservationFeatures(parameters={"a": 2.2, "b": "b", "c": None})
        ]
        obs_ft = deepcopy(observation_features)
        obs_ft = self.t.transform_observation_features(obs_ft)
        self.assertEqual(obs_ft, [ObservationFeatures(parameters={})])

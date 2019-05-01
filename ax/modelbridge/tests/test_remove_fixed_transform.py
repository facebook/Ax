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
from ax.modelbridge.transforms.remove_fixed import RemoveFixed
from ax.utils.common.testutils import TestCase


class RemoveFixedTransformTest(TestCase):
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
        self.t = RemoveFixed(
            search_space=self.search_space,
            observation_features=None,
            observation_data=None,
        )

    def testInit(self):
        self.assertEqual(list(self.t.fixed_parameters.keys()), ["c"])

    def testTransformObservationFeatures(self):
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

    def testTransformSearchSpace(self):
        ss2 = self.search_space.clone()
        ss2 = self.t.transform_search_space(ss2)
        self.assertEqual(ss2.parameters.get("c"), None)

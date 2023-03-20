#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.search_space_to_float import SearchSpaceToFloat
from ax.utils.common.testutils import TestCase


class SearchSpaceToFloatTest(TestCase):
    def setUp(self) -> None:
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "a", lower=1, upper=3, parameter_type=ParameterType.FLOAT
                ),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
            ]
        )
        self.observation_features = [
            ObservationFeatures(parameters={"a": 2, "b": "a"}),
            ObservationFeatures(parameters={"a": 3, "b": "b"}),
            ObservationFeatures(parameters={"a": 3, "b": "c"}),
        ]
        self.transformed_features = [
            ObservationFeatures(parameters={"HASH_PARAM": 805305186152.0}),
            ObservationFeatures(parameters={"HASH_PARAM": 800097055771.0}),
            ObservationFeatures(parameters={"HASH_PARAM": 1551602558.0}),
        ]
        self.t = SearchSpaceToFloat()

    def testTransformSearchSpace(self) -> None:
        ss2 = self.search_space.clone()
        ss2 = self.t.transform_search_space(ss2)
        self.assertEqual(len(ss2.parameters), 1)
        expected_parameter = RangeParameter(
            name="HASH_PARAM",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1e12,
        )
        self.assertEqual(ss2.parameters.get("HASH_PARAM"), expected_parameter)

    def testTransformObservationFeatures(self) -> None:
        obs_ft2 = deepcopy(self.observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, self.transformed_features)

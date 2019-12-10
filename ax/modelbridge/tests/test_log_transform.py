#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.log import Log
from ax.utils.common.testutils import TestCase


class LogTransformTest(TestCase):
    def setUp(self):
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x",
                    lower=1,
                    upper=3,
                    parameter_type=ParameterType.FLOAT,
                    log_scale=True,
                ),
                RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
            ]
        )
        self.t = Log(
            search_space=self.search_space,
            observation_features=None,
            observation_data=None,
        )
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
                )
            ]
        )

    def testInit(self):
        self.assertEqual(self.t.transform_parameters, {"x"})

    def testTransformObservationFeatures(self):
        observation_features = [
            ObservationFeatures(parameters={"x": 2.2, "a": 2, "b": "c"})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [ObservationFeatures(parameters={"x": math.log10(2.2), "a": 2, "b": "c"})],
        )
        self.assertTrue(isinstance(obs_ft2[0].parameters["x"], float))
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

    def testTransformSearchSpace(self):
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        self.assertEqual(ss2.parameters["x"].lower, math.log10(1))
        self.assertEqual(ss2.parameters["x"].upper, math.log10(3))
        t2 = Log(
            search_space=self.search_space_with_target,
            observation_features=None,
            observation_data=None,
        )
        t2.transform_search_space(self.search_space_with_target)
        self.assertEqual(
            self.search_space_with_target.parameters["x"].target_value, math.log10(3)
        )

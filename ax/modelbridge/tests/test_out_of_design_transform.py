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
        self.in_design_features = [
            ObservationFeatures(parameters={"a": 2.2, "b": "b", "c": "a"})
        ]
        self.out_of_design_features_1 = [
            ObservationFeatures(parameters={"a": 2.2, "b": "b", "c": None})
        ]
        self.out_of_design_features_2 = [ObservationFeatures(parameters={})]

    def testTransformOutOfDesign(self):
        t_out_1 = OutOfDesign(
            search_space=self.search_space,
            observation_features=self.out_of_design_features_1,
            observation_data=None,
        )
        obs_ft = deepcopy(self.out_of_design_features_1)
        obs_ft = t_out_1.transform_observation_features(self.out_of_design_features_1)
        self.assertEqual(obs_ft, [ObservationFeatures(parameters={})])

        t_out_2 = OutOfDesign(
            search_space=self.search_space,
            observation_features=self.out_of_design_features_2,
            observation_data=None,
        )
        obs_ft = deepcopy(self.out_of_design_features_2)
        obs_ft = t_out_2.transform_observation_features(self.out_of_design_features_2)
        self.assertEqual(obs_ft, [ObservationFeatures(parameters={})])

    def testTransformInDesign(self):
        # Don't modify in design points.
        t_in = OutOfDesign(
            search_space=self.search_space,
            observation_features=self.in_design_features,
            observation_data=None,
        )
        obs_ft = deepcopy(self.in_design_features)
        obs_ft = t_in.transform_observation_features(obs_ft)
        self.assertEqual(obs_ft, self.in_design_features)

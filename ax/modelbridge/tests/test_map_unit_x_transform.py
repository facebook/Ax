#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.map_unit_x import MapUnitX
from ax.utils.common.testutils import TestCase


class MapUnitXTransformTest(TestCase):
    def setUp(self):
        self.target_lb = MapUnitX.target_lb
        self.target_range = MapUnitX.target_range
        self.target_ub = self.target_lb + self.target_range
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x", lower=1, upper=3, parameter_type=ParameterType.FLOAT
                ),
                RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
            ],
        )
        self.observation_features = [
            ObservationFeatures(
                parameters={"x": 2, "a": 2, "b": "b", "step_1": 0.0, "step_2": 0.0}
            ),
            ObservationFeatures(
                parameters={"x": 2, "a": 2, "b": "b", "step_1": 10.0, "step_2": 20.0}
            ),
            ObservationFeatures(parameters={"x": 2, "a": 2, "b": "b", "step_2": 15.0}),
            ObservationFeatures(
                parameters={"x": 2, "a": 2, "b": "b", "step_1": 2.0, "step_2": 12.0}
            ),
            ObservationFeatures(parameters={"x": 2, "a": 2, "b": "b", "step_1": 3.0}),
            ObservationFeatures(
                parameters={"x": 2, "a": 2, "b": "b", "step_1": 7.0, "step_2": 3.0}
            ),
        ]
        self.t = MapUnitX(
            search_space=self.search_space,
            observation_features=self.observation_features,
            observation_data=[],
        )

    def testInit(self):
        self.assertEqual(self.t.bounds, {"step_1": (0.0, 10.0), "step_2": (0.0, 20.0)})

    def testTransformObservationFeatures(self):
        obs_ft2 = deepcopy(self.observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        expected = [
            ObservationFeatures(
                parameters=self._construct_expected_param_dict(obsf, self.t)
            )
            for obsf in self.observation_features
        ]
        for obsf, expected_obsf in zip(obs_ft2, expected):
            for step in ("step_1", "step_2"):
                if step in expected_obsf.parameters:
                    self.assertAlmostEqual(
                        obsf.parameters[step], expected_obsf.parameters[step]
                    )

        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        for obsf, expected_obsf in zip(obs_ft2, self.observation_features):
            for step in ("step_1", "step_2"):
                if step in expected_obsf.parameters:
                    self.assertAlmostEqual(
                        obsf.parameters[step], expected_obsf.parameters[step]
                    )

    def testTransformSearchSpace(self):
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        self.assertEqual(ss2.parameters, self.search_space.parameters)

    def _construct_expected_param_dict(
        self, obsf: ObservationFeatures, transform: MapUnitX
    ):
        result = {"x": 2, "a": 2, "b": "b"}
        for step in ("step_1", "step_2"):
            if step in obsf.parameters:
                scale_fac = self.target_range / (
                    transform.bounds[step][1] - transform.bounds[step][0]
                )
                result[step] = self.target_lb + obsf.parameters[step] * scale_fac
        return result

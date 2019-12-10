#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.unit_x import UnitX
from ax.utils.common.testutils import TestCase


class UnitXTransformTest(TestCase):
    def setUp(self):
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x", lower=1, upper=3, parameter_type=ParameterType.FLOAT
                ),
                RangeParameter(
                    "y", lower=1, upper=2, parameter_type=ParameterType.FLOAT
                ),
                RangeParameter(
                    "z",
                    lower=1,
                    upper=2,
                    parameter_type=ParameterType.FLOAT,
                    log_scale=True,
                ),
                RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
            ],
            parameter_constraints=[
                ParameterConstraint(constraint_dict={"x": -0.5, "y": 1}, bound=0.5),
                ParameterConstraint(constraint_dict={"x": -0.5, "a": 1}, bound=0.5),
            ],
        )
        self.t = UnitX(
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
                    is_fidelity=True,
                    target_value=3,
                )
            ]
        )

    def testInit(self):
        self.assertEqual(self.t.bounds, {"x": (1.0, 3.0), "y": (1.0, 2.0)})

    def testTransformObservationFeatures(self):
        observation_features = [
            ObservationFeatures(parameters={"x": 2, "y": 2, "z": 2, "a": 2, "b": "b"})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [
                ObservationFeatures(
                    parameters={"x": 0.5, "y": 1.0, "z": 2, "a": 2, "b": "b"}
                )
            ],
        )
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)
        # Test transform partial observation
        obs_ft3 = [ObservationFeatures(parameters={"x": 3.0, "z": 2})]
        obs_ft3 = self.t.transform_observation_features(obs_ft3)
        self.assertEqual(obs_ft3[0], ObservationFeatures(parameters={"x": 1.0, "z": 2}))
        obs_ft5 = self.t.transform_observation_features([ObservationFeatures({})])
        self.assertEqual(obs_ft5[0], ObservationFeatures({}))

    def testTransformSearchSpace(self):
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)

        # Parameters transformed
        true_bounds = {
            "x": (0.0, 1.0),
            "y": (0.0, 1.0),
            "z": (1.0, 2.0),
            "a": (1.0, 2.0),
        }
        for p_name, (l, u) in true_bounds.items():
            self.assertEqual(ss2.parameters[p_name].lower, l)
            self.assertEqual(ss2.parameters[p_name].upper, u)
        self.assertEqual(ss2.parameters["b"].values, ["a", "b", "c"])
        self.assertEqual(len(ss2.parameters), 5)
        # Constraints transformed
        self.assertEqual(
            ss2.parameter_constraints[0].constraint_dict, {"x": -1.0, "y": 1.0}
        )
        self.assertEqual(ss2.parameter_constraints[0].bound, 0.0)
        self.assertEqual(
            ss2.parameter_constraints[1].constraint_dict, {"x": -1.0, "a": 1.0}
        )
        self.assertEqual(ss2.parameter_constraints[1].bound, 1.0)

        # Test transform of target value
        t = UnitX(
            search_space=self.search_space_with_target,
            observation_features=None,
            observation_data=None,
        )
        t.transform_search_space(self.search_space_with_target)
        self.assertEqual(
            self.search_space_with_target.parameters["x"].target_value, 1.0
        )

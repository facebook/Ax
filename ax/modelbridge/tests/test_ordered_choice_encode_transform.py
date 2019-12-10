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
from ax.modelbridge.transforms.ordered_choice_encode import OrderedChoiceEncode
from ax.utils.common.testutils import TestCase


class OrderedChoiceEncodeTransformTest(TestCase):
    def setUp(self):
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x", lower=1, upper=3, parameter_type=ParameterType.FLOAT
                ),
                RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
                ChoiceParameter(
                    "b",
                    parameter_type=ParameterType.FLOAT,
                    values=[1.0, 10.0, 100.0],
                    is_ordered=True,
                ),
                ChoiceParameter(
                    "c",
                    parameter_type=ParameterType.FLOAT,
                    values=[10.0, 100.0, 1000.0],
                    is_ordered=True,
                ),
                ChoiceParameter(
                    "d", parameter_type=ParameterType.STRING, values=["r", "q", "z"]
                ),
            ],
            parameter_constraints=[
                ParameterConstraint(constraint_dict={"x": -0.5, "a": 1}, bound=0.5)
            ],
        )
        self.t = OrderedChoiceEncode(
            search_space=self.search_space,
            observation_features=None,
            observation_data=None,
        )

    def testInit(self):
        self.assertEqual(list(self.t.encoded_parameters.keys()), ["b", "c"])

    def testTransformObservationFeatures(self):
        observation_features = [
            ObservationFeatures(
                parameters={"x": 2.2, "a": 2, "b": 10.0, "c": 100.0, "d": "r"}
            )
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [
                ObservationFeatures(
                    parameters={
                        "x": 2.2,
                        "a": 2,
                        # Index originally; int in RangeParameter.
                        "b": 1,
                        # Index originally; int in RangeParameter.
                        "c": 1,
                        "d": "r",
                    }
                )
            ],
        )
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)
        # Test transform on partial features
        obs_ft3 = [ObservationFeatures(parameters={"x": 2.2, "b": 10.0})]
        obs_ft3 = self.t.transform_observation_features(obs_ft3)
        self.assertEqual(obs_ft3[0], ObservationFeatures(parameters={"x": 2.2, "b": 1}))
        obs_ft5 = self.t.transform_observation_features([ObservationFeatures({})])
        self.assertEqual(obs_ft5[0], ObservationFeatures({}))

    def testTransformSearchSpace(self):
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)

        # Parameter type fixed.
        self.assertEqual(ss2.parameters["x"].parameter_type, ParameterType.FLOAT)
        self.assertEqual(ss2.parameters["a"].parameter_type, ParameterType.INT)
        self.assertEqual(ss2.parameters["b"].parameter_type, ParameterType.INT)
        self.assertEqual(ss2.parameters["c"].parameter_type, ParameterType.INT)
        self.assertEqual(ss2.parameters["d"].parameter_type, ParameterType.STRING)

        self.assertEqual(ss2.parameters["b"].lower, 0)
        self.assertEqual(ss2.parameters["b"].upper, 2)
        self.assertEqual(ss2.parameters["c"].lower, 0)
        self.assertEqual(ss2.parameters["c"].upper, 2)

        # Ensure we error if we try to transform a fidelity parameter
        ss3 = SearchSpace(
            parameters=[
                ChoiceParameter(
                    "b",
                    parameter_type=ParameterType.FLOAT,
                    values=[1.0, 10.0, 100.0],
                    is_ordered=True,
                    is_fidelity=True,
                    target_value=100.0,
                )
            ]
        )
        t = OrderedChoiceEncode(
            search_space=ss3, observation_features=None, observation_data=None
        )
        with self.assertRaises(ValueError):
            t.transform_search_space(ss3)

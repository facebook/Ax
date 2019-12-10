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
from ax.modelbridge.transforms.one_hot import OH_PARAM_INFIX, OneHot
from ax.utils.common.testutils import TestCase


class OneHotTransformTest(TestCase):
    def setUp(self):
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x", lower=1, upper=3, parameter_type=ParameterType.FLOAT
                ),
                RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
                ChoiceParameter(
                    "c", parameter_type=ParameterType.BOOL, values=[True, False]
                ),
                ChoiceParameter(
                    "d",
                    parameter_type=ParameterType.FLOAT,
                    values=[1.0, 10.0, 100.0],
                    is_ordered=True,
                ),
            ],
            parameter_constraints=[
                ParameterConstraint(constraint_dict={"x": -0.5, "a": 1}, bound=0.5)
            ],
        )
        self.t = OneHot(
            search_space=self.search_space,
            observation_features=None,
            observation_data=None,
        )
        self.t2 = OneHot(
            search_space=self.search_space,
            observation_features=None,
            observation_data=None,
            config={"rounding": "randomized"},
        )

        self.transformed_features = ObservationFeatures(
            parameters={
                "x": 2.2,
                "a": 2,
                "b" + OH_PARAM_INFIX + "_0": 0,
                "b" + OH_PARAM_INFIX + "_1": 1,
                "b" + OH_PARAM_INFIX + "_2": 0,
                # Only two choices => one parameter.
                "c" + OH_PARAM_INFIX: 0,
                "d": 10.0,
            }
        )
        self.observation_features = ObservationFeatures(
            parameters={"x": 2.2, "a": 2, "b": "b", "c": False, "d": 10.0}
        )

    def testInit(self):
        self.assertEqual(list(self.t.encoded_parameters.keys()), ["b", "c"])
        self.assertEqual(list(self.t2.encoded_parameters.keys()), ["b", "c"])

    def testTransformObservationFeatures(self):
        observation_features = [self.observation_features]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, [self.transformed_features])
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)
        # Test partial transform
        obs_ft3 = [ObservationFeatures(parameters={"x": 2.2, "b": "b"})]
        obs_ft3 = self.t.transform_observation_features(obs_ft3)
        self.assertEqual(
            obs_ft3[0],
            ObservationFeatures(
                parameters={
                    "x": 2.2,
                    "b" + OH_PARAM_INFIX + "_0": 0,
                    "b" + OH_PARAM_INFIX + "_1": 1,
                    "b" + OH_PARAM_INFIX + "_2": 0,
                }
            ),
        )
        obs_ft5 = self.t.transform_observation_features([ObservationFeatures({})])
        self.assertEqual(obs_ft5[0], ObservationFeatures({}))

    def testRandomizedTransform(self):
        observation_features = [self.observation_features]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t2.transform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, [self.transformed_features])
        obs_ft2 = self.t2.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

    def testTransformSearchSpace(self):
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)

        # Parameter type fixed.
        self.assertEqual(ss2.parameters["x"].parameter_type, ParameterType.FLOAT)
        self.assertEqual(ss2.parameters["a"].parameter_type, ParameterType.INT)
        self.assertEqual(
            ss2.parameters["b" + OH_PARAM_INFIX + "_0"].parameter_type,
            ParameterType.FLOAT,
        )
        self.assertEqual(
            ss2.parameters["b" + OH_PARAM_INFIX + "_1"].parameter_type,
            ParameterType.FLOAT,
        )
        self.assertEqual(
            ss2.parameters["c" + OH_PARAM_INFIX].parameter_type, ParameterType.FLOAT
        )
        self.assertEqual(ss2.parameters["d"].parameter_type, ParameterType.FLOAT)

        # Parameter range fixed to [0,1].
        self.assertEqual(ss2.parameters["b" + OH_PARAM_INFIX + "_0"].lower, 0.0)
        self.assertEqual(ss2.parameters["b" + OH_PARAM_INFIX + "_1"].upper, 1.0)
        self.assertEqual(ss2.parameters["c" + OH_PARAM_INFIX].lower, 0.0)
        self.assertEqual(ss2.parameters["c" + OH_PARAM_INFIX].upper, 1.0)

        # Ensure we error if we try to transform a fidelity parameter
        ss3 = SearchSpace(
            parameters=[
                ChoiceParameter(
                    "b",
                    parameter_type=ParameterType.STRING,
                    values=["a", "b", "c"],
                    is_fidelity=True,
                    target_value="c",
                )
            ]
        )
        t = OneHot(search_space=ss3, observation_features=None, observation_data=None)
        with self.assertRaises(ValueError):
            t.transform_search_space(ss3)

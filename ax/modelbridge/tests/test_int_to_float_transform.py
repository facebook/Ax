#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import OrderConstraint, SumConstraint
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.int_to_float import IntToFloat
from ax.utils.common.testutils import TestCase


class IntToFloatTransformTest(TestCase):
    def setUp(self):
        parameters = [
            RangeParameter("x", lower=1, upper=3, parameter_type=ParameterType.FLOAT),
            RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
            RangeParameter("d", lower=1, upper=3, parameter_type=ParameterType.INT),
            ChoiceParameter(
                "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
            ),
        ]
        self.search_space = SearchSpace(
            parameters=parameters,
            parameter_constraints=[
                OrderConstraint(
                    lower_parameter=parameters[0], upper_parameter=parameters[1]
                )
            ],
        )
        self.t = IntToFloat(
            search_space=self.search_space,
            observation_features=None,
            observation_data=None,
        )
        self.t2 = IntToFloat(
            search_space=self.search_space,
            observation_features=None,
            observation_data=None,
            config={"rounding": "randomized"},
        )

    def testInit(self):
        self.assertEqual(self.t.transform_parameters, {"a", "d"})

    def testTransformObservationFeatures(self):
        observation_features = [
            ObservationFeatures(parameters={"x": 2.2, "a": 2, "b": "b", "d": 4})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [ObservationFeatures(parameters={"x": 2.2, "a": 2, "b": "b", "d": 4})],
        )
        self.assertTrue(isinstance(obs_ft2[0].parameters["a"], float))
        self.assertTrue(isinstance(obs_ft2[0].parameters["d"], float))
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

        # Let the transformed space be a float, verify it becomes an int.
        obs_ft3 = [
            ObservationFeatures(parameters={"x": 2.2, "a": 2.2, "b": "b", "d": 3.8})
        ]
        obs_ft3 = self.t.untransform_observation_features(obs_ft3)
        self.assertEqual(obs_ft3, observation_features)

        # Test forward transform on partial observation
        obs_ft4 = [ObservationFeatures(parameters={"x": 2.2, "d": 4})]
        obs_ft4 = self.t.transform_observation_features(obs_ft4)
        self.assertEqual(obs_ft4, [ObservationFeatures(parameters={"x": 2.2, "d": 4})])
        self.assertTrue(isinstance(obs_ft4[0].parameters["d"], float))
        obs_ft5 = self.t.transform_observation_features([ObservationFeatures({})])
        self.assertEqual(obs_ft5[0], ObservationFeatures({}))

    def testTransformObservationFeaturesRandomized(self):
        observation_features = [
            ObservationFeatures(parameters={"x": 2.2, "a": 2, "b": "b", "d": 4})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t2.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [ObservationFeatures(parameters={"x": 2.2, "a": 2, "b": "b", "d": 4})],
        )
        self.assertTrue(isinstance(obs_ft2[0].parameters["a"], float))
        self.assertTrue(isinstance(obs_ft2[0].parameters["d"], float))
        obs_ft2 = self.t2.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

    def testTransformSearchSpace(self):
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        self.assertTrue(ss2.parameters["a"].parameter_type, ParameterType.FLOAT)
        self.assertTrue(ss2.parameters["d"].parameter_type, ParameterType.FLOAT)

    def testRoundingWithConstrainedIntRanges(self):
        parameters = [
            RangeParameter("x", lower=1, upper=3, parameter_type=ParameterType.INT),
            RangeParameter("y", lower=1, upper=3, parameter_type=ParameterType.INT),
        ]
        constrained_int_search_space = SearchSpace(
            parameters=parameters,
            parameter_constraints=[
                SumConstraint(parameters=parameters, is_upper_bound=True, bound=5)
            ],
        )
        t = IntToFloat(
            search_space=constrained_int_search_space,
            observation_features=None,
            observation_data=None,
        )
        self.assertEqual(t.rounding, "randomized")
        observation_features = [ObservationFeatures(parameters={"x": 2.6, "y": 2.6})]
        self.assertTrue(
            constrained_int_search_space.check_membership(
                t.untransform_observation_features(
                    observation_features=observation_features
                )[0].parameters
            )
        )

    def testRoundingWithImpossiblyConstrainedIntRanges(self):
        parameters = [
            RangeParameter("x", lower=1, upper=3, parameter_type=ParameterType.INT),
            RangeParameter("y", lower=1, upper=3, parameter_type=ParameterType.INT),
        ]
        constrained_int_search_space = SearchSpace(
            parameters=parameters,
            parameter_constraints=[
                SumConstraint(parameters=parameters, is_upper_bound=True, bound=3)
            ],
        )
        t = IntToFloat(
            search_space=constrained_int_search_space,
            observation_features=None,
            observation_data=None,
        )
        self.assertEqual(t.rounding, "randomized")
        observation_features = [ObservationFeatures(parameters={"x": 2.6, "y": 2.6})]
        self.assertFalse(
            constrained_int_search_space.check_membership(
                t.untransform_observation_features(
                    observation_features=observation_features
                )[0].parameters
            )
        )

#!/usr/bin/env python3

from copy import deepcopy

from ae.lazarus.ae.core.observation import ObservationFeatures
from ae.lazarus.ae.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.modelbridge.transforms.int_range_to_choice import IntRangeToChoice
from ae.lazarus.ae.utils.common.testutils import TestCase


class IntRangeToChoiceTransformTest(TestCase):
    def setUp(self):
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter("a", lower=1, upper=5, parameter_type=ParameterType.INT),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
            ],
            parameter_constraints=[],
        )
        self.t = IntRangeToChoice(
            search_space=self.search_space,
            observation_features=None,
            observation_data=None,
        )

    def testInit(self):
        self.assertEqual(self.t.transform_params, {"a"})

    def testTransformObservationFeatures(self):
        observation_features = [ObservationFeatures(parameters={"a": 2, "b": "b"})]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, [ObservationFeatures(parameters={"a": 2, "b": "b"})])
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

    def testTransformSearchSpace(self):
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        self.assertTrue(isinstance(ss2.parameters["a"], ChoiceParameter))
        self.assertTrue(ss2.parameters["a"].values, [1, 2, 3, 4, 5])

#!/usr/bin/env python3

from copy import deepcopy

from ae.lazarus.ae.core.condition import Condition
from ae.lazarus.ae.core.observation import ObservationFeatures
from ae.lazarus.ae.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.generator.transforms.search_space_to_choice import (
    SearchSpaceToChoice,
)
from ae.lazarus.ae.utils.common.testutils import TestCase


class SearchSpaceToChoiceTest(TestCase):
    def setUp(self):
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
        self.signature_to_parameterization = {
            Condition(params=obsf.parameters).signature: obsf.parameters
            for obsf in self.observation_features
        }
        self.transformed_features = [
            ObservationFeatures(
                parameters={
                    "conditions": Condition(params={"a": 2, "b": "a"}).signature
                }
            ),
            ObservationFeatures(
                parameters={
                    "conditions": Condition(params={"a": 3, "b": "b"}).signature
                }
            ),
            ObservationFeatures(
                parameters={
                    "conditions": Condition(params={"a": 3, "b": "c"}).signature
                }
            ),
        ]
        self.t = SearchSpaceToChoice(
            search_space=self.search_space,
            observation_features=self.observation_features,
            observation_data=None,
        )

    def testTransformSearchSpace(self):
        ss2 = self.search_space.clone()
        ss2 = self.t.transform_search_space(ss2)
        self.assertEqual(len(ss2.parameters), 1)
        expected_parameter = ChoiceParameter(
            name="conditions",
            parameter_type=ParameterType.STRING,
            values=list(self.t.signature_to_parameterization.keys()),
        )
        self.assertEqual(ss2.parameters.get("conditions"), expected_parameter)

    def testTransformObservationFeatures(self):
        obs_ft2 = deepcopy(self.observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, self.transformed_features)
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, self.observation_features)

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import numpy as np
from ax.core.arm import Arm
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.transforms.search_space_to_choice import SearchSpaceToChoice
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space


class SearchSpaceToChoiceTest(TestCase):
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
        self.signature_to_parameterization = {
            Arm(parameters=obsf.parameters).signature: obsf.parameters
            for obsf in self.observation_features
        }
        self.transformed_features = [
            ObservationFeatures(
                parameters={"arms": Arm(parameters={"a": 2, "b": "a"}).signature}
            ),
            ObservationFeatures(
                parameters={"arms": Arm(parameters={"a": 3, "b": "b"}).signature}
            ),
            ObservationFeatures(
                parameters={"arms": Arm(parameters={"a": 3, "b": "c"}).signature}
            ),
        ]
        self.observations = [
            Observation(
                data=ObservationData([], np.array([]), np.empty((0, 0))), features=obsf
            )
            for obsf in self.observation_features
        ]
        self.t = SearchSpaceToChoice(
            search_space=self.search_space,
            observations=self.observations,
        )
        self.t2 = SearchSpaceToChoice(
            search_space=self.search_space,
            observations=self.observations[:1],
        )
        self.t3 = SearchSpaceToChoice(
            search_space=self.search_space,
            observations=self.observations,
            config={"use_ordered": True},
        )

    def testTransformSearchSpace(self) -> None:
        ss2 = self.search_space.clone()
        ss2 = self.t.transform_search_space(ss2)
        self.assertEqual(len(ss2.parameters), 1)
        expected_parameter = ChoiceParameter(
            name="arms",
            parameter_type=ParameterType.STRING,
            values=list(self.t.signature_to_parameterization.keys()),
        )
        self.assertEqual(ss2.parameters.get("arms"), expected_parameter)

        # With use_ordered
        ss2 = self.search_space.clone()
        ss2 = self.t3.transform_search_space(ss2)
        self.assertEqual(len(ss2.parameters), 1)
        expected_parameter = ChoiceParameter(
            name="arms",
            parameter_type=ParameterType.STRING,
            values=list(self.t.signature_to_parameterization.keys()),
            is_ordered=True,
        )
        self.assertEqual(ss2.parameters.get("arms"), expected_parameter)

        # Test error if there are fidelities
        ss3 = SearchSpace(
            parameters=[
                RangeParameter(
                    "a",
                    lower=1,
                    upper=3,
                    parameter_type=ParameterType.FLOAT,
                    is_fidelity=True,
                    target_value=3,
                )
            ]
        )
        with self.assertRaises(ValueError):
            SearchSpaceToChoice(
                search_space=ss3,
                observations=[],
            )

    def testTransformSearchSpaceWithFixedParam(self) -> None:
        ss2 = self.search_space.clone()
        ss2 = self.t2.transform_search_space(ss2)
        self.assertEqual(len(ss2.parameters), 1)
        expected_parameter = FixedParameter(
            name="arms",
            parameter_type=ParameterType.STRING,
            value=list(self.t2.signature_to_parameterization.keys())[0],
        )
        self.assertEqual(ss2.parameters.get("arms"), expected_parameter)

    def testTransformObservationFeatures(self) -> None:
        obs_ft2 = deepcopy(self.observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, self.transformed_features)
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, self.observation_features)

    def test_w_robust_search_space(self) -> None:
        rss = get_robust_search_space()
        # Raises an error in __init__.
        with self.assertRaisesRegex(UnsupportedError, "transform is not supported"):
            SearchSpaceToChoice(
                search_space=rss,
                observations=[],
            )

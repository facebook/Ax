#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.modelbridge.transforms.remove_fixed import RemoveFixed
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space


class RemoveFixedTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "a", lower=1, upper=3, parameter_type=ParameterType.FLOAT
                ),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
                FixedParameter("c", parameter_type=ParameterType.STRING, value="a"),
                ChoiceParameter("d", parameter_type=ParameterType.INT, values=[1]),
            ]
        )
        self.t = RemoveFixed(
            search_space=self.search_space,
            observations=[],
        )

    def test_Init(self) -> None:
        self.assertEqual(list(self.t.single_choice_params.keys()), ["c", "d"])

    def test_TransformObservationFeatures(self) -> None:
        observation_features = [
            ObservationFeatures(parameters={"a": 2.2, "b": "b", "c": "a", "d": 1})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2, [ObservationFeatures(parameters={"a": 2.2, "b": "b"})]
        )
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

        observation_features = [
            ObservationFeatures(parameters={"a": 2.2, "b": "b", "c": "a", "d": 1})
        ]
        observation_features_different = [
            ObservationFeatures(parameters={"a": 2.2, "b": "b", "c": "b", "d": 10})
        ]
        # Fixed parameter is out of design. It will still get removed.
        t_obs = self.t.transform_observation_features(observation_features)
        t_obs_different = self.t.transform_observation_features(
            observation_features_different
        )
        self.assertEqual(t_obs, t_obs_different)

    def test_TransformSearchSpace(self) -> None:
        ss2 = self.search_space.clone()
        ss2 = self.t.transform_search_space(ss2)
        self.assertEqual(ss2.parameters.get("c"), None)
        self.assertEqual(ss2.parameters.get("d"), None)

    def test_w_parameter_distributions(self) -> None:
        rss = get_robust_search_space()
        rss.add_parameter(
            FixedParameter("d", parameter_type=ParameterType.STRING, value="a"),
        )
        rss.add_parameter(
            ChoiceParameter("e", parameter_type=ParameterType.INT, values=[1]),
        )
        # Transform a non-distributional parameter.
        t = RemoveFixed(
            search_space=rss,
            observations=[],
        )
        rss = t.transform_search_space(rss)
        # Make sure that the return value is still a RobustSearchSpace.
        self.assertIsInstance(rss, RobustSearchSpace)
        self.assertEqual(len(rss.parameters.keys()), 4)
        # pyre-fixme[16]: `SearchSpace` has no attribute `parameter_distributions`.
        self.assertEqual(len(rss.parameter_distributions), 2)
        self.assertNotIn("d", rss.parameters)
        self.assertNotIn("e", rss.parameters)
        # Test with environmental variables.
        all_params = list(rss.parameters.values())
        rss = RobustSearchSpace(
            parameters=all_params[2:],
            parameter_distributions=rss.parameter_distributions,
            # pyre-fixme[16]: `SearchSpace` has no attribute `num_samples`.
            num_samples=rss.num_samples,
            environmental_variables=all_params[:2],
        )
        rss.add_parameter(
            FixedParameter("d", parameter_type=ParameterType.STRING, value="a"),
        )
        rss.add_parameter(
            ChoiceParameter("e", parameter_type=ParameterType.INT, values=[1]),
        )
        t = RemoveFixed(
            search_space=rss,
            observations=[],
        )
        rss = t.transform_search_space(rss)
        self.assertIsInstance(rss, RobustSearchSpace)
        self.assertEqual(len(rss.parameters.keys()), 4)
        self.assertEqual(len(rss.parameter_distributions), 2)
        # pyre-fixme[16]: `SearchSpace` has no attribute `_environmental_variables`.
        self.assertEqual(len(rss._environmental_variables), 2)
        self.assertNotIn("d", rss.parameters)

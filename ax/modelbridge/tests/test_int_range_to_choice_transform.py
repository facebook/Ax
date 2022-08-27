#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.transforms.int_range_to_choice import IntRangeToChoice
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space


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
        self.assertEqual(self.t.transform_parameters, {"a"})

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

    def test_w_robust_search_space(self):
        rss = get_robust_search_space()
        # Transform a non-distributional parameter.
        t = IntRangeToChoice(
            search_space=rss,
            observation_features=None,
            observation_data=None,
        )
        rss_new = t.transform_search_space(rss)
        # Make sure that the return value is still a RobustSearchSpace.
        self.assertIsInstance(rss_new, RobustSearchSpace)
        self.assertEqual(set(rss.parameters.keys()), set(rss_new.parameters.keys()))
        self.assertEqual(rss.parameter_distributions, rss_new.parameter_distributions)
        self.assertIsInstance(rss_new.parameters.get("z"), ChoiceParameter)
        # Test with environmental variables.
        all_params = list(rss.parameters.values())
        rss = RobustSearchSpace(
            parameters=all_params[2:],
            parameter_distributions=rss.parameter_distributions,
            num_samples=rss.num_samples,
            environmental_variables=all_params[:2],
        )
        t = IntRangeToChoice(
            search_space=rss,
            observation_features=None,
            observation_data=None,
        )
        rss_new = t.transform_search_space(rss)
        self.assertIsInstance(rss_new, RobustSearchSpace)
        self.assertEqual(set(rss.parameters.keys()), set(rss_new.parameters.keys()))
        self.assertEqual(rss.parameter_distributions, rss_new.parameter_distributions)
        self.assertEqual(rss._environmental_variables, rss_new._environmental_variables)
        self.assertIsInstance(rss_new.parameters.get("z"), ChoiceParameter)
        # Error with distributional parameter.
        rss = get_robust_search_space(use_discrete=True)
        t = IntRangeToChoice(
            search_space=rss,
            observation_features=None,
            observation_data=None,
        )
        with self.assertRaisesRegex(UnsupportedError, "transform is not supported"):
            t.transform_search_space(rss)

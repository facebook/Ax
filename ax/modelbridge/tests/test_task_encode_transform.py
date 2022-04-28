#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.modelbridge.transforms.task_encode import TaskEncode
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space


class TaskEncodeTransformTest(TestCase):
    def setUp(self):
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x", lower=1, upper=3, parameter_type=ParameterType.FLOAT
                ),
                ChoiceParameter(
                    "b",
                    parameter_type=ParameterType.FLOAT,
                    values=[1.0, 10.0, 100.0],
                    is_ordered=True,
                ),
                ChoiceParameter(
                    "c",
                    parameter_type=ParameterType.STRING,
                    values=["online", "offline"],
                    is_task=True,
                ),
            ]
        )
        self.t = TaskEncode(
            search_space=self.search_space,
            observation_features=None,
            observation_data=None,
        )

    def testInit(self):
        self.assertEqual(list(self.t.encoded_parameters.keys()), ["c"])

    def testTransformObservationFeatures(self):
        observation_features = [
            ObservationFeatures(parameters={"x": 2.2, "b": 10.0, "c": "online"})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2, [ObservationFeatures(parameters={"x": 2.2, "b": 10.0, "c": 0})]
        )
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)
        # Test transform on partial features
        obs_ft3 = [ObservationFeatures(parameters={"c": "offline"})]
        obs_ft3 = self.t.transform_observation_features(obs_ft3)
        self.assertEqual(obs_ft3[0], ObservationFeatures(parameters={"c": 1}))
        obs_ft5 = self.t.transform_observation_features([ObservationFeatures({})])
        self.assertEqual(obs_ft5[0], ObservationFeatures({}))

    def testTransformSearchSpace(self):
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)

        self.assertIsInstance(ss2.parameters["x"], RangeParameter)
        self.assertIsInstance(ss2.parameters["b"], ChoiceParameter)
        self.assertIsInstance(ss2.parameters["c"], ChoiceParameter)
        self.assertEqual(ss2.parameters["x"].parameter_type, ParameterType.FLOAT)
        self.assertEqual(ss2.parameters["b"].parameter_type, ParameterType.FLOAT)
        self.assertEqual(ss2.parameters["c"].parameter_type, ParameterType.INT)

        self.assertEqual(ss2.parameters["c"].values, [0, 1])

        # Test error if there are fidelities
        ss3 = SearchSpace(
            parameters=[
                ChoiceParameter(
                    "c",
                    parameter_type=ParameterType.STRING,
                    values=["online", "offline"],
                    is_task=True,
                    is_fidelity=True,
                    target_value="online",
                )
            ]
        )
        with self.assertRaises(ValueError):
            TaskEncode(
                search_space=ss3, observation_features=None, observation_data=None
            )

    def test_w_parameter_distributions(self):
        rss = get_robust_search_space()
        rss.parameters["c"]._is_task = True
        # Transform a non-distributional parameter.
        t = TaskEncode(
            search_space=rss,
            observation_features=None,
            observation_data=None,
        )
        rss_new = t.transform_search_space(rss)
        # Make sure that the return value is still a RobustSearchSpace.
        self.assertIsInstance(rss_new, RobustSearchSpace)
        self.assertEqual(set(rss.parameters.keys()), set(rss_new.parameters.keys()))
        self.assertEqual(rss.parameter_distributions, rss_new.parameter_distributions)
        self.assertEqual(rss_new.parameters.get("c").parameter_type, ParameterType.INT)
        # Test with environmental variables.
        all_params = list(rss.parameters.values())
        rss = RobustSearchSpace(
            parameters=all_params[2:],
            parameter_distributions=rss.parameter_distributions,
            num_samples=rss.num_samples,
            environmental_variables=all_params[:2],
        )
        t = TaskEncode(
            search_space=rss,
            observation_features=None,
            observation_data=None,
        )
        rss_new = t.transform_search_space(rss)
        self.assertIsInstance(rss_new, RobustSearchSpace)
        self.assertEqual(set(rss.parameters.keys()), set(rss_new.parameters.keys()))
        self.assertEqual(rss.parameter_distributions, rss_new.parameter_distributions)
        self.assertEqual(rss._environmental_variables, rss_new._environmental_variables)
        self.assertEqual(rss_new.parameters.get("c").parameter_type, ParameterType.INT)

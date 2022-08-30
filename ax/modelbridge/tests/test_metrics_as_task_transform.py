#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import numpy as np

from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.parameter import ChoiceParameter
from ax.modelbridge.transforms.metrics_as_task import MetricsAsTask
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_search_space_for_range_values


class MetricsAsTaskTransformTest(TestCase):
    def setUp(self) -> None:
        self.metric_task_map = {
            "metric1": ["metric2", "metric3"],
            "metric2": ["metric3"],
        }
        self.observations = [
            Observation(
                data=ObservationData(
                    metric_names=["metric1", "metric2", "metric3"],
                    means=np.array([1.0, 2.0, 3.0]),
                    covariance=np.diag([1.0, 2.0, 3.0]),
                ),
                features=ObservationFeatures(parameters={"x": 5.0, "y": 2.0}),
                arm_name="0_0",
            ),
            Observation(
                data=ObservationData(
                    metric_names=["metric3"],
                    means=np.array([30.0]),
                    covariance=np.array([[30.0]]),
                ),
                features=ObservationFeatures(parameters={"x": 10.0, "y": 4.0}),
            ),
        ]
        self.search_space = get_search_space_for_range_values(min=0.0, max=20.0)
        self.expected_new_observations = [
            Observation(
                data=ObservationData(
                    metric_names=["metric1", "metric2", "metric3"],
                    means=np.array([1.0, 2.0, 3.0]),
                    covariance=np.diag([1.0, 2.0, 3.0]),
                ),
                features=ObservationFeatures(
                    parameters={"x": 5.0, "y": 2.0, "METRIC_TASK": "TARGET"}
                ),
                arm_name="0_0",
            ),
            Observation(
                data=ObservationData(
                    metric_names=["metric2", "metric3"],
                    means=np.array([1.0, 1.0]),
                    covariance=np.diag([1.0, 1.0]),
                ),
                features=ObservationFeatures(
                    parameters={"x": 5.0, "y": 2.0, "METRIC_TASK": "metric1"}
                ),
                arm_name="0_0",
            ),
            Observation(
                data=ObservationData(
                    metric_names=["metric3"],
                    means=np.array([2.0]),
                    covariance=np.array([[2.0]]),
                ),
                features=ObservationFeatures(
                    parameters={"x": 5.0, "y": 2.0, "METRIC_TASK": "metric2"}
                ),
                arm_name="0_0",
            ),
            Observation(
                data=ObservationData(
                    metric_names=["metric3"],
                    means=np.array([30.0]),
                    covariance=np.array([[30.0]]),
                ),
                features=ObservationFeatures(
                    parameters={"x": 10.0, "y": 4.0, "METRIC_TASK": "TARGET"}
                ),
            ),
        ]
        self.t = MetricsAsTask(
            search_space=self.search_space,
            observations=self.observations,
            config={"metric_task_map": self.metric_task_map},
        )

    def testInit(self) -> None:
        with self.assertRaises(ValueError):
            MetricsAsTask(
                search_space=self.search_space, observations=self.observations
            )

    def testTransformObservations(self) -> None:
        new_obs = self.t.transform_observations(deepcopy(self.observations))
        self.assertEqual(new_obs, self.expected_new_observations)

        new_obs = self.t.untransform_observations(new_obs)
        self.assertEqual(new_obs, self.observations)

    def testTransformObservationFeatures(self) -> None:
        obsfs_t = self.t.transform_observation_features(
            deepcopy([obs.features for obs in self.observations])
        )
        for obsf in obsfs_t:
            assert obsf.parameters["METRIC_TASK"] == "TARGET"

        obsfs_t = self.t.untransform_observation_features(obsfs_t)
        for obsf in obsfs_t:
            assert "METRIC_TASK" not in obsf.parameters

        with self.assertRaises(ValueError):
            self.t.untransform_observation_features(
                deepcopy([obs.features for obs in self.expected_new_observations])
            )

    def testTransformSearchSpace(self) -> None:
        new_ss = self.t._transform_search_space(deepcopy(self.search_space))
        self.assertEqual(len(new_ss.parameters), 3)
        new_param = new_ss.parameters["METRIC_TASK"]
        self.assertIsInstance(new_param, ChoiceParameter)
        self.assertEqual(
            new_param.values, ["TARGET", "metric1", "metric2"]  # pyre-ignore
        )
        self.assertTrue(new_param.is_task)  # pyre-ignore

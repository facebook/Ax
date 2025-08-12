#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.base import Transform
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.exceptions.core import UnsupportedError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class TransformsTest(TestCase):
    def test_IdentityTransform(self) -> None:
        # Test that the identity transform does not mutate anything
        t = Transform(MagicMock(), MagicMock())
        x = MagicMock()
        ys = []
        ys.append(t.transform_search_space(x))
        ys.append(t.transform_optimization_config(x, x, x))
        ys.append(t.transform_observation_features(x))
        ys.append(t._transform_observation_data(x))
        ys.append(t.untransform_observation_features(x))
        ys.append(t._untransform_observation_data(x))
        self.assertEqual(len(x.mock_calls), 0)
        for y in ys:
            self.assertEqual(y, x)

    def test_TransformObservations(self) -> None:
        # Test that this is an identity transform
        means = np.array([3.0, 4.0])
        metric_names = ["a", "b"]
        covariance = np.array([[1.0, 2.0], [3.0, 4.0]])
        parameters = {"x": 1.0, "y": "cat"}
        arm_name = "armmy"
        observation = Observation(
            features=ObservationFeatures(parameters=parameters),  # pyre-ignore
            data=ObservationData(
                metric_names=metric_names, means=means, covariance=covariance
            ),
            arm_name=arm_name,
        )
        t = Transform()
        obs1 = t.transform_observations([deepcopy(observation)])[0]
        obs2 = t.untransform_observations([deepcopy(obs1)])[0]
        for obs in [obs1, obs2]:
            self.assertTrue(np.array_equal(obs.data.means, means))
            self.assertTrue(np.array_equal(obs.data.covariance, covariance))
            self.assertEqual(obs.data.metric_names, metric_names)
            self.assertEqual(obs.features.parameters, parameters)
            self.assertEqual(obs.arm_name, arm_name)

    def test_with_experiment_data(self) -> None:
        experiment = get_branin_experiment(with_completed_batch=True)
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        with self.assertRaisesRegex(UnsupportedError, "Only one of"):
            Transform(observations=[], experiment_data=experiment_data)
        t = Transform(experiment_data=experiment_data)
        # Errors out since no_op_for_experiment_data defaults to False.
        with self.assertRaisesRegex(NotImplementedError, "transform_experiment_data"):
            t.transform_experiment_data(experiment_data=experiment_data)
        # No-op when no_op_for_experiment_data is True.
        t.no_op_for_experiment_data = True
        self.assertIs(
            t.transform_experiment_data(experiment_data=experiment_data),
            experiment_data,
        )

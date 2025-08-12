#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from math import sqrt

import numpy as np
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.merge_repeated_measurements import MergeRepeatedMeasurements
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.exceptions.core import DataRequiredError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from pandas import DataFrame
from pandas.testing import assert_frame_equal


def compare_obs(
    test: TestCase, obs1: Observation, obs2: Observation, discrepancy_tol: float = 1e-8
) -> None:
    test.assertEqual(obs1.data.metric_names, obs2.data.metric_names)
    test.assertTrue(np.array_equal(obs1.data.means, obs2.data.means))
    discrep = np.max(np.abs(obs1.data.covariance - obs2.data.covariance))
    test.assertLessEqual(discrep, discrepancy_tol)
    test.assertEqual(obs1.features.parameters, obs2.features.parameters)


class MergeRepeatedMeasurementsTransformTest(TestCase):
    def test_Transform(self) -> None:
        obs_feats1 = ObservationFeatures(parameters={"a": 0.0})
        with self.assertRaisesRegex(
            DataRequiredError,
            "`MergeRepeatedMeasurements` transform requires non-empty data",
        ):
            # test that observations are required
            MergeRepeatedMeasurements()
        # test nan in covariance
        observation = Observation(
            data=ObservationData(
                metric_names=["m1"],
                means=np.array([1.0]),
                covariance=np.array([[float("nan")]]),
            ),
            features=obs_feats1,
        )
        with self.assertRaisesRegex(
            NotImplementedError, "All observations must have arm names."
        ):
            MergeRepeatedMeasurements(observations=[observation])
        observation.arm_name = "0_0"
        with self.assertRaisesRegex(
            NotImplementedError, "All metrics must have noise observations."
        ):
            MergeRepeatedMeasurements(observations=[observation])
        # test full covariance
        observation = Observation(
            data=ObservationData(
                metric_names=["m1", "m2"],
                means=np.array([1.0, 1.0]),
                covariance=np.ones((2, 2)),
            ),
            features=obs_feats1,
            arm_name="0_0",
        )
        with self.assertRaisesRegex(
            NotImplementedError, "Only independent metrics are currently supported."
        ):
            MergeRepeatedMeasurements(observations=[observation])

        # test noiseless, different means
        zero_covar = np.zeros((1, 1))
        observations = [
            Observation(
                data=ObservationData(
                    metric_names=["m1"],
                    means=np.array([1.0]),
                    covariance=zero_covar,
                ),
                features=obs_feats1,
                arm_name="0_0",
            ),
            Observation(
                data=ObservationData(
                    metric_names=["m1"],
                    means=np.array([2.0]),
                    covariance=zero_covar,
                ),
                features=obs_feats1,
                arm_name="0_0",
            ),
        ]
        with self.assertRaisesRegex(
            ValueError,
            "All repeated arms with noiseless measurements "
            "must have the same means.",
        ):
            MergeRepeatedMeasurements(observations=observations)
        # test noiseless, same means
        observations = [
            Observation(
                data=ObservationData(
                    metric_names=["m1"],
                    means=np.array([1.0]),
                    covariance=zero_covar,
                ),
                features=obs_feats1,
                arm_name="0_0",
            ),
            Observation(
                data=ObservationData(
                    metric_names=["m1"],
                    means=np.array([1.0]),
                    covariance=zero_covar,
                ),
                features=obs_feats1,
                arm_name="0_0",
            ),
            Observation(
                data=ObservationData(
                    metric_names=["m1"],
                    means=np.array([2.0]),
                    covariance=zero_covar,
                ),
                features=ObservationFeatures(parameters={"a": 2.0}),
                arm_name="0_1",
            ),
        ]
        t = MergeRepeatedMeasurements(observations=observations)
        expected_obs = observations[-2:]
        transformed_obs = t.transform_observations(observations)
        for i in (0, 1):
            compare_obs(
                test=self,
                obs1=expected_obs[i],
                obs2=transformed_obs[i],
                discrepancy_tol=0.0,
            )

        # basic test
        obs_feat1 = ObservationFeatures(parameters={"a": 0.0, "b": 1.0})
        obs1 = Observation(
            data=ObservationData(
                metric_names=["m1", "m2"],
                means=np.array([1.0, 2.0]),
                covariance=np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 2.0],
                    ]
                ),
            ),
            features=obs_feat1,
            arm_name="0_0",
        )
        obs2 = Observation(
            data=ObservationData(
                metric_names=["m1", "m2"],
                means=np.array([1.0, 1.0]),
                covariance=np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 3.0],
                    ]
                ),
            ),
            features=obs_feat1,
            arm_name="0_0",
        )
        # different arm
        obs3 = Observation(
            data=ObservationData(
                metric_names=["m1", "m2"],
                means=np.array([3.0, 1.0]),
                covariance=np.array(
                    [
                        [4.0, 0.0],
                        [0.0, 5.0],
                    ]
                ),
            ),
            features=ObservationFeatures(parameters={"a": 1.0, "b": 0.0}),
            arm_name="0_1",
        )
        expected_obs = Observation(
            data=ObservationData(
                metric_names=["m1", "m2"],
                means=np.array([1.0, 1.6]),
                covariance=np.array([[0.5, 0.0], [0.0, 1.2]]),
            ),
            features=obs_feat1,
            arm_name="0_0",
        )
        observations = [obs1, obs2, obs3]
        observations_copy = deepcopy(observations)
        t = MergeRepeatedMeasurements(observations=observations)
        observations2 = t.transform_observations(observations)
        compare_obs(
            test=self, obs1=expected_obs, obs2=observations2[0], discrepancy_tol=1e-8
        )
        compare_obs(test=self, obs1=obs3, obs2=observations2[1], discrepancy_tol=0.0)
        # test repeating the transform
        observations2_copy = t.transform_observations(observations_copy)
        compare_obs(
            test=self,
            obs1=observations2[0],
            obs2=observations2_copy[0],
            discrepancy_tol=0,
        )
        compare_obs(
            test=self,
            obs1=observations2[1],
            obs2=observations2_copy[1],
            discrepancy_tol=0,
        )
        # check arm names
        arm_names = {obs.arm_name for obs in observations}
        arm_names2 = {obs.arm_name for obs in observations2}
        self.assertEqual(arm_names, arm_names2)

    def test_with_experiment_data(self) -> None:
        # Experiment with similar data as the above test.
        experiment = get_experiment_with_observations(
            observations=[[1.0, 1.0], [1.0, 2.0]],
            sems=[[1.0, sqrt(2.0)], [1.0, sqrt(3.0)]],
            parameterizations=[{"x": 0.0, "y": 1.0}, {"x": 0.0, "y": 1.0}],
        )
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        t = MergeRepeatedMeasurements(experiment_data=experiment_data)
        # Check for correct transform setup.
        w1, w2 = 3 / 5.0, 2 / 5.0
        expected = {
            "m1": {"mean": 1.0, "var": 0.5},
            "m2": {"mean": w1 + 2 * w2, "var": 6 / 5.0},
        }
        actual = t.arm_to_merged["0_0"]
        # m1 is exact, m2 is approximate due to float precision.
        self.assertEqual(actual["m1"], expected["m1"])
        self.assertAlmostEqual(actual["m2"]["mean"], expected["m2"]["mean"], places=5)
        self.assertAlmostEqual(actual["m2"]["var"], expected["m2"]["var"], places=5)
        # Transform the data.
        transformed_data = t.transform_experiment_data(
            experiment_data=deepcopy(experiment_data),
        )
        # Check that the data is transformed correctly.
        # Arm data only retains the first row.
        assert_frame_equal(
            transformed_data.arm_data, experiment_data.arm_data.iloc[[0]]
        )
        # Observation data is overwritten with the merged data.
        expected_obs_data = DataFrame(
            index=transformed_data.arm_data.index,
            columns=transformed_data.observation_data.columns,
            data=[
                [1.0, w1 + 2 * w2, 0.5**0.5, (6 / 5.0) ** 0.5],
            ],
        )
        assert_frame_equal(transformed_data.observation_data, expected_obs_data)

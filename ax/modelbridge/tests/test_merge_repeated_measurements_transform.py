#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.modelbridge.transforms.merge_repeated_measurements import (
    MergeRepeatedMeasurements,
)
from ax.utils.common.testutils import TestCase


def compare_obs(
    test: TestCase, obs1: Observation, obs2: Observation, discrepancy_tol: float = 1e-8
) -> None:
    test.assertEqual(obs1.data.metric_names, obs2.data.metric_names)
    test.assertTrue(np.array_equal(obs1.data.means, obs2.data.means))
    discrep = np.max(np.abs(obs1.data.covariance - obs2.data.covariance))
    test.assertLessEqual(discrep, discrepancy_tol)
    test.assertEqual(obs1.features.parameters, obs2.features.parameters)


class MergeRepeatedMeasurementsTransformTest(TestCase):
    def testTransform(self) -> None:
        obs_feats1 = ObservationFeatures(parameters={"a": 0.0})
        with self.assertRaisesRegex(
            RuntimeError, "MergeRepeatedMeasurements requires observations"
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
            ),
            Observation(
                data=ObservationData(
                    metric_names=["m1"],
                    means=np.array([2.0]),
                    covariance=zero_covar,
                ),
                features=obs_feats1,
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
            ),
            Observation(
                data=ObservationData(
                    metric_names=["m1"],
                    means=np.array([1.0]),
                    covariance=zero_covar,
                ),
                features=obs_feats1,
            ),
            Observation(
                data=ObservationData(
                    metric_names=["m1"],
                    means=np.array([2.0]),
                    covariance=zero_covar,
                ),
                features=ObservationFeatures(parameters={"a": 2.0}),
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
        )
        expected_obs = Observation(
            data=ObservationData(
                metric_names=["m1", "m2"],
                means=np.array([1.0, 1.6]),
                covariance=np.array([[0.5, 0.0], [0.0, 1.2]]),
            ),
            features=obs_feat1,
        )
        observations = [obs1, obs2, obs3]
        t = MergeRepeatedMeasurements(observations=observations)
        observations2 = t.transform_observations(observations)
        compare_obs(
            test=self, obs1=expected_obs, obs2=observations2[0], discrepancy_tol=1e-8
        )
        compare_obs(test=self, obs1=obs3, obs2=observations2[1], discrepancy_tol=0.0)

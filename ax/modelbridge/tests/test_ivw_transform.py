#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.core.observation import ObservationData
from ax.modelbridge.transforms.ivw import IVW, ivw_metric_merge
from ax.utils.common.testutils import TestCase


class IVWTransformTest(TestCase):
    def testNoRepeats(self):
        obsd = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([1.0, 2.0]),
            covariance=np.array([[1.0, 0.2], [0.2, 2.0]]),
        )
        obsd2 = ivw_metric_merge(obsd)
        self.assertEqual(obsd2, obsd)

    def testMerge(self):
        obsd = ObservationData(
            metric_names=["m1", "m2", "m2"],
            means=np.array([1.0, 2.0, 1.0]),
            covariance=np.array([[1.0, 0.2, 0.4], [0.2, 2.0, 0.8], [0.4, 0.8, 3.0]]),
        )
        obsd2 = ivw_metric_merge(obsd)
        self.assertEqual(obsd2.metric_names, ["m1", "m2"])
        self.assertTrue(np.array_equal(obsd2.means, np.array([1.0, 0.6 * 2 + 0.4])))
        cov12 = 0.2 * 0.6 + 0.4 * 0.4
        # var(w1*y1 + w2*y2) =
        # w1 ** 2 * var(y1) + w2 ** 2 * var(y2) + 2 * w1 * w2 * cov(y1, y2)
        cov22 = 0.6 ** 2 * 2.0 + 0.4 ** 2 * 3 + 2 * 0.6 * 0.4 * 0.8
        cov_true = np.array([[1.0, cov12], [cov12, cov22]])
        discrep = np.max(np.abs(obsd2.covariance - cov_true))
        self.assertTrue(discrep < 1e-8)

    def testNoiselessMerge(self):
        # One noiseless
        obsd = ObservationData(
            metric_names=["m1", "m2", "m2"],
            means=np.array([1.0, 2.0, 1.0]),
            covariance=np.array([[1.0, 0.2, 0.4], [0.2, 2.0, 0.8], [0.4, 0.8, 0.0]]),
        )
        obsd2 = ivw_metric_merge(obsd)
        np.array_equal(obsd2.means, np.array([1.0, 1.0]))
        cov_true = np.array([[1.0, 0.4], [0.4, 0.0]])
        self.assertTrue(np.array_equal(obsd2.covariance, cov_true))
        # Conflicting noiseless, default (warn)
        obsd = ObservationData(
            metric_names=["m1", "m2", "m2"],
            means=np.array([1.0, 2.0, 1.0]),
            covariance=np.array([[1.0, 0.2, 0.4], [0.2, 0.0, 0.8], [0.4, 0.8, 0.0]]),
        )
        with self.assertRaises(ValueError):
            obsd2 = ivw_metric_merge(obsd, conflicting_noiseless="wrong")
        obsd2 = ivw_metric_merge(obsd)
        self.assertTrue(np.array_equal(obsd2.means, np.array([1.0, 2.0])))
        cov_true = np.array([[1.0, 0.2], [0.2, 0.0]])
        self.assertTrue(np.array_equal(obsd2.covariance, cov_true))
        # Conflicting noiseless, raise
        with self.assertRaises(ValueError):
            obsd2 = ivw_metric_merge(obsd, conflicting_noiseless="raise")

    def testTransform(self):
        obsd1_0 = ObservationData(
            metric_names=["m1", "m2", "m2"],
            means=np.array([1.0, 2.0, 1.0]),
            covariance=np.array([[1.0, 0.2, 0.4], [0.2, 2.0, 0.8], [0.4, 0.8, 3.0]]),
        )
        obsd1_1 = ObservationData(
            metric_names=["m1", "m1", "m2", "m2"],
            means=np.array([1.0, 1.0, 2.0, 1.0]),
            covariance=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.2, 0.4],
                    [0.0, 0.2, 2.0, 0.8],
                    [0.0, 0.4, 0.8, 3.0],
                ]
            ),
        )
        obsd2_0 = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([1.0, 1.6]),
            covariance=np.array([[1.0, 0.28], [0.28, 1.584]]),
        )
        obsd2_1 = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([1.0, 1.6]),
            covariance=np.array([[0.5, 0.14], [0.14, 1.584]]),
        )
        observation_data = [obsd1_0, obsd1_1]
        t = IVW(None, None, None)
        observation_data2 = t.transform_observation_data(observation_data, [])
        observation_data2_true = [obsd2_0, obsd2_1]
        for i, obsd in enumerate(observation_data2_true):
            self.assertEqual(observation_data2[i].metric_names, obsd.metric_names)
            self.assertTrue(np.array_equal(observation_data2[i].means, obsd.means))
            discrep = np.max(np.abs(observation_data2[i].covariance - obsd.covariance))
            self.assertTrue(discrep < 1e-8)

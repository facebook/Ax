#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from math import isfinite, isnan

import numpy as np
from ax.core.observation import ObservationData
from ax.modelbridge.transforms.power_transform_y import (
    PowerTransformY,
    _compute_power_transforms,
    _compute_inverse_bounds,
)
from ax.modelbridge.transforms.utils import get_data, match_ci_width_truncated
from ax.utils.common.testutils import TestCase
from sklearn.preprocessing import PowerTransformer


class PowerTransformYTest(TestCase):
    def setUp(self):
        self.obsd1 = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([0.5, 0.9]),
            covariance=np.array([[0.03, 0.0], [0.0, 0.001]]),
        )
        self.obsd2 = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([0.1, 0.4]),
            covariance=np.array([[0.005, 0.0], [0.0, 0.05]]),
        )
        self.obsd3 = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([0.9, 0.8]),
            covariance=np.array([[0.02, 0.0], [0.0, 0.01]]),
        )
        self.obsd_nan = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([0.3, 0.2]),
            covariance=np.array([[float("nan"), 0.0], [0.0, float("nan")]]),
        )

    def testInit(self):
        shared_init_args = {
            "search_space": None,
            "observation_features": None,
            "observation_data": [self.obsd1, self.obsd2],
        }
        # Test error for not specifying a config
        with self.assertRaises(ValueError):
            PowerTransformY(**shared_init_args)
        # Test error for not specifying at least one metric
        with self.assertRaises(ValueError):
            PowerTransformY(**shared_init_args, config={})
        # Test default init
        for m in ["m1", "m2"]:
            tf = PowerTransformY(**shared_init_args, config={"metrics": [m]})
            # tf.power_transforms should only exist for m and be a PowerTransformer
            self.assertIsInstance(tf.power_transforms, dict)
            self.assertEqual([*tf.power_transforms], [m])  # Check keys
            self.assertIsInstance(tf.power_transforms[m], PowerTransformer)
            # tf.inv_bounds should only exist for m and be a tuple of length 2
            self.assertIsInstance(tf.inv_bounds, dict)
            self.assertEqual([*tf.inv_bounds], [m])  # Check keys
            self.assertIsInstance(tf.inv_bounds[m], tuple)
            self.assertTrue(len(tf.inv_bounds[m]) == 2)

    def testGetData(self):
        for m in ["m1", "m2"]:
            Ys = get_data([self.obsd1, self.obsd2, self.obsd3], m)
            self.assertIsInstance(Ys, dict)
            self.assertEqual([*Ys], [m])
            if m == "m1":
                self.assertEqual(Ys[m], [0.5, 0.1, 0.9])
            else:
                self.assertEqual(Ys[m], [0.9, 0.4, 0.8])

    def testComputePowerTransform(self):
        Ys = get_data([self.obsd1, self.obsd2, self.obsd3], ["m2"])
        pts = _compute_power_transforms(Ys)
        self.assertEqual(pts["m2"].method, "yeo-johnson")
        self.assertIsInstance(pts["m2"].lambdas_, np.ndarray)
        self.assertEqual(pts["m2"].lambdas_.shape, (1,))
        Y_np = np.array(Ys["m2"])[:, None]
        Y_trans = pts["m2"].transform(Y_np)
        # Output should be standardized
        self.assertAlmostEqual(Y_trans.mean(), 0.0)
        self.assertAlmostEqual(Y_trans.std(), 1.0)
        # Transform back
        Y_np2 = pts["m2"].inverse_transform(Y_trans)
        self.assertAlmostEqual(np.max(np.abs(Y_np - Y_np2)), 0.0)

    def testComputeInverseBounds(self):
        Ys = get_data([self.obsd1, self.obsd2, self.obsd3], ["m2"])
        pt = _compute_power_transforms(Ys)["m2"]
        # lambda < 0: im(f) = (-inf, -1/lambda) without standardization
        pt.lambdas_.fill(-2.5)
        bounds = _compute_inverse_bounds({"m2": pt})["m2"]
        self.assertEqual(bounds[0], -np.inf)
        # Make sure we got the boundary right
        left = pt.inverse_transform(np.array(bounds[1] - 0.01, ndmin=2))
        right = pt.inverse_transform(np.array(bounds[1] + 0.01, ndmin=2))
        self.assertTrue(isnan(right) and not isnan(left))
        # 0 <= lambda <= 2: im(f) = R
        pt.lambdas_.fill(1.0)
        bounds = _compute_inverse_bounds({"m2": pt})["m2"]
        self.assertTrue(bounds == (-np.inf, np.inf))
        # lambda > 2: im(f) = (1 / (2 - lambda), inf) without standardization
        pt.lambdas_.fill(3.5)
        bounds = _compute_inverse_bounds({"m2": pt})["m2"]
        self.assertEqual(bounds[1], np.inf)
        # Make sure we got the boundary right
        left = pt.inverse_transform(np.array(bounds[0] - 0.01, ndmin=2))
        right = pt.inverse_transform(np.array(bounds[0] + 0.01, ndmin=2))
        self.assertTrue(not isnan(right) and isnan(left))

    def testMatchCIWidth(self):
        Ys = get_data([self.obsd1, self.obsd2, self.obsd3], ["m2"])
        pt = _compute_power_transforms(Ys)
        pt["m2"].lambdas_.fill(-3.0)
        bounds = _compute_inverse_bounds(pt)["m2"]

        # Both will be NaN since we are far outside the bounds
        new_mean_1, new_var_1 = match_ci_width_truncated(
            mean=bounds[1] + 2.0,
            variance=0.1,
            transform=lambda y: pt["m2"].inverse_transform(np.array(y, ndmin=2)),
            lower_bound=bounds[0],
            upper_bound=bounds[1],
            margin=0.001,
            clip_mean=False,
        )
        # This will be finite since we clip
        new_mean_2, new_var_2 = match_ci_width_truncated(
            mean=bounds[1] + 2.0,
            variance=0.1,
            transform=lambda y: pt["m2"].inverse_transform(np.array(y, ndmin=2)),
            lower_bound=bounds[0],
            upper_bound=bounds[1],
            margin=0.001,
            clip_mean=True,
        )
        self.assertTrue(isnan(new_mean_1) and isnan(new_var_1))
        self.assertTrue(isfinite(new_mean_2) and isfinite(new_var_2))

    def testTransformAndUntransformOneMetric(self):
        observation_data = [deepcopy(self.obsd1), deepcopy(self.obsd2)]
        pt = PowerTransformY(
            search_space=None,
            observation_features=None,
            observation_data=observation_data,
            config={"metrics": ["m1"]},
        )

        # Transform the data and make sure we don't touch m1
        observation_data_tf = pt.transform_observation_data(observation_data, [])
        for obsd, obsd_orig in zip(observation_data_tf, [self.obsd1, self.obsd2]):
            self.assertNotAlmostEqual(obsd.means[0], obsd_orig.means[0])
            self.assertNotAlmostEqual(obsd.covariance[0][0], obsd_orig.covariance[0][0])
            self.assertAlmostEqual(obsd.means[1], obsd_orig.means[1])
            self.assertAlmostEqual(obsd.covariance[1][1], obsd_orig.covariance[1][1])

        # Untransform the data and make sure the means are the same
        observation_data_untf = pt.untransform_observation_data(observation_data_tf, [])
        for obsd, obsd_orig in zip(observation_data_untf, [self.obsd1, self.obsd2]):
            self.assertAlmostEqual(obsd.means[0], obsd_orig.means[0], places=4)
            self.assertAlmostEqual(obsd.means[1], obsd_orig.means[1], places=4)

        # NaN covar values remain as NaNs
        transformed_obsd_nan = pt.transform_observation_data(
            [deepcopy(self.obsd_nan)], []
        )[0]
        cov_results = np.array(transformed_obsd_nan.covariance)
        self.assertTrue(np.all(np.isnan(np.diag(cov_results))))

    def testTransformAndUntransformAllMetrics(self):
        observation_data = [deepcopy(self.obsd1), deepcopy(self.obsd2)]
        pt = PowerTransformY(
            search_space=None,
            observation_features=None,
            observation_data=observation_data,
            config={"metrics": ["m1", "m2"]},
        )

        observation_data_tf = pt.transform_observation_data(observation_data, [])
        for obsd, obsd_orig in zip(observation_data_tf, [self.obsd1, self.obsd2]):
            for i in range(2):  # Both metrics should be transformed
                self.assertNotAlmostEqual(obsd.means[i], obsd_orig.means[i])
                self.assertNotAlmostEqual(
                    obsd.covariance[i][i], obsd_orig.covariance[i][i]
                )

        # Untransform the data and make sure the means are the same
        observation_data_untf = pt.untransform_observation_data(observation_data_tf, [])
        for obsd, obsd_orig in zip(observation_data_untf, [self.obsd1, self.obsd2]):
            for i in range(2):  # Both metrics should be transformed
                self.assertAlmostEqual(obsd.means[i], obsd_orig.means[i])

        # NaN covar values remain as NaNs
        transformed_obsd_nan = pt.transform_observation_data(
            [deepcopy(self.obsd_nan)], []
        )[0]
        cov_results = np.array(transformed_obsd_nan.covariance)
        self.assertTrue(np.all(np.isnan(np.diag(cov_results))))

    def testCompareToSklearn(self):
        # Make sure the transformed values agree with Sklearn
        observation_data = [self.obsd1, self.obsd2, self.obsd3]

        y_orig = np.array([data.means[0] for data in observation_data])[:, None]
        y1 = PowerTransformer("yeo-johnson").fit(y_orig).transform(y_orig).ravel()

        pt = PowerTransformY(
            search_space=None,
            observation_features=None,
            observation_data=deepcopy(observation_data),
            config={"metrics": ["m1"]},
        )
        observation_data_tf = pt.transform_observation_data(observation_data, [])
        y2 = [data.means[0] for data in observation_data_tf]
        for y1_, y2_ in zip(y1, y2):
            self.assertAlmostEqual(y1_, y2_)

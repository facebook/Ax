#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import numpy as np
from ax.core.observation import ObservationData
from ax.modelbridge.transforms.inverse_gaussian_cdf_y import InverseGaussianCdfY
from ax.utils.common.testutils import TestCase


class InverseGaussianCdfYTransformTest(TestCase):
    def setUp(self):
        self.obsd_mid = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([0.5, 0.9]),
            covariance=np.array([[0.005, 0.0], [0.0, 0.005]]),
        )
        self.obsd_extreme = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([0.0, 1.0]),
            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )
        self.obsd_nan_covars = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([0.5, 0.9]),
            covariance=np.array(
                [[float("nan"), float("nan")], [float("nan"), float("nan")]]
            ),
        )
        self.t = InverseGaussianCdfY(
            search_space=None, observation_features=None, observation_data=None
        )

    def testTransformObservations(self):
        transformed_obsd_mid = self.t.transform_observation_data(
            [deepcopy(self.obsd_mid)], []
        )[0]
        # Approximate assertion for robustness.
        mean_results = np.array(list(transformed_obsd_mid.means))
        expected = np.array([0, 1.28155])
        self.assertTrue(
            np.allclose(mean_results, expected),
            msg=f"Mean Results: {mean_results}. Expected {expected}.",
        )

        cov_results = np.array(transformed_obsd_mid.covariance)
        expected = np.array([[0.0327499, 0.0], [0.0, 0.3684465]])
        self.assertTrue(
            np.allclose(cov_results, expected),
            msg=f"Covariance Result: {cov_results}. Expected {expected}.",
        )

        # Fail with extreme values.
        with self.assertRaises(ValueError):
            self.t.transform_observation_data([deepcopy(self.obsd_extreme)], [])[0]

        # NaN covar values remain as NaNs
        transformed_obsd_nan_covars = self.t.transform_observation_data(
            [deepcopy(self.obsd_nan_covars)], []
        )[0]
        cov_results = np.array(transformed_obsd_nan_covars.covariance)
        self.assertTrue(
            np.all(np.isnan(cov_results)),
            msg=f"Unexpected covariance Result: {cov_results}. Expected all nans.",
        )

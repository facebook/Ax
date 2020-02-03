#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import numpy as np
from ax.core.observation import ObservationData
from ax.modelbridge.transforms.percentile_y import PercentileY
from ax.utils.common.testutils import TestCase


class PercentileYTransformTest(TestCase):
    def setUp(self):
        self.obsd1 = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([0.0, 0.0]),
            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )
        self.obsd2 = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([1.0, 5.0]),
            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )
        self.obsd3 = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([2.0, 25.0]),
            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )
        self.obsd4 = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([3.0, 125.0]),
            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )
        self.obsd_mid = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([1.5, 50.0]),
            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )
        self.obsd_extreme = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([-1.0, 126.0]),
            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )
        self.t = PercentileY(
            search_space=None,
            observation_features=None,
            observation_data=[
                deepcopy(self.obsd1),
                deepcopy(self.obsd2),
                deepcopy(self.obsd3),
                deepcopy(self.obsd4),
            ],
        )
        self.t_with_winsorization = PercentileY(
            search_space=None,
            observation_features=None,
            observation_data=[
                deepcopy(self.obsd1),
                deepcopy(self.obsd2),
                deepcopy(self.obsd3),
                deepcopy(self.obsd4),
            ],
            config={"winsorize": True},
        )

    def testInit(self):
        with self.assertRaises(ValueError):
            PercentileY(search_space=None, observation_features=[], observation_data=[])

    def testTransformObservations(self):
        self.assertListEqual(self.t.percentiles["m1"], [0.0, 1.0, 2.0, 3.0])
        self.assertListEqual(self.t.percentiles["m2"], [0.0, 5.0, 25.0, 125.0])
        transformed_obsd_mid = self.t.transform_observation_data(
            [deepcopy(self.obsd_mid)], []
        )[0]
        self.assertListEqual(list(transformed_obsd_mid.means), [0.5, 0.75])
        transformed_obsd_extreme = self.t.transform_observation_data(
            [deepcopy(self.obsd_extreme)], []
        )[0]
        self.assertListEqual(list(transformed_obsd_extreme.means), [0.0, 1.0])

    def testTransformObservationsWithWinsorization(self):
        self.assertListEqual(self.t.percentiles["m1"], [0.0, 1.0, 2.0, 3.0])
        self.assertListEqual(self.t.percentiles["m2"], [0.0, 5.0, 25.0, 125.0])
        transformed_obsd_mid = self.t_with_winsorization.transform_observation_data(
            [deepcopy(self.obsd_mid)], []
        )[0]
        self.assertListEqual(list(transformed_obsd_mid.means), [0.5, 0.75])
        transformed_obsd_extreme = self.t_with_winsorization.transform_observation_data(
            [deepcopy(self.obsd_extreme)], []
        )[0]
        # Approximate assertion for winsorization
        results = np.array(list(transformed_obsd_extreme.means))
        self.assertTrue(all(np.isclose(results, np.array([0.0847075, 0.9152924]))))

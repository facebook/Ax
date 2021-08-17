#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import numpy as np
from ax.core.observation import ObservationData
from ax.modelbridge.transforms.winsorize import Winsorize
from ax.utils.common.testutils import TestCase


class WinsorizeTransformTest(TestCase):
    def setUp(self):
        self.obsd1 = ObservationData(
            metric_names=["m1", "m2", "m2"],
            means=np.array([0.0, 0.0, 1.0]),
            covariance=np.array([[1.0, 0.2, 0.4], [0.2, 2.0, 0.8], [0.4, 0.8, 3.0]]),
        )
        self.obsd2 = ObservationData(
            metric_names=["m1", "m1", "m2", "m2"],
            means=np.array([1.0, 2.0, 2.0, 1.0]),
            covariance=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.2, 0.4],
                    [0.0, 0.2, 2.0, 0.8],
                    [0.0, 0.4, 0.8, 3.0],
                ]
            ),
        )
        self.t = Winsorize(
            search_space=None,
            observation_features=None,
            observation_data=[deepcopy(self.obsd1), deepcopy(self.obsd2)],
            config={"winsorization_upper": 0.2},
        )
        self.t1 = Winsorize(
            search_space=None,
            observation_features=None,
            observation_data=[deepcopy(self.obsd1), deepcopy(self.obsd2)],
            config={"winsorization_upper": 0.8},
        )
        self.t2 = Winsorize(
            search_space=None,
            observation_features=None,
            observation_data=[deepcopy(self.obsd1), deepcopy(self.obsd2)],
            config={"winsorization_lower": 0.2},
        )
        self.t3 = Winsorize(
            search_space=None,
            observation_features=None,
            observation_data=[deepcopy(self.obsd1), deepcopy(self.obsd2)],
            config={
                "winsorization_upper": 0.6,
                "percentile_bounds": {
                    "m1": (None, None),
                    "m2": (None, 1.9),
                },
            },
        )
        self.t4 = Winsorize(
            search_space=None,
            observation_features=None,
            observation_data=[deepcopy(self.obsd1), deepcopy(self.obsd2)],
            config={
                "winsorization_lower": 0.8,
                "percentile_bounds": {
                    "m1": (None, None),
                    "m2": (0.3, None),
                },
            },
        )

        self.obsd3 = ObservationData(
            metric_names=["m3", "m3", "m3", "m3"],
            means=np.array([0.0, 1.0, 5.0, 3.0]),
            covariance=np.eye(4),
        )
        self.t5 = Winsorize(
            search_space=None,
            observation_features=None,
            observation_data=[
                deepcopy(self.obsd1),
                deepcopy(self.obsd2),
                deepcopy(self.obsd3),
            ],
            config={
                "winsorization_lower": {"m2": 0.4},
                "winsorization_upper": {"m1": 0.6},
            },
        )
        self.t6 = Winsorize(
            search_space=None,
            observation_features=None,
            observation_data=[deepcopy(self.obsd1), deepcopy(self.obsd2)],
            config={
                "winsorization_lower": {"m2": 0.4},
                "winsorization_upper": {"m1": 0.6},
                "percentile_bounds": {
                    "m1": (None, None),
                    "m2": (0.0, None),  # This should leave m2 untouched
                },
            },
        )

    def testInit(self):
        self.assertEqual(self.t.percentiles["m1"], (0.0, 2.0))
        self.assertEqual(self.t.percentiles["m2"], (0.0, 2.0))
        self.assertEqual(self.t1.percentiles["m1"], (0.0, 1.0))
        self.assertEqual(self.t1.percentiles["m2"], (0.0, 1.0))
        self.assertEqual(self.t2.percentiles["m1"], (0.0, 2.0))
        self.assertEqual(self.t2.percentiles["m2"], (0.0, 2.0))
        with self.assertRaises(ValueError):
            Winsorize(search_space=None, observation_features=[], observation_data=[])

    def testTransformObservations(self):
        observation_data = self.t1.transform_observation_data(
            [deepcopy(self.obsd1)], []
        )[0]
        self.assertListEqual(list(observation_data.means), [0.0, 0.0, 1.0])
        observation_data = self.t1.transform_observation_data(
            [deepcopy(self.obsd2)], []
        )[0]
        self.assertListEqual(list(observation_data.means), [1.0, 1.0, 1.0, 1.0])
        observation_data = self.t2.transform_observation_data(
            [deepcopy(self.obsd1)], []
        )[0]
        self.assertListEqual(list(observation_data.means), [0.0, 0.0, 1.0])
        observation_data = self.t2.transform_observation_data(
            [deepcopy(self.obsd2)], []
        )[0]
        self.assertListEqual(list(observation_data.means), [1.0, 2.0, 2.0, 1.0])

    def testInitPercentileBounds(self):
        self.assertEqual(self.t3.percentiles["m1"], (0.0, 1.0))
        self.assertEqual(self.t3.percentiles["m2"], (0.0, 1.9))
        self.assertEqual(self.t4.percentiles["m1"], (1.0, 2.0))
        self.assertEqual(self.t4.percentiles["m2"], (0.3, 2.0))

    def testValueError(self):
        with self.assertRaises(ValueError):
            Winsorize(
                search_space=None,
                observation_features=None,
                observation_data=[deepcopy(self.obsd1), deepcopy(self.obsd2)],
                config={
                    "winsorization_lower": 0.8,
                    "percentile_bounds": {"m1": (0.1, 0.2, 0.3)},  # Too many inputs..
                },
            )

    def testTransformObservationsPercentileBounds(self):
        observation_data = self.t3.transform_observation_data(
            [deepcopy(self.obsd1)], []
        )[0]
        self.assertListEqual(list(observation_data.means), [0.0, 0.0, 1.0])
        observation_data = self.t3.transform_observation_data(
            [deepcopy(self.obsd2)], []
        )[0]
        self.assertListEqual(list(observation_data.means), [1.0, 1.0, 1.9, 1.0])
        observation_data = self.t4.transform_observation_data(
            [deepcopy(self.obsd1)], []
        )[0]
        self.assertListEqual(list(observation_data.means), [1.0, 0.3, 1.0])
        observation_data = self.t4.transform_observation_data(
            [deepcopy(self.obsd2)], []
        )[0]
        self.assertListEqual(list(observation_data.means), [1.0, 2.0, 2.0, 1.0])

    def testTransformObservationsDifferentLowerUpper(self):
        observation_data = self.t5.transform_observation_data(
            [deepcopy(self.obsd2)], []
        )[0]
        self.assertEqual(self.t5.percentiles["m1"], (0.0, 1.0))
        self.assertEqual(self.t5.percentiles["m2"], (1.0, 2.0))
        self.assertEqual(self.t5.percentiles["m3"], (0.0, 5.0))
        self.assertListEqual(list(observation_data.means), [1.0, 1.0, 2.0, 1.0])
        # Nothing should happen to m3
        observation_data = self.t5.transform_observation_data(
            [deepcopy(self.obsd3)], []
        )[0]
        self.assertListEqual(list(observation_data.means), [0.0, 1.0, 5.0, 3.0])
        # With percentile_bounds
        observation_data = self.t6.transform_observation_data(
            [deepcopy(self.obsd2)], []
        )[0]
        self.assertEqual(self.t6.percentiles["m1"], (0.0, 1.0))
        self.assertEqual(self.t6.percentiles["m2"], (0.0, 2.0))
        self.assertListEqual(list(observation_data.means), [1.0, 1.0, 2.0, 1.0])

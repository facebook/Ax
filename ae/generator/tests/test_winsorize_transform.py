#!/usr/bin/env python3

from copy import deepcopy

import numpy as np
from ae.lazarus.ae.core.observation import ObservationData
from ae.lazarus.ae.generator.transforms.winsorize import Winsorize
from ae.lazarus.ae.utils.common.testutils import TestCase


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
            config={"winsorization_limits": (None, 0.2)},
        )
        self.t1 = Winsorize(
            search_space=None,
            observation_features=None,
            observation_data=[deepcopy(self.obsd1), deepcopy(self.obsd2)],
            config={"winsorization_limits": (None, 0.8)},
        )
        self.t2 = Winsorize(
            search_space=None,
            observation_features=None,
            observation_data=[deepcopy(self.obsd1), deepcopy(self.obsd2)],
            config={"winsorization_limits": (0.2, None)},
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

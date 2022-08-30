#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from copy import deepcopy

import numpy as np
from ax.core.observation import ObservationData
from ax.exceptions.core import DataRequiredError
from ax.modelbridge.transforms.winsorize import Winsorize
from ax.utils.common.testutils import TestCase


class WinsorizeTransformTestLegacy(TestCase):
    # pyre-fixme[3]: Return type must be annotated.
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
            # pyre-fixme[6]: For 1st param expected `SearchSpace` but got `None`.
            search_space=None,
            # pyre-fixme[6]: For 2nd param expected `List[ObservationFeatures]` but
            #  got `None`.
            observation_features=None,
            observation_data=[deepcopy(self.obsd1), deepcopy(self.obsd2)],
            config={"winsorization_upper": 0.2},
        )
        self.t1 = Winsorize(
            # pyre-fixme[6]: For 1st param expected `SearchSpace` but got `None`.
            search_space=None,
            # pyre-fixme[6]: For 2nd param expected `List[ObservationFeatures]` but
            #  got `None`.
            observation_features=None,
            observation_data=[deepcopy(self.obsd1), deepcopy(self.obsd2)],
            config={"winsorization_upper": 0.8},
        )
        self.t2 = Winsorize(
            # pyre-fixme[6]: For 1st param expected `SearchSpace` but got `None`.
            search_space=None,
            # pyre-fixme[6]: For 2nd param expected `List[ObservationFeatures]` but
            #  got `None`.
            observation_features=None,
            observation_data=[deepcopy(self.obsd1), deepcopy(self.obsd2)],
            config={"winsorization_lower": 0.2},
        )
        self.t3 = Winsorize(
            # pyre-fixme[6]: For 1st param expected `SearchSpace` but got `None`.
            search_space=None,
            # pyre-fixme[6]: For 2nd param expected `List[ObservationFeatures]` but
            #  got `None`.
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
            # pyre-fixme[6]: For 1st param expected `SearchSpace` but got `None`.
            search_space=None,
            # pyre-fixme[6]: For 2nd param expected `List[ObservationFeatures]` but
            #  got `None`.
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
            # pyre-fixme[6]: For 1st param expected `SearchSpace` but got `None`.
            search_space=None,
            # pyre-fixme[6]: For 2nd param expected `List[ObservationFeatures]` but
            #  got `None`.
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
            # pyre-fixme[6]: For 1st param expected `SearchSpace` but got `None`.
            search_space=None,
            # pyre-fixme[6]: For 2nd param expected `List[ObservationFeatures]` but
            #  got `None`.
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

    # pyre-fixme[3]: Return type must be annotated.
    def testPrintDeprecationWarning(self):
        warnings.simplefilter("always", DeprecationWarning)
        with warnings.catch_warnings(record=True) as ws:
            Winsorize(
                # pyre-fixme[6]: For 1st param expected `SearchSpace` but got `None`.
                search_space=None,
                # pyre-fixme[6]: For 2nd param expected `List[ObservationFeatures]`
                #  but got `None`.
                observation_features=None,
                observation_data=[deepcopy(self.obsd1), deepcopy(self.obsd2)],
                config={"winsorization_upper": 0.2},
            )
            self.assertTrue(
                "Winsorization received an out-of-date `transform_config`, containing "
                "the following deprecated keys: {'winsorization_upper'}. Please "
                "update the config according to the docs of "
                "`ax.modelbridge.transforms.winsorize.Winsorize`."
                in [str(w.message) for w in ws]
            )

    # pyre-fixme[3]: Return type must be annotated.
    def testInit(self):
        self.assertEqual(self.t.cutoffs["m1"], (-float("inf"), 2.0))
        self.assertEqual(self.t.cutoffs["m2"], (-float("inf"), 2.0))
        self.assertEqual(self.t1.cutoffs["m1"], (-float("inf"), 1.0))
        self.assertEqual(self.t1.cutoffs["m2"], (-float("inf"), 1.0))
        self.assertEqual(self.t2.cutoffs["m1"], (0.0, float("inf")))
        self.assertEqual(self.t2.cutoffs["m2"], (0.0, float("inf")))
        with self.assertRaises(DataRequiredError):
            # pyre-fixme[6]: For 1st param expected `SearchSpace` but got `None`.
            Winsorize(search_space=None, observation_features=[], observation_data=[])

    # pyre-fixme[3]: Return type must be annotated.
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

    # pyre-fixme[3]: Return type must be annotated.
    def testInitPercentileBounds(self):
        self.assertEqual(self.t3.cutoffs["m1"], (-float("inf"), 1.0))
        self.assertEqual(self.t3.cutoffs["m2"], (-float("inf"), 1.9))
        self.assertEqual(self.t4.cutoffs["m1"], (1.0, float("inf")))
        self.assertEqual(self.t4.cutoffs["m2"], (0.3, float("inf")))

    # pyre-fixme[3]: Return type must be annotated.
    def testValueError(self):
        with self.assertRaises(ValueError):
            Winsorize(
                # pyre-fixme[6]: For 1st param expected `SearchSpace` but got `None`.
                search_space=None,
                # pyre-fixme[6]: For 2nd param expected `List[ObservationFeatures]`
                #  but got `None`.
                observation_features=None,
                observation_data=[deepcopy(self.obsd1), deepcopy(self.obsd2)],
                config={
                    "winsorization_lower": 0.8,
                    "percentile_bounds": {"m1": (0.1, 0.2, 0.3)},  # Too many inputs..
                },
            )

    # pyre-fixme[3]: Return type must be annotated.
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

    # pyre-fixme[3]: Return type must be annotated.
    def testTransformObservationsDifferentLowerUpper(self):
        observation_data = self.t5.transform_observation_data(
            [deepcopy(self.obsd2)], []
        )[0]
        self.assertEqual(self.t5.cutoffs["m1"], (-float("inf"), 1.0))
        self.assertEqual(self.t5.cutoffs["m2"], (1.0, float("inf")))
        self.assertEqual(self.t5.cutoffs["m3"], (-float("inf"), float("inf")))
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
        self.assertEqual(self.t6.cutoffs["m1"], (-float("inf"), 1.0))
        self.assertEqual(self.t6.cutoffs["m2"], (0.0, float("inf")))
        self.assertListEqual(list(observation_data.means), [1.0, 1.0, 2.0, 1.0])

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from unittest.mock import Mock, PropertyMock

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    observations_from_data,
)
from ax.utils.common.testutils import TestCase


class ObservationsTest(TestCase):
    def testObservationFeatures(self):
        t = np.datetime64("now")
        attrs = {
            "parameters": {"x": 0, "y": "a"},
            "trial_index": 2,
            "start_time": t,
            "end_time": t,
            "random_split": 1,
        }
        obsf = ObservationFeatures(**attrs)
        for k, v in attrs.items():
            self.assertEqual(getattr(obsf, k), v)
        printstr = "ObservationFeatures(parameters={'x': 0, 'y': 'a'}, "
        printstr += "trial_index=2, "
        printstr += "start_time={t}, end_time={t}, ".format(t=t)
        printstr += "random_split=1)"
        self.assertEqual(repr(obsf), printstr)
        obsf2 = ObservationFeatures(**attrs)
        self.assertEqual(hash(obsf), hash(obsf2))
        a = {obsf, obsf2}
        self.assertEqual(len(a), 1)
        self.assertEqual(obsf, obsf2)
        attrs.pop("trial_index")
        obsf3 = ObservationFeatures(**attrs)
        self.assertNotEqual(obsf, obsf3)
        self.assertFalse(obsf == 1)

    def testObservationFeaturesFromArm(self):
        arm = Arm({"x": 0, "y": "a"})
        obsf = ObservationFeatures.from_arm(arm, trial_index=3)
        self.assertEqual(obsf.parameters, arm.parameters)
        self.assertEqual(obsf.trial_index, 3)

    def testObservationData(self):
        attrs = {
            "metric_names": ["a", "b"],
            "means": np.array([4.0, 5.0]),
            "covariance": np.array([[1.0, 4.0], [3.0, 6.0]]),
        }
        obsd = ObservationData(**attrs)
        self.assertEqual(obsd.metric_names, attrs["metric_names"])
        self.assertTrue(np.array_equal(obsd.means, attrs["means"]))
        self.assertTrue(np.array_equal(obsd.covariance, attrs["covariance"]))
        # use legacy printing for numpy (<= 1.13 add spaces in front of floats;
        # to get around tests failing on older versions, peg version to 1.13)
        if np.__version__ >= "1.14":
            np.set_printoptions(legacy="1.13")
        printstr = "ObservationData(metric_names=['a', 'b'], means=[ 4.  5.], "
        printstr += "covariance=[[ 1.  4.]\n [ 3.  6.]])"
        self.assertEqual(repr(obsd), printstr)

    def testObservationDataValidation(self):
        with self.assertRaises(ValueError):
            ObservationData(
                metric_names=["a", "b"],
                means=np.array([4.0]),
                covariance=np.array([[1.0, 4.0], [3.0, 6.0]]),
            )
        with self.assertRaises(ValueError):
            ObservationData(
                metric_names=["a", "b"],
                means=np.array([4.0, 5.0]),
                covariance=np.array([1.0, 4.0]),
            )

    def testObservationDataEq(self):
        od1 = ObservationData(
            metric_names=["a", "b"],
            means=np.array([4.0, 5.0]),
            covariance=np.array([[1.0, 4.0], [3.0, 6.0]]),
        )
        od2 = ObservationData(
            metric_names=["a", "b"],
            means=np.array([4.0, 5.0]),
            covariance=np.array([[1.0, 4.0], [3.0, 6.0]]),
        )
        od3 = ObservationData(
            metric_names=["a", "b"],
            means=np.array([4.0, 5.0]),
            covariance=np.array([[2.0, 4.0], [3.0, 6.0]]),
        )
        self.assertEqual(od1, od2)
        self.assertNotEqual(od1, od3)
        self.assertFalse(od1 == 1)

    def testObservation(self):
        obs = Observation(
            features=ObservationFeatures(parameters={"x": 20}),
            data=ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
            arm_name="0_0",
        )
        self.assertEqual(obs.features, ObservationFeatures(parameters={"x": 20}))
        self.assertEqual(
            obs.data,
            ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
        )
        self.assertEqual(obs.arm_name, "0_0")
        obs2 = Observation(
            features=ObservationFeatures(parameters={"x": 20}),
            data=ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
            arm_name="0_0",
        )
        self.assertEqual(obs, obs2)
        obs3 = Observation(
            features=ObservationFeatures(parameters={"x": 10}),
            data=ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
            arm_name="0_0",
        )
        self.assertNotEqual(obs, obs3)
        self.assertNotEqual(obs, 1)

    def testObservationsFromData(self):
        truth = [
            {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a"},
                "mean": 2.0,
                "sem": 2.0,
                "trial_index": 1,
                "metric_name": "a",
            },
            {
                "arm_name": "0_1",
                "parameters": {"x": 1, "y": "b"},
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 2,
                "metric_name": "a",
            },
            {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a"},
                "mean": 4.0,
                "sem": 4.0,
                "trial_index": 1,
                "metric_name": "b",
            },
        ]
        arms = {obs["arm_name"]: Arm(parameters=obs["parameters"]) for obs in truth}
        experiment = Mock()
        type(experiment).arms_by_name = PropertyMock(return_value=arms)

        df = pd.DataFrame(truth)[
            ["arm_name", "trial_index", "mean", "sem", "metric_name"]
        ]
        data = Data(df=df)
        observations = observations_from_data(experiment, data)

        self.assertEqual(len(observations), 2)
        # Get them in the order we want for tests below
        if observations[0].features.parameters["x"] == 1:
            observations.reverse()

        obsd_truth = {
            "metric_names": [["a", "b"], ["a"]],
            "means": [np.array([2.0, 4.0]), np.array([3])],
            "covariance": [np.diag([4.0, 16.0]), np.array([[9.0]])],
        }
        cname_truth = ["0_0", "0_1"]

        for i, obs in enumerate(observations):
            self.assertEqual(obs.features.parameters, truth[i]["parameters"])
            self.assertEqual(obs.features.trial_index, truth[i]["trial_index"])
            self.assertEqual(obs.data.metric_names, obsd_truth["metric_names"][i])
            self.assertTrue(np.array_equal(obs.data.means, obsd_truth["means"][i]))
            self.assertTrue(
                np.array_equal(obs.data.covariance, obsd_truth["covariance"][i])
            )
            self.assertEqual(obs.arm_name, cname_truth[i])

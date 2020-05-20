#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from unittest.mock import Mock, PropertyMock

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun
from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    observations_from_data,
    separate_observations,
)
from ax.core.trial import Trial
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

    def testUpdateFeatures(self):
        parameters = {"x": 0, "y": "a"}
        new_parameters = {"z": "foo"}

        obsf = ObservationFeatures(parameters=parameters, trial_index=3)

        # Ensure None trial_index doesn't override existing value
        obsf.update_features(ObservationFeatures(parameters={}))
        self.assertEqual(obsf.trial_index, 3)

        # Test override
        new_obsf = ObservationFeatures(
            parameters=new_parameters,
            trial_index=4,
            start_time=pd.Timestamp("2005-02-25"),
            end_time=pd.Timestamp("2005-02-26"),
            random_split=7,
        )
        obsf.update_features(new_obsf)
        self.assertEqual(obsf.parameters, {**parameters, **new_parameters})
        self.assertEqual(obsf.trial_index, 4)
        self.assertEqual(obsf.random_split, 7)
        self.assertEqual(obsf.start_time, pd.Timestamp("2005-02-25"))
        self.assertEqual(obsf.end_time, pd.Timestamp("2005-02-26"))

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
        arms = {
            obs["arm_name"]: Arm(name=obs["arm_name"], parameters=obs["parameters"])
            for obs in truth
        }
        experiment = Mock()
        experiment._trial_indices_by_status = {status: set() for status in TrialStatus}
        trials = {
            obs["trial_index"]: Trial(
                experiment, GeneratorRun(arms=[arms[obs["arm_name"]]])
            )
            for obs in truth
        }
        type(experiment).arms_by_name = PropertyMock(return_value=arms)
        type(experiment).trials = PropertyMock(return_value=trials)

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

    def testObservationsFromDataWithFidelities(self):
        truth = {
            0.5: {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a", "z": 1},
                "mean": 2.0,
                "sem": 2.0,
                "trial_index": 1,
                "metric_name": "a",
                "fidelities": json.dumps({"z": 0.5}),
                "updated_parameters": {"x": 0, "y": "a", "z": 0.5},
                "mean_t": np.array([2.0]),
                "covariance_t": np.array([[4.0]]),
            },
            0.25: {
                "arm_name": "0_1",
                "parameters": {"x": 1, "y": "b", "z": 0.5},
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 2,
                "metric_name": "a",
                "fidelities": json.dumps({"z": 0.25}),
                "updated_parameters": {"x": 1, "y": "b", "z": 0.25},
                "mean_t": np.array([3.0]),
                "covariance_t": np.array([[9.0]]),
            },
            1: {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a", "z": 1},
                "mean": 4.0,
                "sem": 4.0,
                "trial_index": 1,
                "metric_name": "b",
                "fidelities": json.dumps({"z": 1}),
                "updated_parameters": {"x": 0, "y": "a", "z": 1},
                "mean_t": np.array([4.0]),
                "covariance_t": np.array([[16.0]]),
            },
        }
        arms = {
            obs["arm_name"]: Arm(name=obs["arm_name"], parameters=obs["parameters"])
            for _, obs in truth.items()
        }
        experiment = Mock()
        experiment._trial_indices_by_status = {status: set() for status in TrialStatus}
        trials = {
            obs["trial_index"]: Trial(
                experiment, GeneratorRun(arms=[arms[obs["arm_name"]]])
            )
            for _, obs in truth.items()
        }
        type(experiment).arms_by_name = PropertyMock(return_value=arms)
        type(experiment).trials = PropertyMock(return_value=trials)

        df = pd.DataFrame(list(truth.values()))[
            ["arm_name", "trial_index", "mean", "sem", "metric_name", "fidelities"]
        ]
        data = Data(df=df)
        observations = observations_from_data(experiment, data)

        self.assertEqual(len(observations), 3)
        for obs in observations:
            t = truth[obs.features.parameters["z"]]
            self.assertEqual(obs.features.parameters, t["updated_parameters"])
            self.assertEqual(obs.features.trial_index, t["trial_index"])
            self.assertEqual(obs.data.metric_names, [t["metric_name"]])
            self.assertTrue(np.array_equal(obs.data.means, t["mean_t"]))
            self.assertTrue(np.array_equal(obs.data.covariance, t["covariance_t"]))
            self.assertEqual(obs.arm_name, t["arm_name"])

    def testObservationsFromDataWithSomeMissingTimes(self):
        truth = [
            {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a"},
                "mean": 2.0,
                "sem": 2.0,
                "trial_index": 1,
                "metric_name": "a",
                "start_time": 0,
            },
            {
                "arm_name": "0_1",
                "parameters": {"x": 1, "y": "b"},
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 2,
                "metric_name": "a",
                "start_time": 0,
            },
            {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a"},
                "mean": 4.0,
                "sem": 4.0,
                "trial_index": 1,
                "metric_name": "b",
                "start_time": None,
            },
            {
                "arm_name": "0_1",
                "parameters": {"x": 1, "y": "b"},
                "mean": 5.0,
                "sem": 5.0,
                "trial_index": 2,
                "metric_name": "b",
                "start_time": None,
            },
        ]
        arms = {
            obs["arm_name"]: Arm(name=obs["arm_name"], parameters=obs["parameters"])
            for obs in truth
        }
        experiment = Mock()
        experiment._trial_indices_by_status = {status: set() for status in TrialStatus}
        trials = {
            obs["trial_index"]: Trial(
                experiment, GeneratorRun(arms=[arms[obs["arm_name"]]])
            )
            for obs in truth
        }
        type(experiment).arms_by_name = PropertyMock(return_value=arms)
        type(experiment).trials = PropertyMock(return_value=trials)

        df = pd.DataFrame(truth)[
            ["arm_name", "trial_index", "mean", "sem", "metric_name", "start_time"]
        ]
        data = Data(df=df)
        observations = observations_from_data(experiment, data)

        self.assertEqual(len(observations), 4)
        # Get them in the order we want for tests below
        if observations[0].features.parameters["x"] == 1:
            observations.reverse()

        obsd_truth = {
            "metric_names": [["a"], ["a"], ["b"], ["b"]],
            "means": [
                np.array([2.0]),
                np.array([3.0]),
                np.array([4.0]),
                np.array([5.0]),
            ],
            "covariance": [
                np.diag([4.0]),
                np.diag([9.0]),
                np.diag([16.0]),
                np.diag([25.0]),
            ],
        }
        cname_truth = ["0_0", "0_1", "0_0", "0_1"]

        for i, obs in enumerate(observations):
            self.assertEqual(obs.features.parameters, truth[i]["parameters"])
            self.assertEqual(obs.features.trial_index, truth[i]["trial_index"])
            self.assertEqual(obs.data.metric_names, obsd_truth["metric_names"][i])
            self.assertTrue(np.array_equal(obs.data.means, obsd_truth["means"][i]))
            self.assertTrue(
                np.array_equal(obs.data.covariance, obsd_truth["covariance"][i])
            )
            self.assertEqual(obs.arm_name, cname_truth[i])

    def testSeparateObservations(self):
        obs = Observation(
            features=ObservationFeatures(parameters={"x": 20}),
            data=ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
            arm_name="0_0",
        )
        obs_feats, obs_data = separate_observations(observations=[obs])
        self.assertEqual(obs.features, ObservationFeatures(parameters={"x": 20}))
        self.assertEqual(
            obs.data,
            ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
        )
        obs_feats, obs_data = separate_observations(observations=[obs], copy=True)
        self.assertEqual(obs.features, ObservationFeatures(parameters={"x": 20}))
        self.assertEqual(
            obs.data,
            ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
        )

    def testObservationsWithCandidateMetadata(self):
        SOME_METADATA_KEY = "metadatum"
        truth = [
            {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a"},
                "mean": 2.0,
                "sem": 2.0,
                "trial_index": 0,
                "metric_name": "a",
            },
            {
                "arm_name": "1_0",
                "parameters": {"x": 1, "y": "b"},
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 1,
                "metric_name": "a",
            },
        ]
        arms = {
            obs["arm_name"]: Arm(name=obs["arm_name"], parameters=obs["parameters"])
            for obs in truth
        }
        experiment = Mock()
        experiment._trial_indices_by_status = {status: set() for status in TrialStatus}
        trials = {
            obs["trial_index"]: Trial(
                experiment,
                GeneratorRun(
                    arms=[arms[obs["arm_name"]]],
                    candidate_metadata_by_arm_signature={
                        arms[obs["arm_name"]].signature: {
                            SOME_METADATA_KEY: f"value_{obs['trial_index']}"
                        }
                    },
                ),
            )
            for obs in truth
        }
        type(experiment).arms_by_name = PropertyMock(return_value=arms)
        type(experiment).trials = PropertyMock(return_value=trials)

        df = pd.DataFrame(truth)[
            ["arm_name", "trial_index", "mean", "sem", "metric_name"]
        ]
        data = Data(df=df)
        observations = observations_from_data(experiment, data)
        for observation in observations:
            self.assertEqual(
                observation.features.metadata.get(SOME_METADATA_KEY),
                f"value_{observation.features.trial_index}",
            )

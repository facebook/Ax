#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from unittest.mock import Mock, PropertyMock

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun
from ax.core.map_data import MapData, MapKeyInfo
from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    observations_from_data,
    observations_from_map_data,
    recombine_observations,
    separate_observations,
)
from ax.core.trial import Trial
from ax.utils.common.testutils import TestCase


class ObservationsTest(TestCase):
    def testObservationFeatures(self) -> None:
        t = np.datetime64("now")
        attrs = {
            "parameters": {"x": 0, "y": "a"},
            "trial_index": 2,
            "start_time": t,
            "end_time": t,
            "random_split": 1,
        }
        # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool, float,
        #  int, str]]` but got `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[Dict[str, typing.Any]]`
        #  but got `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[int64]` but got
        #  `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[Timestamp]` but got
        #  `Union[Dict[str, Union[int, str]], int, datetime64]`.
        obsf = ObservationFeatures(**attrs)
        for k, v in attrs.items():
            self.assertEqual(getattr(obsf, k), v)
        printstr = "ObservationFeatures(parameters={'x': 0, 'y': 'a'}, "
        printstr += "trial_index=2, "
        printstr += "start_time={t}, end_time={t}, ".format(t=t)
        printstr += "random_split=1)"
        self.assertEqual(repr(obsf), printstr)
        # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool, float,
        #  int, str]]` but got `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[Dict[str, typing.Any]]`
        #  but got `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[int64]` but got
        #  `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[Timestamp]` but got
        #  `Union[Dict[str, Union[int, str]], int, datetime64]`.
        obsf2 = ObservationFeatures(**attrs)
        self.assertEqual(hash(obsf), hash(obsf2))
        a = {obsf, obsf2}
        self.assertEqual(len(a), 1)
        self.assertEqual(obsf, obsf2)
        attrs.pop("trial_index")
        # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool, float,
        #  int, str]]` but got `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[Dict[str, typing.Any]]`
        #  but got `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[int64]` but got
        #  `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[Timestamp]` but got
        #  `Union[Dict[str, Union[int, str]], int, datetime64]`.
        obsf3 = ObservationFeatures(**attrs)
        self.assertNotEqual(obsf, obsf3)
        self.assertFalse(obsf == 1)

    def testClone(self) -> None:
        # Test simple cloning.
        arm = Arm({"x": 0, "y": "a"})
        # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
        obsf = ObservationFeatures.from_arm(arm, trial_index=3)
        self.assertIsNot(obsf, obsf.clone())
        self.assertEqual(obsf, obsf.clone())

        # Test cloning with swapping parameters.
        clone_with_new_params = obsf.clone(replace_parameters={"x": 1, "y": "b"})
        self.assertNotEqual(obsf, clone_with_new_params)
        obsf.parameters = {"x": 1, "y": "b"}
        self.assertEqual(obsf, clone_with_new_params)

    def testObservationFeaturesFromArm(self) -> None:
        arm = Arm({"x": 0, "y": "a"})
        # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
        obsf = ObservationFeatures.from_arm(arm, trial_index=3)
        self.assertEqual(obsf.parameters, arm.parameters)
        self.assertEqual(obsf.trial_index, 3)

    def testUpdateFeatures(self) -> None:
        parameters = {"x": 0, "y": "a"}
        new_parameters = {"z": "foo"}

        # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool, float,
        #  int, str]]` but got `Dict[str, Union[int, str]]`.
        # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
        obsf = ObservationFeatures(parameters=parameters, trial_index=3)

        # Ensure None trial_index doesn't override existing value
        obsf.update_features(ObservationFeatures(parameters={}))
        self.assertEqual(obsf.trial_index, 3)

        # Test override
        new_obsf = ObservationFeatures(
            # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Dict[str, str]`.
            parameters=new_parameters,
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            trial_index=4,
            start_time=pd.Timestamp("2005-02-25"),
            end_time=pd.Timestamp("2005-02-26"),
            # pyre-fixme[6]: For 5th param expected `Optional[int64]` but got `int`.
            random_split=7,
        )
        obsf.update_features(new_obsf)
        self.assertEqual(obsf.parameters, {**parameters, **new_parameters})
        self.assertEqual(obsf.trial_index, 4)
        self.assertEqual(obsf.random_split, 7)
        self.assertEqual(obsf.start_time, pd.Timestamp("2005-02-25"))
        self.assertEqual(obsf.end_time, pd.Timestamp("2005-02-26"))

    def testObservationData(self) -> None:
        attrs = {
            "metric_names": ["a", "b"],
            "means": np.array([4.0, 5.0]),
            "covariance": np.array([[1.0, 4.0], [3.0, 6.0]]),
        }
        # pyre-fixme[6]: For 1st param expected `List[str]` but got
        #  `Union[List[str], ndarray]`.
        # pyre-fixme[6]: For 1st param expected `ndarray` but got `Union[List[str],
        #  ndarray]`.
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
        self.assertEqual(obsd.means_dict, {"a": 4.0, "b": 5.0})
        self.assertEqual(
            obsd.covariance_matrix,
            {"a": {"a": 1.0, "b": 4.0}, "b": {"a": 3.0, "b": 6.0}},
        )

    def testObservationDataValidation(self) -> None:
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

    def testObservationDataEq(self) -> None:
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

    def testObservation(self) -> None:
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

    def testObservationsFromData(self) -> None:
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
            # pyre-fixme[6]: For 1st param expected `Optional[str]` but got
            #  `Union[Dict[str, Union[int, str]], float, str]`.
            # pyre-fixme[6]: For 2nd param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Union[Dict[str, Union[int, str]], float,
            #  str]`.
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

    def testObservationsFromDataWithFidelities(self) -> None:
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
            # pyre-fixme[6]: For 1st param expected `Optional[str]` but got
            #  `Union[Dict[str, Union[float, str]], Dict[str, Union[int, str]], float,
            #  ndarray, str]`.
            # pyre-fixme[6]: For 2nd param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Union[Dict[str, Union[float, str]],
            #  Dict[str, Union[int, str]], float, ndarray, str]`.
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
            # pyre-fixme[6]: For 1st param expected `float` but got `Union[None,
            #  bool, float, int, str]`.
            t = truth[obs.features.parameters["z"]]
            self.assertEqual(obs.features.parameters, t["updated_parameters"])
            self.assertEqual(obs.features.trial_index, t["trial_index"])
            self.assertEqual(obs.data.metric_names, [t["metric_name"]])
            self.assertTrue(np.array_equal(obs.data.means, t["mean_t"]))
            self.assertTrue(np.array_equal(obs.data.covariance, t["covariance_t"]))
            self.assertEqual(obs.arm_name, t["arm_name"])

    def testObservationsFromMapData(self) -> None:
        truth = {
            0.5: {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a", "z": 1},
                "mean": 2.0,
                "sem": 2.0,
                "trial_index": 1,
                "metric_name": "a",
                "updated_parameters": {"x": 0, "y": "a", "z": 0.5},
                "mean_t": np.array([2.0]),
                "covariance_t": np.array([[4.0]]),
                "z": 0.5,
                "timestamp": 50,
            },
            0.25: {
                "arm_name": "0_1",
                "parameters": {"x": 1, "y": "b", "z": 0.5},
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 2,
                "metric_name": "a",
                "updated_parameters": {"x": 1, "y": "b", "z": 0.25},
                "mean_t": np.array([3.0]),
                "covariance_t": np.array([[9.0]]),
                "z": 0.25,
                "timestamp": 25,
            },
            1: {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a", "z": 1},
                "mean": 4.0,
                "sem": 4.0,
                "trial_index": 1,
                "metric_name": "b",
                "updated_parameters": {"x": 0, "y": "a", "z": 1},
                "mean_t": np.array([4.0]),
                "covariance_t": np.array([[16.0]]),
                "z": 1,
                "timestamp": 100,
            },
        }
        arms = {
            # pyre-fixme[6]: For 1st param expected `Optional[str]` but got
            #  `Union[Dict[str, Union[float, str]], Dict[str, Union[int, str]], float,
            #  ndarray, str]`.
            # pyre-fixme[6]: For 2nd param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Union[Dict[str, Union[float, str]],
            #  Dict[str, Union[int, str]], float, ndarray, str]`.
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
            ["arm_name", "trial_index", "mean", "sem", "metric_name", "z", "timestamp"]
        ]
        data = MapData(
            df=df,
            map_key_infos=[
                MapKeyInfo(key="z", default_value=0.0),
                MapKeyInfo(key="timestamp", default_value=0.0),
            ],
        )
        observations = observations_from_map_data(experiment, data)

        self.assertEqual(len(observations), 3)

        for obs in observations:
            # pyre-fixme[6]: For 1st param expected `float` but got `Union[None,
            #  bool, float, int, str]`.
            t = truth[obs.features.parameters["z"]]
            self.assertEqual(obs.features.parameters, t["updated_parameters"])
            self.assertEqual(obs.features.trial_index, t["trial_index"])
            self.assertEqual(obs.data.metric_names, [t["metric_name"]])
            self.assertTrue(np.array_equal(obs.data.means, t["mean_t"]))
            self.assertTrue(np.array_equal(obs.data.covariance, t["covariance_t"]))
            self.assertEqual(obs.arm_name, t["arm_name"])
            self.assertEqual(obs.features.metadata, {"timestamp": t["timestamp"]})

    def testObservationsFromDataAbandoned(self) -> None:
        truth = {
            0.5: {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a", "z": 1},
                "mean": 2.0,
                "sem": 2.0,
                "trial_index": 0,
                "metric_name": "a",
                "updated_parameters": {"x": 0, "y": "a", "z": 0.5},
                "mean_t": np.array([2.0]),
                "covariance_t": np.array([[4.0]]),
                "z": 0.5,
                "timestamp": 50,
            },
            1: {
                "arm_name": "1_0",
                "parameters": {"x": 0, "y": "a", "z": 1},
                "mean": 4.0,
                "sem": 4.0,
                "trial_index": 1,
                "metric_name": "b",
                "updated_parameters": {"x": 0, "y": "a", "z": 1},
                "mean_t": np.array([4.0]),
                "covariance_t": np.array([[16.0]]),
                "z": 1,
                "timestamp": 100,
            },
            0.25: {
                "arm_name": "2_0",
                "parameters": {"x": 1, "y": "a", "z": 0.5},
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 2,
                "metric_name": "a",
                "updated_parameters": {"x": 1, "y": "b", "z": 0.25},
                "mean_t": np.array([3.0]),
                "covariance_t": np.array([[9.0]]),
                "z": 0.25,
                "timestamp": 25,
            },
            0.75: {
                "arm_name": "2_1",
                "parameters": {"x": 1, "y": "b", "z": 0.75},
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 2,
                "metric_name": "a",
                "updated_parameters": {"x": 1, "y": "b", "z": 0.75},
                "mean_t": np.array([3.0]),
                "covariance_t": np.array([[9.0]]),
                "z": 0.75,
                "timestamp": 25,
            },
        }
        arms = {
            # pyre-fixme[6]: For 1st param expected `Optional[str]` but got
            #  `Union[Dict[str, Union[float, str]], Dict[str, Union[int, str]], float,
            #  ndarray, str]`.
            # pyre-fixme[6]: For 2nd param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Union[Dict[str, Union[float, str]],
            #  Dict[str, Union[int, str]], float, ndarray, str]`.
            obs["arm_name"]: Arm(name=obs["arm_name"], parameters=obs["parameters"])
            for _, obs in truth.items()
        }
        experiment = Mock()
        experiment._trial_indices_by_status = {status: set() for status in TrialStatus}
        trials = {
            obs["trial_index"]: (
                Trial(experiment, GeneratorRun(arms=[arms[obs["arm_name"]]]))
            )
            for _, obs in list(truth.items())[:-1]
            # pyre-fixme[16]: Item `Dict` of `Union[Dict[str, typing.Union[float,
            #  str]], Dict[str, typing.Union[int, str]], float, ndarray, str]` has no
            #  attribute `startswith`.
            if not obs["arm_name"].startswith("2")
        }
        batch = BatchTrial(experiment, GeneratorRun(arms=[arms["2_0"], arms["2_1"]]))
        # pyre-fixme[6]: For 1st param expected
        #  `SupportsKeysAndGetItem[Union[Dict[str, Union[float, str]], Dict[str,
        #  Union[int, str]], float, ndarray, str], Trial]` but got `Dict[int,
        #  BatchTrial]`.
        trials.update({2: batch})
        # pyre-fixme[16]: Optional type has no attribute `mark_abandoned`.
        trials.get(1).mark_abandoned()
        # pyre-fixme[16]: Optional type has no attribute `mark_arm_abandoned`.
        trials.get(2).mark_arm_abandoned(arm_name="2_1")
        type(experiment).arms_by_name = PropertyMock(return_value=arms)
        type(experiment).trials = PropertyMock(return_value=trials)

        df = pd.DataFrame(list(truth.values()))[
            ["arm_name", "trial_index", "mean", "sem", "metric_name"]
        ]
        data = Data(df=df)

        # 1 arm is abandoned and 1 trial is abandoned, so only 2 observations should be
        # included.
        obs_no_abandoned = observations_from_data(experiment, data)
        self.assertEqual(len(obs_no_abandoned), 2)

        # 1 arm is abandoned and 1 trial is abandoned, so only 2 observations should be
        # included.
        obs_with_abandoned = observations_from_data(
            experiment, data, include_abandoned=True
        )
        self.assertEqual(len(obs_with_abandoned), 4)

    def testObservationsFromDataWithSomeMissingTimes(self) -> None:
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
            # pyre-fixme[6]: For 1st param expected `Optional[str]` but got
            #  `Union[None, Dict[str, Union[int, str]], float, str]`.
            # pyre-fixme[6]: For 2nd param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Union[None, Dict[str, Union[int, str]],
            #  float, str]`.
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

        self.assertEqual(len(observations), 2)
        # Get them in the order we want for tests below
        if observations[0].features.parameters["x"] == 1:
            observations.reverse()

        obsd_truth = {
            "metric_names": [["a", "b"], ["a", "b"]],
            "means": [np.array([2.0, 4.0]), np.array([3.0, 5.0])],
            "covariance": [np.diag([4.0, 16.0]), np.diag([9.0, 25.0])],
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

    def testSeparateObservations(self) -> None:
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
        with self.assertRaises(ValueError):
            recombine_observations(observation_features=obs_feats, observation_data=[])
        new_obs = recombine_observations(obs_feats, obs_data)[0]
        self.assertEqual(new_obs.features, obs.features)
        self.assertEqual(new_obs.data, obs.data)
        obs_feats, obs_data = separate_observations(observations=[obs], copy=True)
        self.assertEqual(obs.features, ObservationFeatures(parameters={"x": 20}))
        self.assertEqual(
            obs.data,
            ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
        )

    def testObservationsWithCandidateMetadata(self) -> None:
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
            # pyre-fixme[6]: For 1st param expected `Optional[str]` but got
            #  `Union[Dict[str, Union[int, str]], float, str]`.
            # pyre-fixme[6]: For 2nd param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Union[Dict[str, Union[int, str]], float,
            #  str]`.
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
                # pyre-fixme[16]: Optional type has no attribute `get`.
                observation.features.metadata.get(SOME_METADATA_KEY),
                f"value_{observation.features.trial_index}",
            )

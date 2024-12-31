#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
from copy import deepcopy
from unittest import mock
from unittest.mock import Mock, patch

import pandas as pd
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun, GeneratorRunType
from ax.core.runner import Runner
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.result import Ok
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_arms,
    get_branin_arms,
    get_experiment,
    get_objective,
    get_test_map_data_experiment,
)

TEST_DATA = Data(
    df=pd.DataFrame(
        [
            {
                "arm_name": "0_0",
                "metric_name": get_objective().metric.name,
                "mean": 1.0,
                "sem": 2.0,
                "trial_index": 0,
            }
        ]
    )
)


class TrialTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.mock_supports_trial_type = mock.patch(
            f"{get_experiment.__module__}.Experiment.supports_trial_type",
            return_value=True,
        )
        self.mock_supports_trial_type.start()
        self.experiment = get_experiment()
        self.trial = self.experiment.new_trial(ttl_seconds=123, trial_type="foo")
        self.trial.update_run_metadata(metadata={"foo": "bar"})
        self.trial.update_stop_metadata(metadata={"bar": "baz"})
        self.arm = get_arms()[0]
        self.trial.add_arm(self.arm)

    def tearDown(self) -> None:
        self.mock_supports_trial_type.stop()

    def test__validate_can_attach_data(self) -> None:
        self.trial.mark_running(no_runner_required=True)
        self.trial.mark_completed()

        expected_msg = (
            "Trial 0 has already been completed with data. To add more data to "
        )
        with self.assertRaisesRegex(UnsupportedError, expected_msg):
            self.trial._validate_can_attach_data()

    def test_eq(self) -> None:
        new_trial = self.experiment.new_trial()
        self.assertNotEqual(self.trial, new_trial)

    def test_basic_properties(self) -> None:
        self.assertEqual(self.experiment, self.trial.experiment)
        self.assertEqual(self.trial.index, 0)
        self.assertEqual(self.trial.status, TrialStatus.CANDIDATE)
        self.assertTrue(self.trial.status.is_candidate)
        self.assertIsNotNone(self.trial.time_created)
        self.assertEqual(self.trial.arms_by_name["0_0"], self.trial.arm)
        self.assertEqual(self.trial.arms[0].signature, self.arm.signature)
        self.assertEqual(self.trial.abandoned_arms, [])
        self.assertEqual(
            self.trial.generator_run.generator_run_type, GeneratorRunType.MANUAL.name
        )

        # Test empty arms
        with self.assertRaises(AttributeError):
            self.experiment.new_trial().arm_weights

        self.trial.mark_running(no_runner_required=True)
        self.assertTrue(self.trial.status.is_running)

        self.trial._status = TrialStatus.COMPLETED
        self.assertEqual(str(self.trial._status), "TrialStatus.COMPLETED")
        self.assertTrue(self.trial.status.is_completed)
        self.assertTrue(self.trial.completed_successfully)

    def test_adding_new_trials(self) -> None:
        new_arm = get_arms()[1]
        cand_metadata = {new_arm.signature: {"a": "b"}}
        new_trial = self.experiment.new_trial(
            generator_run=GeneratorRun(
                arms=[new_arm],
                # pyre-fixme[6]: For 2nd param expected `Optional[Dict[str,
                #  Optional[Dict[str, typing.Any]]]]` but got `Dict[str, Dict[str,
                #  str]]`.
                candidate_metadata_by_arm_signature=cand_metadata,
            )
        )
        with self.assertRaises(ValueError):
            self.experiment.new_trial(generator_run=GeneratorRun(arms=get_arms()))
        self.assertEqual(new_trial.arms_by_name["1_0"].signature, new_arm.signature)
        with self.assertRaises(KeyError):
            self.trial.arms_by_name["1_0"]
        self.assertEqual(
            new_trial._get_candidate_metadata_from_all_generator_runs(),
            {"1_0": cand_metadata[new_arm.signature]},
        )
        self.assertEqual(
            new_trial._get_candidate_metadata("1_0"), cand_metadata[new_arm.signature]
        )
        self.assertRaises(
            ValueError, new_trial._get_candidate_metadata, "this_is_not_an_arm"
        )

    def test_add_trial_same_arm(self) -> None:
        # Check that adding new arm w/out name works correctly.
        new_trial1 = self.experiment.new_trial(
            generator_run=GeneratorRun(arms=[self.arm.clone(clear_name=True)])
        )
        self.assertEqual(new_trial1.arm.name, self.trial.arm.name)
        self.assertFalse(new_trial1.arm is self.trial.arm)
        # Check that adding new arm with name works correctly.
        new_trial2 = self.experiment.new_trial(
            generator_run=GeneratorRun(arms=[self.arm.clone()])
        )
        self.assertEqual(new_trial2.arm.name, self.trial.arm.name)
        self.assertFalse(new_trial2.arm is self.trial.arm)
        arm_wrong_name = self.arm.clone(clear_name=True)
        arm_wrong_name.name = "wrong_name"
        with self.assertRaises(ValueError):
            new_trial2 = self.experiment.new_trial(
                generator_run=GeneratorRun(arms=[arm_wrong_name])
            )

    def test_abandonment(self) -> None:
        self.assertFalse(self.trial.status.is_abandoned)
        self.trial.mark_abandoned(reason="testing")
        self.assertTrue(self.trial.status.is_abandoned)
        self.assertFalse(self.trial.status.is_failed)
        self.assertTrue(self.trial.did_not_complete)

    def test_failed(self) -> None:
        fail_reason = "testing"
        self.trial.runner = SyntheticRunner()
        self.trial.run()
        self.trial.mark_failed(reason=fail_reason)

        self.assertTrue(self.trial.status.is_failed)
        self.assertTrue(self.trial.did_not_complete)
        self.assertEqual(self.trial.failed_reason, fail_reason)

    def test_trial_run_does_not_overwrite_existing_metadata(self) -> None:
        self.trial.runner = SyntheticRunner(dummy_metadata="y")
        self.trial.update_run_metadata({"orig_metadata": "x"})
        self.trial.run()
        self.assertDictEqual(
            self.trial.run_metadata,
            {
                "name": "test_0",
                "orig_metadata": "x",
                "dummy_metadata": "y",
                # this is set in setUp
                "foo": "bar",
            },
        )

    def test_mark_as(self) -> None:
        for terminal_status in (
            TrialStatus.ABANDONED,
            TrialStatus.FAILED,
            TrialStatus.COMPLETED,
            TrialStatus.EARLY_STOPPED,
        ):
            self.setUp()
            # Note: This only tests the no-runner case (and thus not staging)
            for status in (TrialStatus.RUNNING, terminal_status):
                kwargs = {}
                if status == TrialStatus.RUNNING:
                    kwargs["no_runner_required"] = True
                if status == TrialStatus.ABANDONED:
                    kwargs["reason"] = "test_reason_abandon"
                if status == TrialStatus.FAILED:
                    kwargs["reason"] = "test_reason_failed"
                self.trial.mark_as(status=status, **kwargs)
                self.assertTrue(self.trial.status == status)

                if status == TrialStatus.ABANDONED:
                    self.assertEqual(self.trial.abandoned_reason, "test_reason_abandon")
                    self.assertIsNone(self.trial.failed_reason)
                elif status == TrialStatus.FAILED:
                    self.assertEqual(self.trial.failed_reason, "test_reason_failed")
                    self.assertIsNone(self.trial.abandoned_reason)

                if status != TrialStatus.RUNNING:
                    self.assertTrue(self.trial.status.is_terminal)

                if status in [
                    TrialStatus.RUNNING,
                    TrialStatus.EARLY_STOPPED,
                    TrialStatus.COMPLETED,
                ]:
                    self.assertTrue(self.trial.status.expecting_data)
                else:
                    self.assertFalse(self.trial.status.expecting_data)

    def test_stop(self) -> None:
        # test bad old status
        with self.assertRaisesRegex(ValueError, "Can only stop STAGED or RUNNING"):
            self.trial.stop(new_status=TrialStatus.ABANDONED)

        # test bad new status
        self.trial.mark_running(no_runner_required=True)
        with self.assertRaisesRegex(ValueError, "New status of a stopped trial must"):
            self.trial.stop(new_status=TrialStatus.CANDIDATE)

        # dummy runner for testing stopping functionality
        class DummyStopRunner(Runner):
            def run(self, trial):
                pass

            def stop(self, trial, reason):
                return {"reason": reason} if reason else {}

        # test valid stopping
        for reason, new_status in itertools.product(
            (None, "because"),
            (TrialStatus.COMPLETED, TrialStatus.ABANDONED, TrialStatus.EARLY_STOPPED),
        ):
            self.setUp()
            self.trial._runner = DummyStopRunner()
            self.trial.mark_running()
            self.assertEqual(self.trial.status, TrialStatus.RUNNING)
            self.trial.stop(new_status=new_status, reason=reason)
            self.assertEqual(self.trial.status, new_status)
            self.assertEqual(
                self.trial.stop_metadata, {} if reason is None else {"reason": reason}
            )

    @patch(
        f"{BaseTrial.__module__}.{BaseTrial.__name__}.lookup_data",
        return_value=TEST_DATA,
    )
    # pyre-fixme[3]: Return type must be annotated.
    def test_objective_mean(self, _mock):
        self.assertEqual(self.trial.objective_mean, 1.0)

    @patch(
        f"{BaseTrial.__module__}.{BaseTrial.__name__}.lookup_data", return_value=Data()
    )
    # pyre-fixme[3]: Return type must be annotated.
    def test_objective_mean_empty_df(self, _mock):
        with self.assertRaisesRegex(ValueError, "not yet in data for trial."):
            self.assertIsNone(self.trial.objective_mean)

    @patch(
        f"{BaseTrial.__module__}.{BaseTrial.__name__}.fetch_data_results",
        return_value={get_objective().metric.name: Ok(TEST_DATA)},
    )
    def test_fetch_data_result(self, mock: Mock) -> None:
        metric_name = get_objective().metric.name
        results = self.experiment.trials[0].fetch_data_results()

        self.assertTrue(results[metric_name].is_ok())
        self.assertEqual(
            results[metric_name].ok, self.experiment.trials[0].fetch_data()
        )

    def test_Repr(self) -> None:
        repr_ = (
            "Trial(experiment_name='test', index=0, "
            "status=TrialStatus.CANDIDATE, arm=Arm(name='0_0', "
            "parameters={'w': 0.85, 'x': 1, 'y': 'baz', 'z': False}))"
        )
        self.assertEqual(str(self.trial), repr_)

    def test_update_run_metadata(self) -> None:
        self.assertEqual(len(self.trial.run_metadata), 1)
        old_run_metadata = deepcopy(self.trial.run_metadata)
        self.trial.update_run_metadata({"something": "new"})
        self.assertDictEqual(
            self.trial.run_metadata, {**old_run_metadata, "something": "new"}
        )

    def test_update_stop_metadata(self) -> None:
        self.assertEqual(len(self.trial.stop_metadata), 1)
        old_stop_metadata = deepcopy(self.trial.stop_metadata)
        self.trial.update_stop_metadata({"something": "new"})
        self.assertEqual(
            self.trial.stop_metadata, {**old_stop_metadata, "something": "new"}
        )

    def test_update_trial_data(self) -> None:
        # Verify components before we attach trial data
        self.assertEqual(1, len(self.trial.arms))
        arm_name = self.trial.arm.name

        self.assertEqual(
            2,
            len(self.trial.experiment.metrics)
            - len(self.trial.experiment.tracking_metrics),
        )
        self.assertTrue("m1" in self.trial.experiment.metrics)
        self.assertTrue("m2" in self.trial.experiment.metrics)

        data = self.trial.lookup_data().df.to_dict(orient="index")
        self.assertTrue(len(data) == 0)

        # Attach data
        self.trial.update_trial_data(raw_data={"m1": 1.0, "m2": 2.0})

        # Confirm the expected state after attaching data
        data = (
            self.trial.lookup_data()
            .df.set_index(["arm_name", "metric_name"])
            .to_dict(orient="index")
        )

        self.assertEqual(1.0, data[(arm_name, "m1")]["mean"])
        self.assertEqual(2.0, data[(arm_name, "m2")]["mean"])

        # Try to attach MapData.
        with self.assertRaisesRegex(
            UserInputError,
            "The format of the `raw_data` is not compatible with `Data`. ",
        ):
            self.trial.update_trial_data(raw_data=[({"time": 0}, {"m1": 1.0})])

        # Try to attach Data to a MapData experiment.
        map_experiment = get_test_map_data_experiment(
            num_trials=3, num_fetches=2, num_complete=1
        )
        map_trial = map_experiment.new_trial().add_arm(
            arm=get_branin_arms(n=1, seed=0)[0]
        )
        map_trial.update_trial_data(raw_data=[({"time": 0}, {"m1": 1.0})])
        with self.assertRaisesRegex(
            UserInputError,
            "The format of the `raw_data` is not compatible with `MapData`. ",
        ):
            map_trial.update_trial_data(raw_data={"m1": 1.0})

        # Error if the MapData inputs are not formatted correctly.
        with self.assertRaisesRegex(
            UserInputError,
            "Raw data does not conform to the expected structure.",
        ):
            map_trial.update_trial_data(raw_data=[["aa", {"m1": 1.0}]])

    def test_clone_to(self) -> None:
        # cloned trial attached to the same experiment
        self.trial.mark_running(no_runner_required=True)
        new_trial = self.trial.clone_to()
        self.assertIs(new_trial.experiment, self.trial.experiment)
        # Test equality of all attributes except index, time_created, and experiment.
        for k, v in new_trial.__dict__.items():
            if k in ["_index", "_time_created", "_experiment", "_time_run_started"]:
                continue
            self.assertEqual(v, self.trial.__dict__[k])

        # cloned trial attached to a new experiment
        new_experiment = get_experiment()
        new_trial = self.trial.clone_to(new_experiment)
        self.assertEqual(new_trial, self.trial)

        # make sure updating cloned trial doesn't affect original one
        new_trial._status = TrialStatus.COMPLETED
        self.assertTrue(new_trial.status.is_completed)
        self.assertFalse(self.trial.status.is_completed)

        # check that trial_type is cloned correctly
        self.assertEqual(new_trial.trial_type, "foo")

        # test clear_trial_type
        new_trial = self.trial.clone_to(clear_trial_type=True)
        self.assertIsNone(new_trial.trial_type)

    def test_update_trial_status_on_clone(self) -> None:
        for status in [
            TrialStatus.CANDIDATE,
            TrialStatus.STAGED,
            TrialStatus.RUNNING,
            TrialStatus.EARLY_STOPPED,
            TrialStatus.COMPLETED,
            TrialStatus.FAILED,
            TrialStatus.ABANDONED,
        ]:
            self.trial._failed_reason = self.trial._abandoned_reason = None
            if status != TrialStatus.CANDIDATE:
                self.trial.mark_as(
                    status=status, unsafe=True, no_runner_required=True, reason="test"
                )
            test_trial = self.trial.clone_to()
            # Overwrite unimportant attrs before equality check.
            test_trial._index = self.trial.index
            test_trial._time_created = self.trial._time_created
            test_trial._time_staged = self.trial._time_staged
            self.assertEqual(self.trial, test_trial)

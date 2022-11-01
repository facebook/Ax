#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from unittest.mock import patch

import pandas as pd
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun, GeneratorRunType
from ax.core.runner import Runner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_arms, get_experiment, get_objective


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
        self.experiment = get_experiment()
        self.trial = self.experiment.new_trial()
        self.arm = get_arms()[0]
        self.trial.add_arm(self.arm)

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

        self.trial._status = TrialStatus.RUNNING
        self.assertTrue(self.trial.status.is_running)

        self.trial._status = TrialStatus.COMPLETED
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
                    kwargs["reason"] = "test_reason"
                self.trial.mark_as(status=status, **kwargs)
                self.assertTrue(self.trial.status == status)

                if status == TrialStatus.ABANDONED:
                    self.assertEqual(self.trial.abandoned_reason, "test_reason")
                else:
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

    def testRepr(self) -> None:
        repr_ = (
            "Trial(experiment_name='test', index=0, "
            "status=TrialStatus.CANDIDATE, arm=Arm(name='0_0', "
            "parameters={'w': 0.85, 'x': 1, 'y': 'baz', 'z': False}))"
        )
        self.assertEqual(str(self.trial), repr_)

    def test_update_run_metadata(self) -> None:
        self.assertEqual(len(self.trial.run_metadata), 0)
        self.trial.update_run_metadata({"something": "new"})
        self.assertEqual(self.trial.run_metadata, {"something": "new"})

    def test_update_stop_metadata(self) -> None:
        self.assertEqual(len(self.trial.stop_metadata), 0)
        self.trial.update_stop_metadata({"something": "new"})
        self.assertEqual(self.trial.stop_metadata, {"something": "new"})

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pandas as pd
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun, GeneratorRunType
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
    def setUp(self):
        self.experiment = get_experiment()
        self.trial = self.experiment.new_trial()
        self.arm = get_arms()[0]
        self.trial.add_arm(self.arm)

    def test_eq(self):
        new_trial = self.experiment.new_trial()
        self.assertNotEqual(self.trial, new_trial)

    def test_basic_properties(self):
        self.assertEqual(self.experiment, self.trial.experiment)
        self.assertEqual(self.trial.index, 0)
        self.assertEqual(self.trial.status, TrialStatus.CANDIDATE)
        self.assertTrue(self.trial.status.is_candidate)
        self.assertIsNotNone(self.trial.time_created)
        self.assertEqual(self.trial.arms_by_name["0_0"], self.trial.arm)
        self.assertEqual(self.trial.arms, [self.arm])
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

    def test_adding_new_trials(self):
        new_arm = get_arms()[1]
        cand_metadata = {new_arm.signature: {"a": "b"}}
        new_trial = self.experiment.new_trial(
            generator_run=GeneratorRun(
                arms=[new_arm], candidate_metadata_by_arm_signature=cand_metadata
            )
        )
        with self.assertRaises(ValueError):
            self.experiment.new_trial(generator_run=GeneratorRun(arms=get_arms()))
        self.assertEqual(new_trial.arms_by_name["1_0"], new_arm)
        with self.assertRaises(KeyError):
            self.trial.arms_by_name["1_0"]
        self.assertEqual(
            new_trial._get_candidate_metadata_from_all_generator_runs(),
            {"1_0": cand_metadata[new_arm.signature]},
        )

    def test_add_trial_same_arm(self):
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

    def test_abandonment(self):
        self.assertFalse(self.trial.status.is_abandoned)
        self.trial.mark_abandoned(reason="testing")
        self.assertTrue(self.trial.status.is_abandoned)
        self.assertFalse(self.trial.status.is_failed)
        self.assertTrue(self.trial.did_not_complete)

    def test_mark_as(self):
        for terminal_status in (
            TrialStatus.ABANDONED,
            TrialStatus.FAILED,
            TrialStatus.COMPLETED,
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

    @patch(
        f"{BaseTrial.__module__}.{BaseTrial.__name__}.fetch_data",
        return_value=TEST_DATA,
    )
    def test_objective_mean(self, _mock):
        self.assertEqual(self.trial.objective_mean, 1.0)

    @patch(
        f"{BaseTrial.__module__}.{BaseTrial.__name__}.fetch_data", return_value=Data()
    )
    def test_objective_mean_empty_df(self, _mock):
        with self.assertRaisesRegex(ValueError, "No data was retrieved for trial"):
            self.assertIsNone(self.trial.objective_mean)

    def testRepr(self):
        repr_ = (
            "Trial(experiment_name='test', index=0, "
            "status=TrialStatus.CANDIDATE, arm=Arm(name='0_0', "
            "parameters={'w': 0.85, 'x': 1, 'y': 'baz', 'z': False}))"
        )
        self.assertEqual(str(self.trial), repr_)

    def test_update_run_metadata(self):
        self.assertEqual(len(self.trial.run_metadata), 0)
        self.trial.update_run_metadata({"something": "new"})
        self.assertEqual(self.trial.run_metadata, {"something": "new"})

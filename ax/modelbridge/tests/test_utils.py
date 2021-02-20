#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import numpy as np
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.modelbridge.modelbridge_utils import (
    clamp_observation_features,
    get_pending_observation_features,
    pending_observations_as_array,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment


class TestModelbridgeUtils(TestCase):
    def setUp(self) -> None:
        self.experiment = get_experiment()
        self.arm = Arm({"x": 1, "y": "foo", "z": True, "w": 4})
        self.trial = self.experiment.new_trial(GeneratorRun([self.arm]))
        self.experiment_2 = get_experiment()
        self.batch_trial = self.experiment_2.new_batch_trial(GeneratorRun([self.arm]))
        self.batch_trial.set_status_quo_with_weight(self.experiment_2.status_quo, 1)
        self.obs_feat = ObservationFeatures.from_arm(
            arm=self.trial.arm, trial_index=np.int64(self.trial.index)
        )

    def test_get_pending_observation_features(self):
        # Pending observations should be none if there aren't any.
        self.assertIsNone(get_pending_observation_features(self.experiment))
        self.trial.mark_running(no_runner_required=True)
        # Now that the trial is deployed, it should become a pending trial on the
        # experiment and appear as pending for all metrics.
        self.assertEqual(
            get_pending_observation_features(self.experiment),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # With `fetch_data` on trial returning data for metric "m2", that metric
        # should no longer have pending observation features.
        with patch.object(
            self.trial,
            "fetch_data",
            return_value=Data.from_evaluations(
                {self.trial.arm.name: {"m2": (1, 0)}}, trial_index=self.trial.index
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(self.experiment),
                {"tracking": [self.obs_feat], "m2": [], "m1": [self.obs_feat]},
            )
        # When a trial is marked failed, it should no longer appear in pending...
        self.trial.mark_failed()
        self.assertIsNone(get_pending_observation_features(self.experiment))
        # ... unless specified to include failed trials in pending observations.
        self.assertEqual(
            get_pending_observation_features(
                self.experiment, include_failed_as_pending=True
            ),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # When a trial is abandoned, it should appear in pending features whether
        # or not there is data for it.
        self.trial._status = TrialStatus.ABANDONED  # Cannot re-mark a failed trial.
        self.assertEqual(
            get_pending_observation_features(
                self.experiment, include_failed_as_pending=True
            ),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # Checking with data for all metrics.
        with patch.object(
            self.trial,
            "fetch_data",
            return_value=Data.from_evaluations(
                {self.trial.arm.name: {"m1": (1, 0), "m2": (1, 0), "tracking": (1, 0)}},
                trial_index=self.trial.index,
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(self.experiment),
                {
                    "tracking": [self.obs_feat],
                    "m2": [self.obs_feat],
                    "m1": [self.obs_feat],
                },
            )

    def test_get_pending_observation_features_batch_trial(self):
        # Check the same functionality for batched trials.
        self.assertIsNone(get_pending_observation_features(self.experiment_2))
        self.batch_trial.mark_running(no_runner_required=True)
        sq_obs_feat = ObservationFeatures.from_arm(
            self.batch_trial.arms_by_name.get("status_quo"),
            trial_index=self.batch_trial.index,
        )
        self.assertEqual(
            get_pending_observation_features(self.experiment_2),
            {
                "tracking": [self.obs_feat, sq_obs_feat],
                "m2": [self.obs_feat, sq_obs_feat],
                "m1": [self.obs_feat, sq_obs_feat],
            },
        )

    def test_get_pending_observation_features_based_on_trial_status(self):
        # Pending observations should be none if there aren't any as trial is
        # candidate.
        self.assertTrue(self.trial.status.is_candidate)
        self.assertIsNone(get_pending_observation_features(self.experiment))
        self.trial.mark_staged()
        # Now that the trial is staged, it should become a pending trial on the
        # experiment and appear as pending for all metrics.
        self.assertEqual(
            get_pending_observation_features(self.experiment),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # Same should be true for running trial.
        # NOTE: Can't mark a staged trial running unless it uses a runner that
        # specifically requires staging; hacking around that here since the marking
        # logic does not matter for this test.
        self.trial._status = TrialStatus.RUNNING
        # Now that the trial is staged, it should become a pending trial on the
        # experiment and appear as pending for all metrics.
        self.assertEqual(
            get_pending_observation_features(self.experiment),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # When a trial is marked failed, it should no longer appear in pending.
        self.trial.mark_failed()
        self.assertIsNone(get_pending_observation_features(self.experiment))
        # And if the trial is abandoned, it should always appear in pending features.
        self.trial._status = TrialStatus.ABANDONED  # Cannot re-mark a failed trial.
        self.assertEqual(
            get_pending_observation_features(
                self.experiment, include_failed_as_pending=True
            ),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )

    def test_pending_observations_as_array(self):
        # Mark a trial dispatched so that there are pending observations.
        self.trial.mark_running(no_runner_required=True)
        # If outcome names are respected, unlisted metrics should be filtered out.
        self.assertEqual(
            [
                x.tolist()
                for x in pending_observations_as_array(
                    pending_observations=get_pending_observation_features(
                        self.experiment
                    ),
                    outcome_names=["m2", "m1"],
                    param_names=["x", "y", "z", "w"],
                )
            ],
            [[["1", "foo", "True", "4"]], [["1", "foo", "True", "4"]]],
        )
        self.experiment.attach_data(
            Data.from_evaluations(
                {self.trial.arm.name: {"m2": (1, 0)}}, trial_index=self.trial.index
            )
        )
        # With `fetch_data` on trial returning data for metric "m2", that metric
        # should no longer have pending observation features.
        with patch.object(
            self.trial,
            "fetch_data",
            return_value=Data.from_evaluations(
                {self.trial.arm.name: {"m2": (1, 0)}}, trial_index=self.trial.index
            ),
        ):
            pending = get_pending_observation_features(self.experiment)
        # There should be no pending observations for metric m2 now, since the
        # only trial there is, has been updated with data for it.
        self.assertEqual(
            [
                x.tolist()
                for x in pending_observations_as_array(
                    pending_observations=pending,
                    outcome_names=["m2", "m1"],
                    param_names=["x", "y", "z", "w"],
                )
            ],
            [[], [["1", "foo", "True", "4"]]],
        )

    def testClampObservationFeaturesNearBounds(self):
        cases = [
            (
                ObservationFeatures(
                    parameters={"w": 1.0, "x": 2, "y": "foo", "z": True}
                ),
                ObservationFeatures(
                    parameters={"w": 1.0, "x": 2, "y": "foo", "z": True}
                ),
            ),
            (
                ObservationFeatures(
                    parameters={"w": 0.0, "x": 2, "y": "foo", "z": True}
                ),
                ObservationFeatures(
                    parameters={"w": 0.5, "x": 2, "y": "foo", "z": True}
                ),
            ),
            (
                ObservationFeatures(
                    parameters={"w": 100.0, "x": 2, "y": "foo", "z": True}
                ),
                ObservationFeatures(
                    parameters={"w": 5.5, "x": 2, "y": "foo", "z": True}
                ),
            ),
            (
                ObservationFeatures(
                    parameters={"w": 1.0, "x": 0, "y": "foo", "z": True}
                ),
                ObservationFeatures(
                    parameters={"w": 1.0, "x": 1, "y": "foo", "z": True}
                ),
            ),
            (
                ObservationFeatures(
                    parameters={"w": 1.0, "x": 11, "y": "foo", "z": True}
                ),
                ObservationFeatures(
                    parameters={"w": 1.0, "x": 10, "y": "foo", "z": True}
                ),
            ),
        ]
        search_space = get_experiment().search_space
        for obs_ft, expected_obs_ft in cases:
            actual_obs_ft = clamp_observation_features([obs_ft], search_space)
            self.assertEqual(actual_obs_ft[0], expected_obs_ft)

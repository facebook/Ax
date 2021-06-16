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
from ax.core.metric import Metric
from ax.core.objective import Objective, MultiObjective
from ax.core.observation import ObservationFeatures, ObservationData
from ax.core.outcome_constraint import (
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
    ObjectiveThreshold,
)
from ax.core.types import ComparisonOp
from ax.modelbridge.modelbridge_utils import (
    clamp_observation_features,
    get_pending_observation_features,
    pending_observations_as_array,
    extract_outcome_constraints,
    extract_objective_thresholds,
    observation_data_to_array,
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
            "lookup_data",
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
        # When an arm is abandoned, it should appear in pending features whether
        # or not there is data for it.
        self.batch_trial.mark_arm_abandoned(arm_name="0_0")
        # Checking with data for all metrics.
        with patch.object(
            self.batch_trial,
            "fetch_data",
            return_value=Data.from_evaluations(
                {
                    self.batch_trial.arms[0].name: {
                        "m1": (1, 0),
                        "m2": (1, 0),
                        "tracking": (1, 0),
                    }
                },
                trial_index=self.trial.index,
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(
                    self.experiment, include_failed_as_pending=True
                ),
                {
                    "tracking": [self.obs_feat],
                    "m2": [self.obs_feat],
                    "m1": [self.obs_feat],
                },
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

    def test_extract_outcome_constraints(self):
        outcomes = ["m1", "m2", "m3"]
        # pass no outcome constraints
        self.assertIsNone(extract_outcome_constraints([], outcomes))

        outcome_constraints = [
            OutcomeConstraint(metric=Metric("m1"), op=ComparisonOp.LEQ, bound=0)
        ]
        res = extract_outcome_constraints(outcome_constraints, outcomes)
        self.assertEqual(res[0].shape, (1, 3))
        self.assertListEqual(list(res[0][0]), [1, 0, 0])
        self.assertEqual(res[1][0][0], 0)

        outcome_constraints = [
            OutcomeConstraint(metric=Metric("m1"), op=ComparisonOp.LEQ, bound=0),
            ScalarizedOutcomeConstraint(
                metrics=[Metric("m2"), Metric("m3")],
                weights=[0.5, 0.5],
                op=ComparisonOp.GEQ,
                bound=1,
            ),
        ]
        res = extract_outcome_constraints(outcome_constraints, outcomes)
        self.assertEqual(res[0].shape, (2, 3))
        self.assertListEqual(list(res[0][0]), [1, 0, 0])
        self.assertListEqual(list(res[0][1]), [0, -0.5, -0.5])
        self.assertEqual(res[1][0][0], 0)
        self.assertEqual(res[1][1][0], -1)

    def test_extract_objective_thresholds(self):
        outcomes = ["m1", "m2", "m3", "m4"]
        objective = MultiObjective(
            objectives=[Objective(metric=Metric(name)) for name in outcomes[:3]]
        )
        objective_thresholds = [
            ObjectiveThreshold(
                metric=Metric(name),
                op=ComparisonOp.LEQ,
                bound=float(i + 2),
                relative=False,
            )
            for i, name in enumerate(outcomes[:3])
        ]

        # None of no thresholds
        self.assertIsNone(
            extract_objective_thresholds(
                objective_thresholds=[], objective=objective, outcomes=outcomes
            )
        )

        # Working case
        obj_t = extract_objective_thresholds(
            objective_thresholds=objective_thresholds,
            objective=objective,
            outcomes=outcomes,
        )
        expected_obj_t_not_nan = np.array([2.0, 3.0, 4.0])
        self.assertTrue(np.array_equal(obj_t[:3], expected_obj_t_not_nan[:3]))
        self.assertTrue(np.isnan(obj_t[-1]))
        self.assertEqual(obj_t.shape[0], 4)

        # Fails if threshold not provided for all objective metrics
        with self.assertRaises(ValueError):
            extract_objective_thresholds(
                objective_thresholds=objective_thresholds[:2],
                objective=objective,
                outcomes=outcomes,
            )

        # Fails if number of thresholds doesn't equal number of objectives
        objective2 = Objective(Metric("m1"))
        with self.assertRaises(ValueError):
            extract_objective_thresholds(
                objective_thresholds=objective_thresholds,
                objective=objective2,
                outcomes=outcomes,
            )

        # Works with a single objective, single threshold
        obj_t = extract_objective_thresholds(
            objective_thresholds=objective_thresholds[:1],
            objective=objective2,
            outcomes=outcomes,
        )
        self.assertEqual(obj_t[0], 2.0)
        self.assertTrue(np.all(np.isnan(obj_t[1:])))
        self.assertEqual(obj_t.shape[0], 4)

        # Fails if relative
        objective_thresholds[2] = ObjectiveThreshold(
            metric=Metric("m3"), op=ComparisonOp.LEQ, bound=3
        )
        with self.assertRaises(ValueError):
            extract_objective_thresholds(
                objective_thresholds=objective_thresholds,
                objective=objective,
                outcomes=outcomes,
            )
        objective_thresholds[2] = ObjectiveThreshold(
            metric=Metric("m3"), op=ComparisonOp.LEQ, bound=3, relative=True
        )
        with self.assertRaises(ValueError):
            extract_objective_thresholds(
                objective_thresholds=objective_thresholds,
                objective=objective,
                outcomes=outcomes,
            )

    def testObservationDataToArray(self):
        outcomes = ["a", "b", "c"]
        obsd = ObservationData(
            metric_names=["c", "a", "b"],
            means=np.array([1, 2, 3]),
            covariance=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        )
        Y, Ycov = observation_data_to_array(outcomes=outcomes, observation_data=[obsd])
        self.assertTrue(np.array_equal(Y, np.array([[2, 3, 1]])))
        self.assertTrue(
            np.array_equal(Ycov, np.array([[[5, 6, 4], [8, 9, 7], [2, 3, 1]]]))
        )

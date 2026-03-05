#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.evaluations_to_data import raw_evaluations_to_data
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.trial_status import TrialStatus
from ax.core.types import ComparisonOp
from ax.core.utils import (
    _maybe_update_trial_status_to_complete,
    batch_trial_only,
    best_feasible_objective,
    compute_metric_availability,
    extract_pending_observations,
    get_missing_metrics,
    get_missing_metrics_by_name,
    get_model_times,
    get_model_trace_of_times,
    get_pending_observation_features,
    get_pending_observation_features_based_on_trial_status as get_pending_status,
    get_target_trial_index,
    is_bandit_experiment,
    MetricAvailability,
    MissingMetrics,
)
from ax.exceptions.core import AxError
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_data_batch,
    get_branin_experiment,
    get_experiment,
    get_hierarchical_search_space_experiment,
)
from pyre_extensions import none_throws


class UtilsTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.empty_experiment = get_experiment()
        self.experiment = get_experiment()
        self.arm = Arm({"x": 5, "y": "foo", "z": True, "w": 5, "d": 11.0})
        self.trial = self.experiment.new_trial(GeneratorRun([self.arm]))
        self.experiment_2 = get_experiment()
        self.batch_trial = self.experiment_2.new_batch_trial(GeneratorRun([self.arm]))
        self.batch_trial.add_status_quo_arm(weight=1)
        self.obs_feat = ObservationFeatures.from_arm(
            arm=self.trial.arm, trial_index=self.trial.index
        )
        self.hss_arm = Arm({"model": "XGBoost", "num_boost_rounds": 12})
        self.hss_exp = get_hierarchical_search_space_experiment()
        self.hss_gr = GeneratorRun(
            arms=[self.hss_arm],
            candidate_metadata_by_arm_signature={
                self.hss_arm.signature: {
                    Keys.FULL_PARAMETERIZATION: {
                        "model_name": "XGBoost",
                        "num_boost_rounds": 12,
                        "learning_rate": 0.01,
                        "l2_reg_weight": 0.0001,
                    }
                }
            },
        )
        self.hss_trial = self.hss_exp.new_trial(self.hss_gr)
        self.hss_cand_metadata = self.hss_trial._get_candidate_metadata(
            arm_name=self.hss_arm.name
        )
        self.hss_full_parameterization = self.hss_cand_metadata.get(
            Keys.FULL_PARAMETERIZATION
        ).copy()
        self.hss_obs_feat = ObservationFeatures.from_arm(
            arm=self.hss_arm,
            trial_index=self.hss_trial.index,
            metadata=self.hss_cand_metadata,
        )
        self.df = pd.DataFrame(
            [
                {
                    "arm_name": "0_0",
                    "mean": 2.0,
                    "sem": 0.2,
                    "trial_index": 1,
                    "metric_name": "a",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                    "metric_signature": "a",
                },
                {
                    "arm_name": "0_0",
                    "mean": 1.8,
                    "sem": 0.3,
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                    "metric_signature": "b",
                },
                {
                    "arm_name": "0_1",
                    "mean": float("nan"),
                    "sem": float("nan"),
                    "trial_index": 1,
                    "metric_name": "a",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                    "metric_signature": "a",
                },
                {
                    "arm_name": "0_1",
                    "mean": 3.7,
                    "sem": 0.5,
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                    "metric_signature": "b",
                },
                {
                    "arm_name": "0_2",
                    "mean": 0.5,
                    "sem": None,
                    "trial_index": 1,
                    "metric_name": "a",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                    "metric_signature": "a",
                },
                {
                    "arm_name": "0_2",
                    "mean": float("nan"),
                    "sem": float("nan"),
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                    "metric_signature": "b",
                },
                {
                    "arm_name": "0_2",
                    "mean": float("nan"),
                    "sem": float("nan"),
                    "trial_index": 1,
                    "metric_name": "c",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                    "metric_signature": "c",
                },
            ]
        )

        self.data = Data(df=self.df)

        self.optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric(name="a"), minimize=False),
            outcome_constraints=[
                OutcomeConstraint(
                    metric=Metric(name="b"),
                    op=ComparisonOp.GEQ,
                    bound=0,
                    relative=False,
                )
            ],
        )
        self.batch_experiment = get_branin_experiment(with_completed_trial=False)
        self.batch_experiment.status_quo = Arm(
            name="status_quo", parameters={"x1": 0.0, "x2": 0.0}
        )

    def test_get_missing_metrics_by_name(self) -> None:
        expected = {"a": {("0_1", 1)}, "b": {("0_2", 1)}}
        actual = get_missing_metrics_by_name(self.data, ["a", "b"])
        self.assertEqual(actual, expected)

    def test_get_missing_metrics(self) -> None:
        expected = MissingMetrics(
            {"a": {("0_1", 1)}},
            {"b": {("0_2", 1)}},
            {"c": {("0_0", 1), ("0_1", 1), ("0_2", 1)}},
        )
        actual = get_missing_metrics(self.data, self.optimization_config)
        self.assertEqual(actual, expected)

    def test_best_feasible_objective(self) -> None:
        bfo = best_feasible_objective(
            self.optimization_config,
            values={"a": np.array([1.0, 3.0, 2.0]), "b": np.array([0.0, -1.0, 0.0])},
        )
        self.assertEqual(list(bfo), [1.0, 1.0, 2.0])

    def test_get_model_times(self) -> None:
        exp = get_branin_experiment(num_trial=2)
        fit_times, gen_times = get_model_trace_of_times(exp)
        total_fit_time, total_gen_time = get_model_times(exp)
        fit_times_not_none = [none_throws(elt) for elt in fit_times]
        gen_times_not_none = [none_throws(elt) for elt in gen_times]
        self.assertTrue(all(elt >= 0 for elt in fit_times_not_none))
        self.assertTrue(all(elt >= 0 for elt in gen_times_not_none))
        self.assertEqual(sum(fit_times_not_none), total_fit_time)
        self.assertEqual(sum(gen_times_not_none), total_gen_time)

    def test_get_pending_observation_features(self) -> None:
        # Pending observations should be none if there aren't any.
        self.assertIsNone(get_pending_observation_features(self.empty_experiment))
        # Candidate trial is included as pending trial.
        self.assertEqual(
            get_pending_observation_features(self.experiment),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # Still a pending trial after deployment.
        self.trial.mark_running(no_runner_required=True)
        self.assertEqual(
            get_pending_observation_features(self.experiment),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # With data for metric "m2", that metric should no longer have pending
        # observation features.
        with patch.object(
            self.experiment,
            "lookup_data",
            return_value=raw_evaluations_to_data(
                {self.trial.arm.name: {"m2": (1, 0)}},
                trial_index=self.trial.index,
                metric_name_to_signature={"m2": "m2"},
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(self.experiment),
                {"tracking": [self.obs_feat], "m2": [], "m1": [self.obs_feat]},
            )
        # A completed trial without data should still appear as pending.
        self.trial.mark_completed()
        self.assertEqual(
            get_pending_observation_features(self.experiment),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # A completed trial with data for some metrics should be pending only
        # for metrics without data.
        with patch.object(
            self.experiment,
            "lookup_data",
            return_value=raw_evaluations_to_data(
                {self.trial.arm.name: {"m2": (1, 0)}},
                trial_index=self.trial.index,
                metric_name_to_signature={"m2": "m2"},
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(self.experiment),
                {"tracking": [self.obs_feat], "m2": [], "m1": [self.obs_feat]},
            )
        # When a trial is marked failed, it should no longer appear in pending.
        self.trial._status = TrialStatus.FAILED
        self.assertIsNone(get_pending_observation_features(self.experiment))
        # Abandoned trials without data should appear as pending for all metrics.
        self.trial._status = TrialStatus.ABANDONED  # Cannot re-mark a failed trial.
        self.assertEqual(
            get_pending_observation_features(self.experiment),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # Abandoned trials with data for some metrics should only be pending
        # for metrics without data.
        with patch.object(
            self.experiment,
            "lookup_data",
            return_value=raw_evaluations_to_data(
                {self.trial.arm.name: {"m2": (1, 0)}},
                trial_index=self.trial.index,
                metric_name_to_signature={"m2": "m2"},
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(self.experiment),
                {"tracking": [self.obs_feat], "m2": [], "m1": [self.obs_feat]},
            )
        # Individually abandoned arms in a batch trial should NOT appear
        # in pending features.
        self.trial._status = TrialStatus.FAILED  # Remove trial from pending.
        self.batch_trial.mark_arm_abandoned(arm_name="0_0")
        self.assertIsNone(get_pending_observation_features(self.experiment))

    def test_update_trial_status(self) -> None:
        """
        Test that _maybe_update_trial_status_to_complete always marks the trial
        as COMPLETED, even when optimization config metrics are missing.
        """
        # Create an experiment with optimization config
        experiment = get_experiment()

        # Make sure optimization_config is not None
        self.assertIsNotNone(experiment.optimization_config)
        trial = experiment.new_trial(GeneratorRun([self.arm]))
        trial.mark_running(no_runner_required=True)

        # Attach data for only one metric (not all required by optimization config)
        data = Data(
            df=pd.DataFrame(
                [
                    {
                        "arm_name": none_throws(trial.arm).name,
                        "mean": 1.0,
                        "sem": 0.1,
                        "trial_index": trial.index,
                        "metric_name": "m1",  # Only attach data for m1, not m2
                        "start_time": "2018-01-01",
                        "end_time": "2018-01-02",
                        "metric_signature": "m1",
                    }
                ]
            )
        )
        experiment.attach_data(data)

        # The trial should be marked as COMPLETED even with missing metrics
        with self.assertLogs("ax.core.utils", level="WARNING") as log:
            _maybe_update_trial_status_to_complete(
                experiment=experiment, trial_index=trial.index
            )
        self.assertEqual(trial.status, TrialStatus.COMPLETED)
        self.assertTrue(
            any("missing optimization config metrics" in msg for msg in log.output)
        )

        with self.subTest("Test with no opt config"):
            experiment = get_experiment()

            # Set optimization_config to None
            experiment._optimization_config = None
            self.assertIsNone(experiment.optimization_config)

            trial = experiment.new_trial(GeneratorRun([self.arm]))
            trial.mark_running(no_runner_required=True)

            # Should still mark as COMPLETED when no opt config
            _maybe_update_trial_status_to_complete(
                experiment=experiment, trial_index=trial.index
            )
            self.assertEqual(trial.status, TrialStatus.COMPLETED)

        with self.subTest("Test with all metrics present"):
            experiment = get_experiment()
            self.assertIsNotNone(experiment.optimization_config)
            trial = experiment.new_trial(GeneratorRun([self.arm]))
            trial.mark_running(no_runner_required=True)

            # Attach data for all required metrics
            all_data = Data(
                df=pd.DataFrame(
                    [
                        {
                            "arm_name": none_throws(trial.arm).name,
                            "mean": 1.0,
                            "sem": 0.1,
                            "trial_index": trial.index,
                            "metric_name": metric_name,
                            "metric_signature": metric_name,
                        }
                        for metric_name in none_throws(
                            experiment.optimization_config
                        ).metrics
                    ]
                )
            )
            experiment.attach_data(all_data)

            _maybe_update_trial_status_to_complete(
                experiment=experiment, trial_index=trial.index
            )
            self.assertEqual(trial.status, TrialStatus.COMPLETED)

    def test_completed_incomplete_trials_are_pending(self) -> None:
        """Test that COMPLETED trials with incomplete metric availability appear
        as pending in both pending observation feature functions."""
        experiment = get_experiment()
        self.assertIsNotNone(experiment.optimization_config)
        arm = Arm({"x": 5, "y": "foo", "z": True, "w": 5, "d": 11.0})
        trial = experiment.new_trial(GeneratorRun([arm]))
        trial.mark_running(no_runner_required=True)

        # Attach data for only m1 (missing m2 from opt config)
        data = Data(
            df=pd.DataFrame(
                [
                    {
                        "arm_name": none_throws(trial.arm).name,
                        "mean": 1.0,
                        "sem": 0.1,
                        "trial_index": trial.index,
                        "metric_name": "m1",
                        "metric_signature": "m1",
                    }
                ]
            )
        )
        experiment.attach_data(data)
        trial.mark_completed()

        obs_feat = ObservationFeatures.from_arm(
            arm=none_throws(trial.arm), trial_index=trial.index
        )

        # get_pending_observation_features: COMPLETED+INCOMPLETE trial should
        # be pending for the missing metric m2 but NOT for the present metric m1.
        pending = get_pending_observation_features(experiment)
        self.assertIsNotNone(pending)
        # m2 is missing from trial data, so this trial's arm is pending for m2
        self.assertIn(obs_feat, pending["m2"])
        # m1 is present, so this trial's arm should NOT be pending for m1
        self.assertNotIn(obs_feat, pending["m1"])

        # get_pending_observation_features_based_on_trial_status:
        # COMPLETED+INCOMPLETE trial should be pending for all metrics
        # (this function doesn't do per-metric tracking).
        pending_status = get_pending_status(experiment)
        self.assertIsNotNone(pending_status)
        self.assertIn(obs_feat, pending_status["m1"])
        self.assertIn(obs_feat, pending_status["m2"])

    def test_completed_complete_trials_not_pending(self) -> None:
        """Test that COMPLETED trials with complete metric availability do NOT
        appear as pending."""
        experiment = get_experiment()
        self.assertIsNotNone(experiment.optimization_config)
        arm = Arm({"x": 5, "y": "foo", "z": True, "w": 5, "d": 11.0})
        trial = experiment.new_trial(GeneratorRun([arm]))
        trial.mark_running(no_runner_required=True)

        # Attach data for all opt config metrics
        data = Data(
            df=pd.DataFrame(
                [
                    {
                        "arm_name": none_throws(trial.arm).name,
                        "mean": 1.0,
                        "sem": 0.1,
                        "trial_index": trial.index,
                        "metric_name": metric_name,
                        "metric_signature": metric_name,
                    }
                    for metric_name in none_throws(
                        experiment.optimization_config
                    ).metrics
                ]
            )
        )
        experiment.attach_data(data)
        trial.mark_completed()

        obs_feat = ObservationFeatures.from_arm(
            arm=none_throws(trial.arm), trial_index=trial.index
        )

        # get_pending_observation_features: COMPLETED+COMPLETE trial should
        # NOT be pending for any metric.
        pending = get_pending_observation_features(experiment)
        if pending is not None:
            for metric_name in none_throws(experiment.optimization_config).metrics:
                self.assertNotIn(obs_feat, pending.get(metric_name, []))

        # get_pending_observation_features_based_on_trial_status:
        # COMPLETED+COMPLETE trial should NOT be pending.
        pending_status = get_pending_status(experiment)
        if pending_status is not None:
            for metric_name in none_throws(experiment.optimization_config).metrics:
                self.assertNotIn(obs_feat, pending_status.get(metric_name, []))

    def test_get_pending_observation_features_multi_trial(self) -> None:
        # With data for metric "m2", that metric should no longer have pending
        # observation features.
        self.trial.mark_running(no_runner_required=True)
        with patch.object(
            self.experiment,
            "lookup_data",
            return_value=raw_evaluations_to_data(
                {self.trial.arm.name: {"m2": (1, 0)}},
                trial_index=self.trial.index,
                metric_name_to_signature={"m2": "m2"},
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(self.experiment),
                {"tracking": [self.obs_feat], "m2": [], "m1": [self.obs_feat]},
            )

        # Make sure that trial_index is set correctly
        other_obs_feat = ObservationFeatures.from_arm(arm=self.trial.arm, trial_index=1)
        other_trial = self.experiment.new_trial(GeneratorRun([self.arm]))
        other_trial.mark_running(no_runner_required=True)

        trial_0_data = raw_evaluations_to_data(
            {self.trial.arm.name: {"m2": (1, 0)}},
            trial_index=self.trial.index,
            metric_name_to_signature={"m2": "m2"},
        )
        trial_1_data = raw_evaluations_to_data(
            {other_trial.arm.name: {"m2": (1, 0), "tracking": (1, 0)}},
            trial_index=other_trial.index,
            metric_name_to_signature={"m2": "m2", "tracking": "tracking"},
        )
        combined_data = Data.from_multiple_data([trial_0_data, trial_1_data])
        with patch.object(
            self.experiment,
            "lookup_data",
            return_value=combined_data,
        ):
            pending = get_pending_observation_features(self.experiment)
            self.assertEqual(
                pending,
                {
                    "tracking": [self.obs_feat],
                    "m2": [],
                    "m1": [self.obs_feat, other_obs_feat],
                },
            )

    def test_get_pending_observation_features_out_of_design(self) -> None:
        # Out of design points are excluded depending on the kwarg.
        with patch.object(
            self.experiment.search_space,
            "check_membership",
            return_value=False,
        ):
            self.assertIsNone(
                get_pending_observation_features(
                    self.experiment, include_out_of_design_points=False
                ),
            )

        with patch.object(
            self.experiment.search_space,
            "check_membership",
            return_value=False,
        ):
            self.assertEqual(
                get_pending_observation_features(
                    self.experiment, include_out_of_design_points=True
                ),
                {
                    "tracking": [self.obs_feat],
                    "m2": [self.obs_feat],
                    "m1": [self.obs_feat],
                },
            )

    def test_get_pending_observation_features_hss(self) -> None:
        # The trial is candidate, it should be a pending trial on the
        # experiment and appear as pending for all metrics.
        pending = get_pending_observation_features(self.hss_exp)
        self.assertEqual(
            pending,
            {
                "m1": [self.hss_obs_feat],
                "m2": [self.hss_obs_feat],
            },
        )

        # Check that transforming observation features works correctly (it should inject
        # full parameterization into resulting obs.feats.)
        for p in none_throws(pending).values():
            for pf in p:
                self.assertEqual(
                    none_throws(pf.metadata),
                    none_throws(self.hss_gr.candidate_metadata_by_arm_signature)[
                        self.hss_arm.signature
                    ],
                )

        # With data for metric "m2", that metric should no longer have pending
        # observation features.
        self.hss_trial.mark_running(no_runner_required=True)
        with patch.object(
            self.hss_exp,
            "lookup_data",
            return_value=raw_evaluations_to_data(
                {self.hss_trial.arm.name: {"m2": (1, 0)}},
                trial_index=self.hss_trial.index,
                metric_name_to_signature={"m2": "m2"},
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(self.hss_exp),
                {"m2": [], "m1": [self.hss_obs_feat]},
            )
        # When a trial is marked failed, it should no longer appear in pending.
        self.hss_trial.mark_failed()
        self.assertIsNone(get_pending_observation_features(self.hss_exp))

        # Abandoned arms should not appear in pending features.
        hss_exp = get_hierarchical_search_space_experiment()
        hss_batch_trial = hss_exp.new_batch_trial(generator_run=self.hss_gr)
        hss_batch_trial.mark_arm_abandoned(hss_batch_trial.arms[0].name)
        # Mark the trial failed, so that only abandoned arm remains.
        hss_batch_trial.mark_running(no_runner_required=True).mark_failed()
        self.assertIsNone(get_pending_observation_features(hss_exp))

    def test_get_pending_observation_features_batch_trial(self) -> None:
        # Check the same functionality for batched trials.
        # Status quo of this experiment is out-of-design, so it shouldn't be
        # among the pending points.
        self.assertEqual(
            get_pending_observation_features(self.experiment_2),
            {
                "tracking": [self.obs_feat],
                "m2": [self.obs_feat],
                "m1": [self.obs_feat],
            },
        )

        # Status quo of this experiment is out-of-design, so it shouldn't be
        # among the pending points.
        sq_obs_feat = ObservationFeatures.from_arm(
            self.batch_trial.arms_by_name.get("status_quo"),
            trial_index=self.batch_trial.index,
        )
        self.assertEqual(
            get_pending_observation_features(
                self.experiment_2,
                include_out_of_design_points=True,
            ),
            {
                "tracking": [self.obs_feat, sq_obs_feat],
                "m2": [self.obs_feat, sq_obs_feat],
                "m1": [self.obs_feat, sq_obs_feat],
            },
        )
        self.batch_trial.mark_running(no_runner_required=True)
        self.batch_trial.mark_completed()

        # Set SQ to in-design; then we can expect it to appear among the pending
        # points without specifying `include_out_of_design_points=True`.
        exp = get_experiment(with_status_quo=False)
        in_design_status_quo = Arm(
            name="in_design_status_quo",
            parameters={"w": 5.45, "x": 5, "y": "bar", "z": True, "d": 11.9},
        )
        exp.status_quo = in_design_status_quo
        batch = exp.new_batch_trial().add_arm(self.arm)
        batch.add_status_quo_arm(weight=1)
        self.assertEqual(batch.status_quo, in_design_status_quo)
        self.assertTrue(
            exp.search_space.check_membership(
                in_design_status_quo.parameters, raise_error=True
            )
        )
        batch.mark_running(no_runner_required=True)
        sq_obs_feat = ObservationFeatures.from_arm(
            in_design_status_quo,
            trial_index=batch.index,
        )
        self.assertEqual(
            get_pending_observation_features(exp),
            {
                "tracking": [self.obs_feat, sq_obs_feat],
                "m2": [self.obs_feat, sq_obs_feat],
                "m1": [self.obs_feat, sq_obs_feat],
            },
        )

    def test_get_pending_observation_features_based_on_trial_status(self) -> None:
        # The trial is candidate, it should be a pending trial on the
        # experiment and appear as pending for all metrics.
        self.assertEqual(
            get_pending_status(self.experiment),
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
            get_pending_status(self.experiment),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # When a trial is marked failed, it should no longer appear in pending.
        self.trial.mark_failed()
        self.assertIsNone(get_pending_status(self.experiment))
        # Abandoned trials should appear in pending features.
        self.trial._status = TrialStatus.ABANDONED  # Cannot re-mark a failed trial.
        self.assertEqual(
            get_pending_status(self.experiment),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )

    def test_get_pending_observation_features_based_on_trial_status_hss(self) -> None:
        # The HSS trial is candidate, so it should appear pending.
        pending = get_pending_status(self.hss_exp)
        self.assertEqual(
            pending,
            {
                "m1": [self.hss_obs_feat],
                "m2": [self.hss_obs_feat],
            },
        )

        # Same should be true for running trial.
        # NOTE: Can't mark a staged trial running unless it uses a runner that
        # specifically requires staging; hacking around that here since the marking
        # logic does not matter for this test.
        self.hss_trial._status = TrialStatus.RUNNING
        # Now that the trial is staged, it should become a pending trial on the
        # experiment and appear as pending for all metrics.
        pending = get_pending_status(self.hss_exp)
        self.assertEqual(
            pending,
            {
                "m1": [self.hss_obs_feat],
                "m2": [self.hss_obs_feat],
            },
        )
        # When a trial is marked failed, it should no longer appear in pending.
        self.hss_trial.mark_failed()
        self.assertIsNone(get_pending_status(self.hss_exp))
        # Abandoned trials should appear in pending features.
        self.hss_trial._status = TrialStatus.ABANDONED  # Cannot re-mark a failed trial.
        pending = get_pending_status(self.hss_exp)
        self.assertEqual(
            pending,
            {
                "m1": [self.hss_obs_feat],
                "m2": [self.hss_obs_feat],
            },
        )

    def test_extract_pending_observations(self) -> None:
        exp_with_many_trials = get_experiment()
        for _ in range(100):
            exp_with_many_trials.new_trial().add_arm(self.arm)

        exp_with_many_trials_and_batch = deepcopy(exp_with_many_trials)
        exp_with_many_trials_and_batch.new_batch_trial().add_arm(self.arm)

        m = extract_pending_observations.__module__
        with (
            patch(f"{m}.get_pending_observation_features") as mock_pending,
            patch(
                f"{m}.get_pending_observation_features_based_on_trial_status"
            ) as mock_pending_ts,
        ):
            # Check the typical case: few trials, we can use regular `get_pending...`.
            extract_pending_observations(experiment=self.experiment)
            mock_pending.assert_called_once_with(
                experiment=self.experiment, include_out_of_design_points=False
            )
            mock_pending.reset_mock()

            # Check out-of-design filter propagation.
            extract_pending_observations(
                experiment=self.experiment, include_out_of_design_points=True
            )
            mock_pending.assert_called_once_with(
                experiment=self.experiment, include_out_of_design_points=True
            )
            mock_pending.reset_mock()

            # Check many-trials case and out-of-design filter propagation.
            extract_pending_observations(
                experiment=exp_with_many_trials, include_out_of_design_points=True
            )
            mock_pending_ts.assert_called_once_with(
                experiment=exp_with_many_trials, include_out_of_design_points=True
            )

            # Check "many-trials but batch trial present" case
            # and out-of-design filter propagation.
            extract_pending_observations(
                experiment=exp_with_many_trials_and_batch,
                include_out_of_design_points=True,
            )
            mock_pending_ts.assert_called_once_with(
                experiment=exp_with_many_trials, include_out_of_design_points=True
            )

    def test_get_target_trial_index_non_batch(self) -> None:
        # Testing with non-BatchTrial. Should only return the index of the
        # SQ trial if it exists and has data.
        experiment = get_branin_experiment(with_completed_trial=True)
        self.assertIsNone(get_target_trial_index(experiment=experiment))
        # Add SQ but it is doesn't have data yet.
        experiment.status_quo = Arm(
            name="status_quo", parameters={"x1": 0.0, "x2": 0.0}
        )
        self.assertIsNone(get_target_trial_index(experiment=experiment))
        # Add data to SQ.
        trial = experiment.new_trial().add_arm(experiment.status_quo)
        trial.mark_running(no_runner_required=True)
        experiment.attach_data(get_branin_data(trials=[trial]))
        self.assertEqual(get_target_trial_index(experiment=experiment), trial.index)

    def test_get_target_trial_index_stale_trial_filtering(self) -> None:
        trials = []
        for days_ago in [15, 5]:  # old trial (stale), new trial (recent)
            trial = self.batch_experiment.new_batch_trial().add_arm(
                self.batch_experiment.status_quo
            )
            trial.mark_completed(unsafe=True)
            trial._time_completed = datetime.now() - timedelta(days=days_ago)
            self.batch_experiment.attach_data(get_branin_data_batch(batch=trial))
            trials.append(trial)

        self.assertEqual(
            get_target_trial_index(experiment=self.batch_experiment),
            trials[1].index,  # newer trial
        )

    def test_get_target_trial_index_all_stale_fallback(self) -> None:
        trial = self.batch_experiment.new_batch_trial().add_arm(
            self.batch_experiment.status_quo
        )
        trial.mark_completed(unsafe=True)
        trial._time_completed = datetime.now() - timedelta(days=15)  # stale
        self.batch_experiment.attach_data(get_branin_data_batch(batch=trial))

        # fallback to stale trial over none
        self.assertEqual(
            get_target_trial_index(experiment=self.batch_experiment), trial.index
        )

    def test_get_target_trial_index_longrun_to_shortrun_fallback(self) -> None:
        # long run without data
        long_run_trial = self.batch_experiment.new_batch_trial(
            trial_type=Keys.LONG_RUN
        ).add_arm(self.batch_experiment.status_quo)
        long_run_trial.mark_running(no_runner_required=True)

        # short run with data
        short_run_trial = self.batch_experiment.new_batch_trial().add_arm(
            self.batch_experiment.status_quo
        )
        short_run_trial.mark_running(no_runner_required=True)
        self.batch_experiment.attach_data(get_branin_data_batch(batch=short_run_trial))

        # ahould fallback to short-run trial since long-run has no SQ data
        self.assertEqual(
            get_target_trial_index(experiment=self.batch_experiment),
            short_run_trial.index,
        )

        # once long-run trial has data, should return long-run trial
        self.batch_experiment.attach_data(get_branin_data_batch(batch=long_run_trial))
        self.assertEqual(
            get_target_trial_index(experiment=self.batch_experiment),
            long_run_trial.index,
        )

    def test_get_target_trial_index_opt_config_metric_filtering(self) -> None:
        # add tracking metric, opt config is already branin
        self.batch_experiment.add_tracking_metric(Metric(name="test_metric"))

        # trial with opt config data only
        trial = (
            self.batch_experiment.new_batch_trial()
            .add_arm(self.batch_experiment.status_quo)
            .mark_running(no_runner_required=True)
        )
        self.batch_experiment.attach_data(get_branin_data_batch(batch=trial))

        # default should pass because we'll have opt config data
        self.assertEqual(
            get_target_trial_index(
                experiment=self.batch_experiment, require_data_for_all_metrics=False
            ),
            trial.index,
        )

        # when require_data_for_all_metrics=True, should return None
        # because there are no trials with data for all metrics
        self.assertIsNone(
            get_target_trial_index(
                experiment=self.batch_experiment, require_data_for_all_metrics=True
            )
        )

    def test_batch_trial_only_decorator(self) -> None:
        # Create a mock function to decorate
        def mock_func(trial: BatchTrial) -> None:
            pass

        experiment = get_branin_experiment(with_completed_trial=True)
        decorated_func = batch_trial_only()(mock_func)

        # Test that decorator raises an error for missing trial keyword arg
        with self.assertRaises(AxError) as e:
            decorated_func()
        self.assertRegex(str(e.exception), r"Expected a keyword argument `trial` to .*")

        # Test that decorator raises an error for non-batch trial
        with self.assertRaises(AxError) as e:
            decorated_func(trial="not a batch trial")
        self.assertRegex(
            str(e.exception),
            r"Expected the argument `trial` to `.*` to be a `BatchTrial`, but got .*",
        )

        # Test that decorator works for batch trial
        batch_trial = BatchTrial(experiment=experiment)
        decorated_func(trial=batch_trial)

    def test_batch_trial_only_decorator_with_custom_message(self) -> None:
        # Create a mock function to decorate
        def mock_func(trial: BatchTrial) -> None:
            pass

        # Test that decorator raises an error with custom message
        custom_message = "Batch trials only!"
        decorated_func = batch_trial_only(msg=custom_message)(mock_func)
        with self.assertRaisesRegex(AxError, custom_message):
            decorated_func(trial="not a batch trial")

    def test_is_bandit_experiment_util(self) -> None:
        with self.subTest("non-bandit GS"):
            self.assertFalse(
                is_bandit_experiment(generation_strategy_name="non-bandit GS")
            )

        with self.subTest("bandit GS"):
            self.assertTrue(
                is_bandit_experiment(
                    generation_strategy_name=(
                        Keys.FACTORIAL_PLUS_EMPIRICAL_BAYES_THOMPSON_SAMPLING
                    )
                )
            )

    def test_get_target_trial_index_only_selects_completed_trials(self) -> None:
        # should return None since the only trial with data is failed
        failed_trial = (
            self.batch_experiment.new_batch_trial()
            .add_arm(self.batch_experiment.status_quo)
            .mark_running(no_runner_required=True)
        )
        self.batch_experiment.attach_data(get_branin_data_batch(batch=failed_trial))
        failed_trial.mark_failed()
        self.assertIsNone(get_target_trial_index(experiment=self.batch_experiment))

        # should return the completed trial, not the failed one
        completed_trial = (
            self.batch_experiment.new_batch_trial()
            .add_arm(self.batch_experiment.status_quo)
            .mark_running(no_runner_required=True)
        )
        self.batch_experiment.attach_data(get_branin_data_batch(batch=completed_trial))
        completed_trial.mark_completed(unsafe=True)
        self.assertEqual(
            get_target_trial_index(experiment=self.batch_experiment),
            completed_trial.index,
        )

        # should return the completed trial, not the failed or abandoned ones
        abandoned_trial = (
            self.batch_experiment.new_batch_trial()
            .add_arm(self.batch_experiment.status_quo)
            .mark_running(no_runner_required=True)
        )
        self.batch_experiment.attach_data(get_branin_data_batch(batch=abandoned_trial))
        abandoned_trial.mark_abandoned()
        self.assertEqual(
            get_target_trial_index(experiment=self.batch_experiment),
            completed_trial.index,
        )


class TestMetricAvailability(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Constrained experiment: requires "branin" (objective) + "branin_e"
        # (absolute constraint).
        self.exp = get_branin_experiment(
            with_trial=True,
            with_completed_trial=False,
            with_absolute_constraint=True,
            num_trial=3,
        )

    def test_availability_levels(self) -> None:
        """Test COMPLETE, INCOMPLETE, and NOT_OBSERVED availability in a single
        experiment with multiple trials."""
        exp = self.exp

        # Trial 0: all metrics → COMPLETE
        exp.trials[0].mark_running(no_runner_required=True)
        exp.attach_data(
            get_branin_data(trial_indices=[0], metrics=["branin", "branin_e"])
        )
        exp.trials[0].mark_completed()

        # Trial 1: partial metrics → INCOMPLETE
        exp.trials[1].mark_running(no_runner_required=True)
        exp.attach_data(get_branin_data(trial_indices=[1], metrics=["branin"]))
        exp.trials[1].mark_completed()

        # Trial 2: no data → NOT_OBSERVED
        exp.trials[2].mark_running(no_runner_required=True)
        exp.trials[2].mark_completed()

        result = compute_metric_availability(experiment=exp)
        self.assertEqual(
            [result[i] for i in range(3)],
            [
                MetricAvailability.COMPLETE,
                MetricAvailability.INCOMPLETE,
                MetricAvailability.NOT_OBSERVED,
            ],
        )

        # Tracking-only data also counts as NOT_OBSERVED.
        exp_tracking = get_branin_experiment(
            with_trial=True,
            with_completed_trial=False,
        )
        trial = exp_tracking.trials[0]
        trial.mark_running(no_runner_required=True)
        exp_tracking.attach_data(
            get_branin_data(trial_indices=[0], metrics=["tracking_metric"])
        )
        trial.mark_completed()
        result = compute_metric_availability(experiment=exp_tracking)
        self.assertEqual(result[0], MetricAvailability.NOT_OBSERVED)

    def test_no_optimization_config_raises(self) -> None:
        """An error is raised when no optimization config is available."""
        exp = get_branin_experiment(
            has_optimization_config=False,
            with_trial=True,
            with_completed_trial=False,
        )
        exp.trials[0].mark_running(no_runner_required=True)
        exp.trials[0].mark_completed()

        with self.assertRaisesRegex(ValueError, "optimization config is required"):
            compute_metric_availability(experiment=exp)

    def test_custom_optimization_config(self) -> None:
        """An explicit optimization_config overrides the experiment's, and a
        subset config can change the result."""
        exp = get_branin_experiment(
            with_trial=True,
            with_completed_trial=False,
            with_absolute_constraint=True,
        )
        trial = exp.trials[0]
        trial.mark_running(no_runner_required=True)
        # Attach data for "branin" only (missing "branin_e").
        exp.attach_data(get_branin_data(trial_indices=[0], metrics=["branin"]))
        trial.mark_completed()

        # Against experiment's opt config (requires branin + branin_e): INCOMPLETE.
        result = compute_metric_availability(experiment=exp)
        self.assertEqual(result[0], MetricAvailability.INCOMPLETE)

        # Custom config requiring only "branin": COMPLETE.
        custom_config = OptimizationConfig(
            objective=Objective(metric=Metric(name="branin"), minimize=False),
        )
        result = compute_metric_availability(
            experiment=exp, optimization_config=custom_config
        )
        self.assertEqual(result[0], MetricAvailability.COMPLETE)

        # Custom config requiring an unrelated metric: INCOMPLETE.
        other_config = OptimizationConfig(
            objective=Objective(metric=Metric(name="branin"), minimize=False),
            outcome_constraints=[
                OutcomeConstraint(
                    metric=Metric(name="other_metric"),
                    op=ComparisonOp.LEQ,
                    bound=10.0,
                    relative=False,
                ),
            ],
        )
        result = compute_metric_availability(
            experiment=exp, optimization_config=other_config
        )
        self.assertEqual(result[0], MetricAvailability.INCOMPLETE)

    def test_metric_names_parameter(self) -> None:
        """The metric_names parameter overrides optimization_config for
        determining required metrics."""
        exp = get_branin_experiment(
            with_trial=True,
            with_completed_trial=False,
        )
        trial = exp.trials[0]
        trial.mark_running(no_runner_required=True)
        # Attach data for "branin" only.
        exp.attach_data(get_branin_data(trial_indices=[0], metrics=["branin"]))
        trial.mark_completed()

        # With metric_names={"branin"}: COMPLETE (data exists).
        result = compute_metric_availability(experiment=exp, metric_names={"branin"})
        self.assertEqual(result[0], MetricAvailability.COMPLETE)

        # With metric_names={"branin", "other"}: INCOMPLETE (missing "other").
        result = compute_metric_availability(
            experiment=exp, metric_names={"branin", "other"}
        )
        self.assertEqual(result[0], MetricAvailability.INCOMPLETE)

        # With metric_names={"nonexistent"}: NOT_OBSERVED.
        result = compute_metric_availability(
            experiment=exp, metric_names={"nonexistent"}
        )
        self.assertEqual(result[0], MetricAvailability.NOT_OBSERVED)

        # metric_names takes precedence over optimization_config.
        result = compute_metric_availability(
            experiment=exp, metric_names={"branin", "other"}
        )
        self.assertEqual(result[0], MetricAvailability.INCOMPLETE)

    def test_curve_data(self) -> None:
        """For curve data, a metric observed at any step counts as available;
        a metric with zero rows is still missing."""
        exp = get_branin_experiment(
            with_trial=True,
            with_completed_trial=False,
        )
        exp.optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric(name="metric_a"), minimize=False),
            outcome_constraints=[
                OutcomeConstraint(
                    metric=Metric(name="metric_b"),
                    op=ComparisonOp.LEQ,
                    bound=10.0,
                    relative=False,
                ),
            ],
        )
        trial = exp.trials[0]
        trial.mark_running(no_runner_required=True)
        arm_name = trial.arm.name  # pyre-ignore[16]

        # Both metrics present at various steps → COMPLETE.
        df_both = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "arm_name": arm_name,
                    "metric_name": metric,
                    "metric_signature": metric,
                    "mean": float(i),
                    "sem": 0.0,
                    "step": step,
                }
                for i, (metric, step) in enumerate(
                    [("metric_a", 0), ("metric_a", 1), ("metric_a", 2), ("metric_b", 0)]
                )
            ]
        )
        exp.attach_data(Data(df=df_both))
        trial.mark_completed()
        result = compute_metric_availability(experiment=exp)
        self.assertEqual(result[0], MetricAvailability.COMPLETE)

        # Only metric_a present, metric_b missing → INCOMPLETE.
        exp2 = get_branin_experiment(
            with_trial=True,
            with_completed_trial=False,
        )
        exp2.optimization_config = none_throws(exp.optimization_config)
        trial2 = exp2.trials[0]
        trial2.mark_running(no_runner_required=True)
        arm_name2 = trial2.arm.name  # pyre-ignore[16]
        df_partial = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "arm_name": arm_name2,
                    "metric_name": "metric_a",
                    "metric_signature": "metric_a",
                    "mean": float(step),
                    "sem": 0.0,
                    "step": step,
                }
                for step in range(2)
            ]
        )
        exp2.attach_data(Data(df=df_partial))
        trial2.mark_completed()
        result2 = compute_metric_availability(experiment=exp2)
        self.assertEqual(result2[0], MetricAvailability.INCOMPLETE)

    def test_trial_indices_and_empty(self) -> None:
        """Specific trial_indices limits computation; empty returns empty dict."""
        exp = get_branin_experiment(
            with_trial=True,
            with_completed_trial=True,
            num_trial=3,
        )
        # Specific trial index.
        result = compute_metric_availability(experiment=exp, trial_indices=[0])
        self.assertEqual(len(result), 1)
        self.assertIn(0, result)
        self.assertEqual(result[0], MetricAvailability.COMPLETE)

        # Empty trial indices.
        result = compute_metric_availability(experiment=exp, trial_indices=[])
        self.assertEqual(result, {})

    def test_all_trials_included_by_default(self) -> None:
        """All trials (including non-terminal) are computed by default."""
        exp = get_branin_experiment(
            with_trial=True,
            with_completed_trial=False,
            num_trial=3,
        )
        # Trial 0: completed with data.
        exp.trials[0].mark_running(no_runner_required=True)
        exp.attach_data(get_branin_data(trial_indices=[0], metrics=["branin"]))
        exp.trials[0].mark_completed()

        # Trial 1: CANDIDATE (non-terminal, no data).
        # Trial 2: failed with data.
        exp.trials[2].mark_running(no_runner_required=True)
        exp.attach_data(get_branin_data(trial_indices=[2], metrics=["branin"]))
        exp.trials[2].mark_failed()

        result = compute_metric_availability(experiment=exp)
        # All 3 trials are included.
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], MetricAvailability.COMPLETE)
        self.assertEqual(result[1], MetricAvailability.NOT_OBSERVED)
        self.assertEqual(result[2], MetricAvailability.COMPLETE)

    def test_early_stopped_and_stale_trials(self) -> None:
        """Early-stopped trials are included; stale trials are also included
        when computing for all trials by default."""
        exp = get_branin_experiment(
            with_trial=True,
            with_completed_trial=False,
            num_trial=2,
        )
        # Trial 0: early stopped with data.
        exp.trials[0].mark_running(no_runner_required=True)
        exp.attach_data(get_branin_data(trial_indices=[0], metrics=["branin"]))
        exp.trials[0].mark_early_stopped()

        # Trial 1: stale (no data).
        exp.trials[1].mark_stale()

        result = compute_metric_availability(experiment=exp)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], MetricAvailability.COMPLETE)
        self.assertEqual(result[1], MetricAvailability.NOT_OBSERVED)

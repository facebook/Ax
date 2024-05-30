#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.core.utils import (
    best_feasible_objective,
    extract_pending_observations,
    get_missing_metrics,
    get_missing_metrics_by_name,
    get_model_times,
    get_model_trace_of_times,
    get_pending_observation_features,
    get_pending_observation_features_based_on_trial_status as get_pending_status,
    MissingMetrics,
)

from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment,
    get_hierarchical_search_space_experiment,
    get_robust_branin_experiment,
)
from pyre_extensions import none_throws


class UtilsTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.empty_experiment = get_experiment()
        self.experiment = get_experiment()
        self.arm = Arm({"x": 5, "y": "foo", "z": True, "w": 5})
        self.trial = self.experiment.new_trial(GeneratorRun([self.arm]))
        self.experiment_2 = get_experiment()
        self.batch_trial = self.experiment_2.new_batch_trial(GeneratorRun([self.arm]))
        self.batch_trial.set_status_quo_with_weight(self.experiment_2.status_quo, 1)
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
        self.hss_obs_feat_all_params = ObservationFeatures.from_arm(
            arm=Arm(self.hss_full_parameterization),
            trial_index=self.hss_trial.index,
            metadata={Keys.FULL_PARAMETERIZATION: self.hss_full_parameterization},
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
                },
                {
                    "arm_name": "0_0",
                    "mean": 1.8,
                    "sem": 0.3,
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_1",
                    "mean": float("nan"),
                    "sem": float("nan"),
                    "trial_index": 1,
                    "metric_name": "a",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_1",
                    "mean": 3.7,
                    "sem": 0.5,
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_2",
                    "mean": 0.5,
                    "sem": None,
                    "trial_index": 1,
                    "metric_name": "a",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_2",
                    "mean": float("nan"),
                    "sem": float("nan"),
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_2",
                    "mean": float("nan"),
                    "sem": float("nan"),
                    "trial_index": 1,
                    "metric_name": "c",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
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
        exp = get_robust_branin_experiment(num_sobol_trials=2)
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
        # When a trial is marked failed, it should no longer appear in pending.
        self.trial.mark_failed()
        self.assertIsNone(get_pending_observation_features(self.experiment))
        # When a trial is abandoned, it should appear in pending features whether
        # or not there is data for it.
        self.trial._status = TrialStatus.ABANDONED  # Cannot re-mark a failed trial.
        self.assertEqual(
            get_pending_observation_features(self.experiment),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # When an arm is abandoned, it should appear in pending features whether
        # or not there is data for it.
        self.batch_trial.mark_arm_abandoned(arm_name="0_0")
        # Checking with data for all metrics.
        with patch.object(
            self.batch_trial,
            "fetch_data",
            return_value=Metric._wrap_trial_data_multi(
                data=Data.from_evaluations(
                    {
                        self.batch_trial.arms[0].name: {
                            "m1": (1, 0),
                            "m2": (1, 0),
                            "tracking": (1, 0),
                        }
                    },
                    trial_index=self.trial.index,
                ),
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
        # Checking with data for all metrics.
        with patch.object(
            self.trial,
            "fetch_data",
            return_value=Metric._wrap_trial_data_multi(
                data=Data.from_evaluations(
                    {
                        self.trial.arm.name: {
                            "m1": (1, 0),
                            "m2": (1, 0),
                            "tracking": (1, 0),
                        }
                    },
                    trial_index=self.trial.index,
                ),
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

    def test_get_pending_observation_features_multi_trial(self) -> None:
        # With `fetch_data` on trial returning data for metric "m2", that metric
        # should no longer have pending observation features.
        self.trial.mark_running(no_runner_required=True)
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

        # Make sure that trial_index is set correctly
        other_obs_feat = ObservationFeatures.from_arm(arm=self.trial.arm, trial_index=1)
        other_trial = self.experiment.new_trial(GeneratorRun([self.arm]))
        other_trial.mark_running(no_runner_required=True)

        with patch.object(
            self.trial,
            "lookup_data",
            return_value=Data.from_evaluations(
                {self.trial.arm.name: {"m2": (1, 0)}}, trial_index=self.trial.index
            ),
        ):
            with patch.object(
                other_trial,
                "lookup_data",
                return_value=Data.from_evaluations(
                    {other_trial.arm.name: {"m2": (1, 0), "tracking": (1, 0)}},
                    trial_index=other_trial.index,
                ),
            ):
                pending = get_pending_observation_features(self.experiment)
                print(pending)
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

        # With `fetch_data` on trial returning data for metric "m2", that metric
        # should no longer have pending observation features.
        self.hss_trial.mark_running(no_runner_required=True)
        with patch.object(
            self.hss_trial,
            "lookup_data",
            return_value=Data.from_evaluations(
                {self.hss_trial.arm.name: {"m2": (1, 0)}},
                trial_index=self.hss_trial.index,
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(self.hss_exp),
                {"m2": [], "m1": [self.hss_obs_feat]},
            )
        # When a trial is marked failed, it should no longer appear in pending.
        self.hss_trial.mark_failed()
        self.assertIsNone(get_pending_observation_features(self.hss_exp))

        # When an arm is abandoned, it should appear in pending features whether
        # or not there is data for it.
        hss_exp = get_hierarchical_search_space_experiment()
        hss_batch_trial = hss_exp.new_batch_trial(generator_run=self.hss_gr)
        hss_batch_trial.mark_arm_abandoned(hss_batch_trial.arms[0].name)
        # Mark the trial failed, so that only abandoned arm shows up.
        hss_batch_trial.mark_running(no_runner_required=True).mark_failed()
        # Checking with data for all metrics.
        with patch.object(
            hss_batch_trial,
            "fetch_data",
            return_value=Metric._wrap_trial_data_multi(
                data=Data.from_evaluations(
                    {
                        hss_batch_trial.arms[0].name: {
                            "m1": (1, 0),
                            "m2": (1, 0),
                        }
                    },
                    trial_index=hss_batch_trial.index,
                ),
            ),
        ):
            pending = get_pending_observation_features(hss_exp)
            self.assertEqual(
                pending,
                {
                    "m1": [self.hss_obs_feat],
                    "m2": [self.hss_obs_feat],
                },
            )
            # Check that candidate metadata is property propagated for abandoned arm.
            for p in none_throws(pending).values():
                for pf in p:
                    self.assertEqual(
                        none_throws(pf.metadata),
                        none_throws(self.hss_gr.candidate_metadata_by_arm_signature)[
                            self.hss_arm.signature
                        ],
                    )

        # Checking with data for all metrics.
        with patch.object(
            hss_batch_trial,
            "fetch_data",
            return_value=Metric._wrap_trial_data_multi(
                data=Data.from_evaluations(
                    {
                        hss_batch_trial.arms[0].name: {
                            "m1": (1, 0),
                            "m2": (1, 0),
                        }
                    },
                    trial_index=hss_batch_trial.index,
                ),
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(hss_exp),
                {
                    "m2": [self.hss_obs_feat],
                    "m1": [self.hss_obs_feat],
                },
            )

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
            parameters={"w": 5.45, "x": 5, "y": "bar", "z": True},
        )
        exp.status_quo = in_design_status_quo
        batch = exp.new_batch_trial().add_arm(self.arm)
        batch.set_status_quo_with_weight(exp.status_quo, 1)
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
        # And if the trial is abandoned, it should always appear in pending features.
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
        # And if the trial is abandoned, it should always appear in pending features.
        self.hss_trial._status = TrialStatus.ABANDONED  # Cannot re-mark a failed trial.
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

    def test_extract_pending_observations(self) -> None:
        exp_with_many_trials = get_experiment()
        for _ in range(100):
            exp_with_many_trials.new_trial().add_arm(self.arm)

        exp_with_many_trials_and_batch = deepcopy(exp_with_many_trials)
        exp_with_many_trials_and_batch.new_batch_trial().add_arm(self.arm)

        m = extract_pending_observations.__module__
        with patch(f"{m}.get_pending_observation_features") as mock_pending, patch(
            f"{m}.get_pending_observation_features_based_on_trial_status"
        ) as mock_pending_ts:
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

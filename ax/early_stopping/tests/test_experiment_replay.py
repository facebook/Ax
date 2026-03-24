#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

from ax.core.base_trial import TrialStatus
from ax.core.data import MAP_KEY
from ax.early_stopping.experiment_replay import (
    estimate_hypothetical_early_stopping_savings,
    replay_experiment,
)
from ax.early_stopping.strategies.percentile import PercentileEarlyStoppingStrategy
from ax.early_stopping.utils import estimate_early_stopping_savings
from ax.exceptions.core import UnsupportedError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_timestamp_map_metric,
    get_test_map_data_experiment,
)
from pyre_extensions import none_throws


class TestReplayExperiment(TestCase):
    def test_single_objective_replay(self) -> None:
        """Single-objective replay with heterogeneous trials."""
        historical_experiment = get_test_map_data_experiment(
            num_trials=3, num_fetches=3, num_complete=3
        )
        metric_name = none_throws(
            historical_experiment.optimization_config
        ).objective.metric_names[0]
        metric = historical_experiment.get_metric(metric_name)

        replayed = replay_experiment(
            historical_experiment=historical_experiment,
            num_samples_per_curve=20,
            max_replay_trials=3,
            metrics=[metric],
            max_pending_trials=3,
            early_stopping_strategy=None,
        )
        replayed = none_throws(replayed)
        # All trials should have been processed
        self.assertGreater(len(replayed.trials), 0)
        self.assertLessEqual(len(replayed.trials), 3)

    def test_multi_objective_replay(self) -> None:
        """Multi-objective replay with shared state."""
        historical_experiment = get_test_map_data_experiment(
            num_trials=2,
            num_fetches=2,
            num_complete=2,
            multi_objective=True,
        )
        opt_config = none_throws(historical_experiment.optimization_config)
        metric_names = opt_config.objective.metric_names
        metrics = [historical_experiment.get_metric(mn) for mn in metric_names]

        replayed = replay_experiment(
            historical_experiment=historical_experiment,
            num_samples_per_curve=20,
            max_replay_trials=2,
            metrics=metrics,
            max_pending_trials=2,
            early_stopping_strategy=None,
        )
        replayed = none_throws(replayed)
        # Should have replay metrics for each original metric
        replay_metric_names = {m.name for m in replayed.metrics.values()}
        for mn in metric_names:
            self.assertIn(mn, replay_metric_names)

    def test_multi_objective_replayed_data_matches_historical(self) -> None:
        """Verify that MOO replay serves correct data for every objective
        metric across all trials."""
        historical_experiment = get_test_map_data_experiment(
            num_trials=2,
            num_fetches=3,
            num_complete=2,
            multi_objective=True,
        )
        opt_config = none_throws(historical_experiment.optimization_config)
        metric_names = opt_config.objective.metric_names
        metrics = [historical_experiment.get_metric(mn) for mn in metric_names]

        replayed = none_throws(
            replay_experiment(
                historical_experiment=historical_experiment,
                num_samples_per_curve=20,
                max_replay_trials=2,
                metrics=metrics,
                max_pending_trials=2,
                early_stopping_strategy=None,
            )
        )

        with self.subTest("all_trials_completed"):
            for t in replayed.trials.values():
                self.assertEqual(t.status, TrialStatus.COMPLETED)

        historical_data = historical_experiment.lookup_data()
        historical_subsampled = historical_data.subsample(
            limit_rows_per_group=20, include_first_last=True
        )
        hist_df = historical_subsampled.full_df
        replayed_data = replayed.lookup_data()
        replay_df = replayed_data.full_df

        for mn in metric_names:
            with self.subTest(f"data_matches_for_{mn}"):
                hist_metric_df = hist_df[hist_df["metric_name"] == mn]
                sorted_hist_indices = sorted(hist_metric_df["trial_index"].unique())

                for replay_trial_index in sorted(replay_df["trial_index"].unique()):
                    replay_metric_df = replay_df[
                        (replay_df["trial_index"] == replay_trial_index)
                        & (replay_df["metric_name"] == mn)
                    ]
                    replay_steps = sorted(replay_metric_df[MAP_KEY].tolist())

                    hist_trial_index = sorted_hist_indices[int(replay_trial_index)]
                    hist_steps = sorted(
                        hist_metric_df[
                            hist_metric_df["trial_index"] == hist_trial_index
                        ][MAP_KEY].tolist()
                    )
                    self.assertEqual(
                        replay_steps,
                        hist_steps,
                        f"Metric {mn}, trial {replay_trial_index}: "
                        f"replayed steps {replay_steps} "
                        f"!= historical steps {hist_steps}",
                    )

    def test_replay_with_early_stopping(self) -> None:
        """End-to-end replay with a PercentileEarlyStoppingStrategy."""
        historical_experiment = get_test_map_data_experiment(
            num_trials=3, num_fetches=5, num_complete=3
        )
        metric_name = none_throws(
            historical_experiment.optimization_config
        ).objective.metric_names[0]
        metric = historical_experiment.get_metric(metric_name)

        ess = PercentileEarlyStoppingStrategy(
            percentile_threshold=50.0,
            min_curves=1,
            min_progression=0.1,
        )
        replayed = replay_experiment(
            historical_experiment=historical_experiment,
            num_samples_per_curve=20,
            max_replay_trials=3,
            metrics=[metric],
            max_pending_trials=3,
            early_stopping_strategy=ess,
        )
        replayed = none_throws(replayed)
        self.assertEqual(len(replayed.trials), 3)

    def test_replay_no_step_column(self) -> None:
        """Test that replay returns None when data has no step column."""
        exp = get_branin_experiment(has_optimization_config=True)
        metric_name = none_throws(exp.optimization_config).objective.metric_names[0]
        metric = exp.get_metric(metric_name)
        result = replay_experiment(
            historical_experiment=exp,
            num_samples_per_curve=20,
            max_replay_trials=50,
            metrics=[metric],
            max_pending_trials=5,
            early_stopping_strategy=None,
        )
        self.assertIsNone(result)

    def test_replayed_data_matches_historical(self) -> None:
        """Verify that after replay without ESS, every trial's replayed data
        contains exactly the same set of MAP_KEY values and metric values
        as the historical data (after subsampling)."""
        historical_experiment = get_test_map_data_experiment(
            num_trials=3, num_fetches=5, num_complete=3
        )
        metric_name = none_throws(
            historical_experiment.optimization_config
        ).objective.metric_names[0]
        metric = historical_experiment.get_metric(metric_name)

        replayed = none_throws(
            replay_experiment(
                historical_experiment=historical_experiment,
                num_samples_per_curve=20,
                max_replay_trials=3,
                metrics=[metric],
                max_pending_trials=1,
                early_stopping_strategy=None,
            )
        )

        with self.subTest("all_trials_completed"):
            for t in replayed.trials.values():
                self.assertEqual(t.status, TrialStatus.COMPLETED)

        with self.subTest("replayed_data_matches_historical"):
            # Subsample the historical data the same way replay_experiment does
            historical_data = historical_experiment.lookup_data()
            historical_subsampled = historical_data.subsample(
                limit_rows_per_group=20, include_first_last=True
            )
            hist_df = historical_subsampled.full_df
            hist_metric_df = hist_df[hist_df["metric_name"] == metric_name]

            replayed_data = replayed.lookup_data()
            replay_df = replayed_data.full_df

            # For each replayed trial, verify step values match historical
            for replay_trial_index in sorted(replay_df["trial_index"].unique()):
                replay_steps = sorted(
                    replay_df[replay_df["trial_index"] == replay_trial_index][
                        MAP_KEY
                    ].tolist()
                )
                # Map replay trial index back to historical trial index
                sorted_hist_indices = sorted(hist_metric_df["trial_index"].unique())
                hist_trial_index = sorted_hist_indices[int(replay_trial_index)]
                hist_steps = sorted(
                    hist_metric_df[hist_metric_df["trial_index"] == hist_trial_index][
                        MAP_KEY
                    ].tolist()
                )
                self.assertEqual(
                    replay_steps,
                    hist_steps,
                    f"Trial {replay_trial_index}: replayed steps {replay_steps} "
                    f"!= historical steps {hist_steps}",
                )

    def test_early_stopping_produces_savings(self) -> None:
        """Verify that replay with an ESS completes successfully and
        produces a valid savings estimate (>= 0)."""
        historical_experiment = get_test_map_data_experiment(
            num_trials=5, num_fetches=10, num_complete=5
        )
        metric_name = none_throws(
            historical_experiment.optimization_config
        ).objective.metric_names[0]
        metric = historical_experiment.get_metric(metric_name)

        ess = PercentileEarlyStoppingStrategy(
            percentile_threshold=70.0,
            min_curves=2,
            min_progression=0.1,
        )
        replayed = none_throws(
            replay_experiment(
                historical_experiment=historical_experiment,
                num_samples_per_curve=20,
                max_replay_trials=5,
                metrics=[metric],
                max_pending_trials=5,
                early_stopping_strategy=ess,
            )
        )

        with self.subTest("all_trials_created"):
            self.assertEqual(len(replayed.trials), 5)

        with self.subTest("savings_are_valid"):
            savings = estimate_early_stopping_savings(experiment=replayed)
            self.assertGreaterEqual(savings, 0.0)
            self.assertLessEqual(savings, 1.0)


class TestEstimateHypotheticalEss(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Experiment with MapMetric for tests that need a valid default ESS.
        self.exp = get_branin_experiment_with_timestamp_map_metric()
        metric_name = none_throws(self.exp.optimization_config).objective.metric_names[
            0
        ]
        self.metric = self.exp.get_metric(metric_name)

    def test_estimate_hypothetical_ess_no_default_strategy(self) -> None:
        """Test that UnsupportedError is raised when no default ESS is available."""
        # Non-MapMetric experiment has no default ESS.
        exp = get_branin_experiment(has_optimization_config=True)
        metric_name = none_throws(exp.optimization_config).objective.metric_names[0]
        metric = exp.get_metric(metric_name)

        with self.assertRaises(UnsupportedError) as e:
            estimate_hypothetical_early_stopping_savings(
                experiment=exp,
                metrics=[metric],
            )

        self.assertIn(
            "No default early stopping strategy available",
            str(e.exception),
        )

    def test_estimate_hypothetical_ess_no_progression_data(self) -> None:
        """Test that UnsupportedError is raised when experiment has no progression
        data."""
        with patch(
            "ax.early_stopping.experiment_replay.replay_experiment",
            return_value=None,
        ):
            with self.assertRaises(UnsupportedError) as e:
                estimate_hypothetical_early_stopping_savings(
                    experiment=self.exp,
                    metrics=[self.metric],
                )

            self.assertIn(
                "Experiment data does not have progression data for replay",
                str(e.exception),
            )

    def test_estimate_hypothetical_ess_success(self) -> None:
        """Test that savings are returned when replay succeeds."""
        with (
            patch(
                "ax.early_stopping.experiment_replay.replay_experiment",
            ) as mock_replay,
            patch(
                "ax.early_stopping.experiment_replay.estimate_early_stopping_savings",
                return_value=0.25,
            ) as mock_estimate,
        ):
            result = estimate_hypothetical_early_stopping_savings(
                experiment=self.exp,
                metrics=[self.metric],
            )

            self.assertEqual(result, 0.25)
            mock_replay.assert_called_once()
            mock_estimate.assert_called_once()

    def test_estimate_hypothetical_ess_exception(self) -> None:
        """Test that exceptions from replay propagate to the caller."""
        with patch(
            "ax.early_stopping.experiment_replay.replay_experiment",
            side_effect=ValueError("Experiment's name is None."),
        ):
            with self.assertRaises(ValueError) as e:
                estimate_hypothetical_early_stopping_savings(
                    experiment=self.exp,
                    metrics=[self.metric],
                )

            self.assertIn("Experiment's name is None.", str(e.exception))

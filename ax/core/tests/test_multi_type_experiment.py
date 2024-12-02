#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.multi_type_experiment import (
    filter_trials_by_type,
    get_trial_indices_for_statuses,
)
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.branin import BraninMetric
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_arms, get_multi_type_experiment


class MultiTypeExperimentTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_multi_type_experiment()

    def test_MTExperimentFlow(self) -> None:
        self.assertTrue(self.experiment.supports_trial_type("type1"))
        self.assertTrue(self.experiment.supports_trial_type("type2"))
        self.assertFalse(self.experiment.supports_trial_type(None))

        n = 10
        arms = get_branin_arms(n=n, seed=0)

        b1 = self.experiment.new_batch_trial()
        b1.add_arms_and_weights(arms=arms)
        self.assertEqual(b1.trial_type, "type1")
        b1.run()
        self.assertEqual(b1.run_metadata["dummy_metadata"], "dummy1")

        self.experiment.update_runner("type2", SyntheticRunner(dummy_metadata="dummy3"))
        b2 = self.experiment.new_batch_trial()
        b2.trial_type = "type2"
        b2.add_arms_and_weights(arms=arms)
        self.assertEqual(b2.trial_type, "type2")
        b2.run()
        self.assertEqual(b2.run_metadata["dummy_metadata"], "dummy3")

        df = self.experiment.fetch_data().df
        for _, row in df.iterrows():
            # Make sure proper metric present for each batch only
            self.assertEqual(
                row["metric_name"], "m1" if row["trial_index"] == 0 else "m2"
            )

        arm_0_slice = df.loc[df["arm_name"] == "0_0"]
        self.assertNotEqual(
            float(arm_0_slice[df["trial_index"] == 0]["mean"].item()),
            float(arm_0_slice[df["trial_index"] == 1]["mean"].item()),
        )
        self.assertEqual(len(df), 2 * n)
        self.assertEqual(self.experiment.default_trials, {0})
        # Set 2 metrics to be equal
        self.experiment.update_tracking_metric(
            BraninMetric("m2", ["x1", "x2"]), trial_type="type2"
        )
        df = self.experiment.fetch_data().df
        arm_0_slice = df.loc[df["arm_name"] == "0_0"]
        self.assertAlmostEqual(
            float(arm_0_slice[df["trial_index"] == 0]["mean"].item()),
            float(arm_0_slice[df["trial_index"] == 1]["mean"].item()),
            places=10,
        )

    def test_Repr(self) -> None:
        self.assertEqual(str(self.experiment), "MultiTypeExperiment(test_exp)")

    def test_Eq(self) -> None:
        exp2 = get_multi_type_experiment()

        # Should be equal to start
        self.assertTrue(self.experiment == exp2)

        self.experiment.add_tracking_metric(
            BraninMetric("m3", ["x2", "x1"]), trial_type="type1", canonical_name="m4"
        )

        # Test different set of metrics
        self.assertFalse(self.experiment == exp2)

        exp2.add_tracking_metric(
            BraninMetric("m3", ["x2", "x1"]), trial_type="type1", canonical_name="m5"
        )

        # Test different metric definitions
        self.assertFalse(self.experiment == exp2)

        exp2.update_tracking_metric(
            BraninMetric("m3", ["x2", "x1"]), trial_type="type1", canonical_name="m4"
        )

        # Should be the same
        self.assertTrue(self.experiment == exp2)

        exp2.remove_tracking_metric("m3")
        self.assertFalse(self.experiment == exp2)

    def test_BadBehavior(self) -> None:
        # Add trial type that already exists
        with self.assertRaises(ValueError):
            self.experiment.add_trial_type("type1", SyntheticRunner())

        # Update runner for non-existent trial type
        with self.assertRaises(ValueError):
            self.experiment.update_runner("type3", SyntheticRunner())

        # Add metric for trial_type that doesn't exist
        with self.assertRaises(ValueError):
            self.experiment.add_tracking_metric(
                BraninMetric("m2", ["x1", "x2"]), "type3"
            )

        # Try to remove metric that doesn't exist
        with self.assertRaises(ValueError):
            self.experiment.remove_tracking_metric("m3")

        # Try to change optimization metric to non-primary trial type
        with self.assertRaises(ValueError):
            self.experiment.update_tracking_metric(
                BraninMetric("m1", ["x1", "x2"]), "type2"
            )

        # Update metric definition for trial_type that doesn't exist
        with self.assertRaises(ValueError):
            self.experiment.update_tracking_metric(
                BraninMetric("m2", ["x1", "x2"]), "type3"
            )

        # Try to get runner for trial_type that's not supported
        batch = self.experiment.new_batch_trial()
        batch._trial_type = "type3"  # Force override trial type
        with self.assertRaises(ValueError):
            self.experiment.runner_for_trial(batch)

        # Try making trial with unsupported trial type
        with self.assertRaises(ValueError):
            self.experiment.new_batch_trial(trial_type="type3")

        # Try resetting runners.
        with self.assertRaises(NotImplementedError):
            self.experiment.reset_runners(SyntheticRunner())

    def test_setting_opt_config(self) -> None:
        self.assertDictEqual(
            self.experiment._metric_to_trial_type, {"m1": "type1", "m2": "type2"}
        )
        self.experiment.optimization_config = OptimizationConfig(
            Objective(BraninMetric("m3", ["x1", "x2"]), minimize=True)
        )
        self.assertDictEqual(
            self.experiment._metric_to_trial_type, {"m2": "type2", "m3": "type1"}
        )

    def test_runner_for_trial_type(self) -> None:
        runner = self.experiment.runner_for_trial_type(trial_type="type1")
        self.assertIs(runner, self.experiment._trial_type_to_runner["type1"])
        with self.assertRaisesRegex(
            ValueError, "Trial type `invalid` is not supported."
        ):
            self.experiment.runner_for_trial_type(trial_type="invalid")

    def test_add_tracking_metrics(self) -> None:
        type1_metrics = [
            BraninMetric("m3_type1", ["x1", "x2"]),
            BraninMetric("m4_type1", ["x1", "x2"]),
        ]
        type2_metrics = [
            BraninMetric("m3_type2", ["x1", "x2"]),
            BraninMetric("m4_type2", ["x1", "x2"]),
        ]
        default_type_metrics = [
            BraninMetric("m5_default_type", ["x1", "x2"]),
        ]
        self.experiment.add_tracking_metrics(
            metrics=type1_metrics + type2_metrics + default_type_metrics,
            metrics_to_trial_types={
                "m3_type1": "type1",
                "m4_type1": "type1",
                "m3_type2": "type2",
                "m4_type2": "type2",
            },
        )
        self.assertDictEqual(
            self.experiment._metric_to_trial_type,
            {
                "m1": "type1",
                "m2": "type2",
                "m3_type1": "type1",
                "m4_type1": "type1",
                "m3_type2": "type2",
                "m4_type2": "type2",
                "m5_default_type": "type1",
            },
        )

    def test_stop_trial_runs_multi_type_experiment(self) -> None:
        # Setup 3 trials with 2 runners
        self.experiment.new_batch_trial(trial_type="type1")
        self.experiment.new_batch_trial(trial_type="type2")
        self.experiment.new_batch_trial(trial_type="type2")
        runner1 = self.experiment.runner_for_trial_type(trial_type="type1")
        runner2 = self.experiment.runner_for_trial_type(trial_type="type2")

        with patch.object(
            runner1, "stop", return_value=None
        ) as mock_runner_stop1, patch.object(
            runner2, "stop", return_value=None
        ) as mock_runner_stop2, patch.object(
            BaseTrial, "mark_early_stopped"
        ) as mock_mark_stopped:
            self.experiment.stop_trial_runs(
                trials=[self.experiment.trials[0], self.experiment.trials[1]]
            )
            mock_runner_stop1.assert_called_once()
            mock_runner_stop2.assert_called()
            mock_mark_stopped.assert_called()


class MultiTypeExperimentUtilsTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_multi_type_experiment()
        self.experiment.new_batch_trial(trial_type="type1")
        self.experiment.new_batch_trial(trial_type="type2")

    def test_filter_trials_by_type(self) -> None:
        trials = self.experiment.trials.values()
        self.assertEqual(len(trials), 2)
        filtered = filter_trials_by_type(trials, trial_type="type1")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].trial_type, "type1")
        filtered = filter_trials_by_type(trials, trial_type="type2")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].trial_type, "type2")
        filtered = filter_trials_by_type(trials, trial_type="invalid")
        self.assertEqual(len(filtered), 0)
        filtered = filter_trials_by_type(trials, trial_type=None)
        self.assertEqual(len(filtered), 2)

    def test_get_trial_indices_for_statuses(self) -> None:
        self.assertEqual(
            get_trial_indices_for_statuses(
                experiment=self.experiment,
                statuses={TrialStatus.CANDIDATE, TrialStatus.STAGED},
                trial_type="type1",
            ),
            {0},
        )
        self.assertEqual(
            get_trial_indices_for_statuses(
                experiment=self.experiment,
                statuses={TrialStatus.CANDIDATE, TrialStatus.STAGED},
                trial_type="type2",
            ),
            {1},
        )
        self.assertEqual(
            get_trial_indices_for_statuses(
                experiment=self.experiment,
                statuses={TrialStatus.CANDIDATE, TrialStatus.STAGED},
            ),
            {0, 1},
        )
        self.experiment.trials[0].mark_running(no_runner_required=True)
        self.experiment.trials[1].mark_abandoned()
        self.assertEqual(
            get_trial_indices_for_statuses(
                experiment=self.experiment,
                statuses={TrialStatus.RUNNING},
                trial_type="type1",
            ),
            {0},
        )
        self.assertEqual(
            get_trial_indices_for_statuses(
                experiment=self.experiment,
                statuses={TrialStatus.ABANDONED},
                trial_type="type2",
            ),
            {1},
        )

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.metrics.branin import BraninMetric
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_arms, get_multi_type_experiment


class MultiTypeExperimentTest(TestCase):
    def setUp(self):
        self.experiment = get_multi_type_experiment()

    def testMTExperimentFlow(self):
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
            float(arm_0_slice[df["trial_index"] == 0]["mean"]),
            float(arm_0_slice[df["trial_index"] == 1]["mean"]),
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
            float(arm_0_slice[df["trial_index"] == 0]["mean"]),
            float(arm_0_slice[df["trial_index"] == 1]["mean"]),
            places=10,
        )

    def testRepr(self):
        self.assertEqual(str(self.experiment), "MultiTypeExperiment(test_exp)")

    def testEq(self):
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

    def testBadBehavior(self):
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

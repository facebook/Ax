#!/usr/bin/env python3

from ae.lazarus.ae.core.multi_type_experiment import MultiTypeExperiment
from ae.lazarus.ae.core.objective import Objective
from ae.lazarus.ae.core.optimization_config import OptimizationConfig
from ae.lazarus.ae.metrics.branin import BraninMetric
from ae.lazarus.ae.runners.synthetic import SyntheticRunner
from ae.lazarus.ae.tests.fake import get_branin_arms, get_branin_search_space
from ae.lazarus.ae.utils.common.testutils import TestCase


class MultiTypeExperimentTest(TestCase):
    def setUp(self):
        self.oc = OptimizationConfig(Objective(BraninMetric("m1", ["x1", "x2"])))
        self.experiment = MultiTypeExperiment(
            name="test_exp",
            search_space=get_branin_search_space(),
            default_trial_type="type1",
            default_runner=SyntheticRunner(dummy_metadata="dummy1"),
        )
        self.experiment.optimization_config = self.oc

    def testMTExperimentFlow(self):
        self.experiment.add_trial_type(
            trial_type="type2", runner=SyntheticRunner(dummy_metadata="dummy2")
        )
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

        # Switch the order of variables so metric gives different results
        self.experiment.add_metric(BraninMetric("m2", ["x2", "x1"]), trial_type="type2")

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

        # Set 2 metrics to be equal
        self.experiment.update_metric(
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
        exp2 = MultiTypeExperiment(
            name="test_exp",
            search_space=get_branin_search_space(),
            default_trial_type="type1",
            default_runner=SyntheticRunner(dummy_metadata="dummy1"),
            optimization_config=self.oc,
        )

        self.experiment.add_metric(
            BraninMetric("m2", ["x2", "x1"]), trial_type="type1", canonical_name="m3"
        )

        # Test different metrics
        self.assertFalse(self.experiment == exp2)

        exp2.add_metric(
            BraninMetric("m2", ["x2", "x1"]), trial_type="type1", canonical_name="m4"
        )

        # Test different metrics
        self.assertFalse(self.experiment == exp2)

        exp2.update_metric(
            BraninMetric("m2", ["x2", "x1"]), trial_type="type1", canonical_name="m3"
        )

        # Should be the same
        self.assertTrue(self.experiment == exp2)

    def testBadBehavior(self):
        # Add batch type that already exists
        with self.assertRaises(ValueError):
            self.experiment.add_trial_type("type1", SyntheticRunner())

        # Update runner for non-existent batch type
        with self.assertRaises(ValueError):
            self.experiment.update_runner("type2", SyntheticRunner())

        # Add metric for trial_type that doesn't exist
        with self.assertRaises(ValueError):
            self.experiment.add_metric(BraninMetric("m2", ["x1", "x2"]), "type2")

        # Try to change optimization metric to non-primary batch type
        self.experiment.add_trial_type("type2", SyntheticRunner())
        with self.assertRaises(ValueError):
            self.experiment.update_metric(BraninMetric("m1", ["x1", "x2"]), "type2")

        # Update metric definition for trial_type that doesn't exist
        self.experiment.add_metric(BraninMetric("m2", ["x1", "x2"]), "type2")
        with self.assertRaises(ValueError):
            self.experiment.update_metric(BraninMetric("m2", ["x1", "x2"]), "type3")

        # Try to get runner for trial_type that's not supported
        batch = self.experiment.new_batch_trial()
        batch._trial_type = "type3"  # Force override batch type
        with self.assertRaises(ValueError):
            self.experiment.runner_for_trial(batch)

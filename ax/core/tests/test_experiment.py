#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.parameter import FixedParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.metrics.branin import BraninMetric
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_arm,
    get_branin_arms,
    get_branin_search_space,
    get_data,
    get_experiment,
    get_optimization_config,
    get_search_space,
    get_status_quo,
)


class ExperimentTest(TestCase):
    def setUp(self):
        self.experiment = get_experiment()

    def _setupBraninExperiment(self, n: int) -> Experiment:
        exp = Experiment(
            name="test3",
            search_space=get_branin_search_space(),
            tracking_metrics=[BraninMetric(name="b", param_names=["x1", "x2"])],
            runner=SyntheticRunner(),
        )
        batch = exp.new_batch_trial()
        batch.add_arms_and_weights(arms=get_branin_arms(n=n, seed=0))
        batch.run()

        (
            exp.new_batch_trial()
            .add_arms_and_weights(arms=get_branin_arms(n=3 * n, seed=1))
            .run()
        )
        return exp

    def testExperimentInit(self):
        self.assertEqual(self.experiment.name, "test")
        self.assertEqual(self.experiment.description, "test description")
        self.assertEqual(self.experiment.name, "test")
        self.assertIsNotNone(self.experiment.time_created)
        self.assertEqual(self.experiment.experiment_type, None)
        self.assertEqual(self.experiment.num_abandoned_arms, 0)

    def testExperimentName(self):
        self.assertTrue(self.experiment.has_name)
        self.experiment.name = None
        self.assertFalse(self.experiment.has_name)
        with self.assertRaises(ValueError):
            self.experiment.name
        self.experiment.name = "test"

    def testExperimentType(self):
        self.experiment.experiment_type = "test"
        self.assertEqual(self.experiment.experiment_type, "test")

    def testEq(self):
        self.assertEqual(self.experiment, self.experiment)

        experiment2 = Experiment(
            name="test2",
            search_space=get_search_space(),
            optimization_config=get_optimization_config(),
            status_quo=get_arm(),
            description="test description",
        )
        self.assertNotEqual(self.experiment, experiment2)

    def testTrackingMetricsMerge(self):
        # Tracking and optimization metrics should get merged
        # m1 is on optimization_config while m3 is not
        exp = Experiment(
            name="test2",
            search_space=get_search_space(),
            optimization_config=get_optimization_config(),
            tracking_metrics=[Metric(name="m1"), Metric(name="m3")],
        )
        self.assertEqual(len(exp.optimization_config.metrics) + 1, len(exp.metrics))

    def testBasicBatchCreation(self):
        batch = self.experiment.new_batch_trial()
        self.assertEqual(len(self.experiment.trials), 1)
        self.assertEqual(self.experiment.trials[0], batch)

        # Try (and fail) to re-attach batch
        with self.assertRaises(ValueError):
            self.experiment._attach_trial(batch)

        # Try (and fail) to attach batch to another experiment
        with self.assertRaises(ValueError):
            new_exp = get_experiment()
            new_exp._attach_trial(batch)

    def testRepr(self):
        self.assertEqual("Experiment(test)", str(self.experiment))

    def testBasicProperties(self):
        self.assertEqual(self.experiment.status_quo, get_status_quo())
        self.assertEqual(self.experiment.search_space, get_search_space())
        self.assertEqual(self.experiment.optimization_config, get_optimization_config())
        self.assertEqual(self.experiment.is_test, True)

    def testMetricSetters(self):
        # Establish current metrics size
        self.assertEqual(
            len(get_optimization_config().metrics) + 1, len(self.experiment.metrics)
        )

        # Add optimization config with 1 different metric
        opt_config = get_optimization_config()
        opt_config.outcome_constraints[0].metric = Metric(name="m3")
        self.experiment.optimization_config = opt_config

        # Verify total metrics size is the same.
        self.assertEqual(
            len(get_optimization_config().metrics) + 1, len(self.experiment.metrics)
        )

        # Test adding new tracking metric
        self.experiment.add_tracking_metric(Metric(name="m4"))
        self.assertEqual(
            len(get_optimization_config().metrics) + 2, len(self.experiment.metrics)
        )

        # Test adding new tracking metrics
        self.experiment.add_tracking_metrics([Metric(name="z1")])
        self.assertEqual(
            len(get_optimization_config().metrics) + 3, len(self.experiment.metrics)
        )

        # Verify update_tracking_metric updates the metric definition
        self.assertIsNone(self.experiment.metrics["m4"].lower_is_better)
        self.experiment.update_tracking_metric(Metric(name="m4", lower_is_better=True))
        self.assertTrue(self.experiment.metrics["m4"].lower_is_better)

        # Verify unable to add existing metric
        with self.assertRaises(ValueError):
            self.experiment.add_tracking_metric(Metric(name="m4"))

        # Verify unable to add existing metric
        with self.assertRaises(ValueError):
            self.experiment.add_tracking_metrics([Metric(name="z1"), Metric(name="m4")])

        # Verify unable to add metric in optimization config
        with self.assertRaises(ValueError):
            self.experiment.add_tracking_metric(Metric(name="m1"))

        # Verify unable to add metric in optimization config
        with self.assertRaises(ValueError):
            self.experiment.add_tracking_metrics([Metric(name="z2"), Metric(name="m1")])

        # Cannot update metric not already on experiment
        with self.assertRaises(ValueError):
            self.experiment.update_tracking_metric(Metric(name="m5"))

        # Cannot remove metric not already on experiment
        with self.assertRaises(ValueError):
            self.experiment.remove_tracking_metric(metric_name="m5")

    def testSearchSpaceSetter(self):
        one_param_ss = SearchSpace(parameters=[get_search_space().parameters["w"]])

        # Verify all search space ok with no trials
        self.experiment.search_space = one_param_ss
        self.assertEqual(len(self.experiment.parameters), 1)

        # Reset search space and add batch to trigger validations
        self.experiment.search_space = get_search_space()
        self.experiment.new_batch_trial()

        # Try search space with too few parameters
        with self.assertRaises(ValueError):
            self.experiment.search_space = one_param_ss

        # Try search space with different type
        bad_type_ss = get_search_space()
        bad_type_ss.parameters["x"]._parameter_type = ParameterType.FLOAT
        with self.assertRaises(ValueError):
            self.experiment.search_space = bad_type_ss

        # Try search space with additional parameters
        extra_param_ss = get_search_space()
        extra_param_ss.add_parameter(FixedParameter("l", ParameterType.FLOAT, 0.5))
        with self.assertRaises(ValueError):
            self.experiment.search_space = extra_param_ss

    def testStatusQuoSetter(self):
        sq_parameters = self.experiment.status_quo.parameters
        self.experiment.status_quo = None
        self.assertIsNone(self.experiment.status_quo)

        # Verify normal update
        sq_parameters["w"] = 3.5
        self.experiment.status_quo = Arm(sq_parameters)
        self.assertEqual(self.experiment.status_quo.parameters["w"], 3.5)
        self.assertEqual(self.experiment.status_quo.name, "status_quo")
        self.assertTrue("status_quo" in self.experiment.arms_by_name)

        # Verify all None values
        self.experiment.status_quo = Arm({n: None for n in sq_parameters.keys()})
        self.assertIsNone(self.experiment.status_quo.parameters["w"])

        # Try extra param
        sq_parameters["a"] = 4
        with self.assertRaises(ValueError):
            self.experiment.status_quo = Arm(sq_parameters)

        # Try wrong type
        sq_parameters.pop("a")
        sq_parameters["w"] = "hello"
        with self.assertRaises(ValueError):
            self.experiment.status_quo = Arm(sq_parameters)

        # Verify arms_by_signature, arms_by_name only contains status_quo
        self.assertEqual(len(self.experiment.arms_by_signature), 1)
        self.assertEqual(len(self.experiment.arms_by_name), 1)

        # Change status quo, verify still just 1 arm
        sq_parameters["w"] = 3.6
        self.experiment.status_quo = Arm(sq_parameters)
        self.assertEqual(len(self.experiment.arms_by_signature), 1)
        self.assertEqual(len(self.experiment.arms_by_name), 1)

        # Make a batch, add status quo to it, then change exp status quo, verify 2 arms
        batch = self.experiment.new_batch_trial()
        batch.set_status_quo_with_weight(self.experiment.status_quo, 1)
        sq_parameters["w"] = 3.7
        self.experiment.status_quo = Arm(sq_parameters)
        self.assertEqual(len(self.experiment.arms_by_signature), 2)
        self.assertEqual(len(self.experiment.arms_by_name), 2)
        self.assertEqual(self.experiment.status_quo.name, "status_quo_e0")
        self.assertTrue("status_quo_e0" in self.experiment.arms_by_name)

        # Try missing param
        sq_parameters.pop("w")
        with self.assertRaises(ValueError):
            self.experiment.status_quo = Arm(sq_parameters)

        # Actually name the status quo.
        exp = Experiment(
            name="test3",
            search_space=get_branin_search_space(),
            tracking_metrics=[BraninMetric(name="b", param_names=["x1", "x2"])],
            runner=SyntheticRunner(),
        )
        batch = exp.new_batch_trial()
        arms = get_branin_arms(n=1, seed=0)
        batch.add_arms_and_weights(arms=arms)
        self.assertIsNone(exp.status_quo)
        exp.status_quo = arms[0]
        self.assertEqual(exp.status_quo.name, "0_0")

        # Try setting sq to existing arm with different name
        with self.assertRaises(ValueError):
            exp.status_quo = Arm(arms[0].parameters, name="new_name")

    def testRegisterArm(self):
        # Create a new arm, register on experiment
        parameters = self.experiment.status_quo.parameters
        parameters["w"] = 3.5
        arm = Arm(name="my_arm_name", parameters=parameters)
        self.experiment._register_arm(arm)
        self.assertEqual(self.experiment.arms_by_name[arm.name], arm)
        self.assertEqual(self.experiment.arms_by_signature[arm.signature], arm)

    def testFetchAndStoreData(self):
        n = 10
        exp = self._setupBraninExperiment(n)
        batch = exp.trials[0]

        # Test fetch data
        batch_data = batch.fetch_data()
        self.assertEqual(len(batch_data.df), n)

        exp_data = exp.fetch_data()
        exp_data2 = exp.metrics["b"].fetch_experiment_data(exp)
        self.assertEqual(len(exp_data2.df), 4 * n)
        self.assertEqual(len(exp_data.df), 4 * n)
        self.assertEqual(len(exp.arms_by_name), 4 * n)

        # Verify data lookup is empty
        self.assertEqual(len(exp.lookup_data_for_trial(0)[0].df), 0)

        # Test local storage
        t1 = exp.attach_data(batch_data)
        t2 = exp.attach_data(exp_data)

        full_dict = exp.data_by_trial
        self.assertEqual(len(full_dict), 2)  # data for 2 trials
        self.assertEqual(len(full_dict[0]), 2)  # 2 data objs for batch 0

        # Test retrieving original batch 0 data
        self.assertEqual(len(exp.lookup_data_for_ts(t1).df), n)
        self.assertEqual(len(exp.lookup_data_for_trial(0)[0].df), n)

        # Test retrieving full exp data
        self.assertEqual(len(exp.lookup_data_for_ts(t2).df), 4 * n)

        with self.assertRaisesRegex(ValueError, ".* for metric"):
            exp.attach_data(batch_data, combine_with_last_data=True)

        new_data = Data(
            df=pd.DataFrame.from_records(
                [{"arm_name": "0_0", "metric_name": "z", "mean": 3, "trial_index": 0}]
            )
        )
        t3 = exp.attach_data(new_data, combine_with_last_data=True)
        self.assertEqual(len(full_dict[0]), 3)  # 3 data objs for batch 0 now
        self.assertIn("z", exp.lookup_data_for_ts(t3).df["metric_name"].tolist())

        # Verify we don't get the data if the trial is abandoned
        batch._status = TrialStatus.ABANDONED
        self.assertEqual(len(batch.fetch_data().df), 0)
        self.assertEqual(len(exp.fetch_data().df), 3 * n)

        # For `CANDIDATE` trials, we append attached data to fetched data,
        # so the attached data row with metric name "z" should appear in fetched
        # data.
        batch._status = TrialStatus.CANDIDATE
        self.assertEqual(len(batch.fetch_data().df), n + 1)
        # n arms in trial #0, 3 * n arms in trial #1
        self.assertEqual(len(exp.fetch_data().df), 4 * n + 1)
        metrics_in_data = set(batch.fetch_data().df["metric_name"].values)
        self.assertEqual(metrics_in_data, {"b", "z"})

        # Verify we do get the stored data if there are an unimplemented metrics.
        del exp._data_by_trial[0][t3]  # Remove attached data for nonexistent metric.
        exp.remove_tracking_metric(metric_name="b")  # Remove implemented metric.
        exp.add_tracking_metric(Metric(name="dummy"))  # Add unimplemented metric.
        batch._status = TrialStatus.RUNNING
        # Data should be getting looked up now.
        self.assertEqual(batch.fetch_data(), exp.lookup_data_for_ts(t1))
        self.assertEqual(exp.fetch_data(), exp.lookup_data_for_ts(t2))
        metrics_in_data = set(batch.fetch_data().df["metric_name"].values)
        # Data for metric "z" should no longer be present since we removed it.
        self.assertEqual(metrics_in_data, {"b"})

        # Check that error will be raised if dummy and implemented metrics are
        # fetched at once.
        with self.assertRaisesRegex(ValueError, "Unexpected combination"):
            exp.fetch_data(
                [BraninMetric(name="b", param_names=["x1", "x2"]), Metric(name="m")]
            )

    def testEmptyMetrics(self):
        empty_experiment = Experiment(
            name="test_experiment", search_space=get_search_space()
        )
        self.assertEqual(empty_experiment.num_trials, 0)
        with self.assertRaises(ValueError):
            empty_experiment.fetch_data()
        batch = empty_experiment.new_batch_trial()
        self.assertEqual(empty_experiment.num_trials, 1)
        with self.assertRaises(ValueError):
            batch.fetch_data()
        empty_experiment.add_tracking_metric(Metric(name="some_metric"))
        empty_experiment.attach_data(get_data())
        self.assertFalse(empty_experiment.fetch_data().df.empty)

    def testNumArmsNoDeduplication(self):
        exp = Experiment(name="test_experiment", search_space=get_search_space())
        arm = get_arm()
        exp.new_batch_trial().add_arm(arm)
        trial = exp.new_batch_trial().add_arm(arm)
        self.assertEqual(exp.sum_trial_sizes, 2)
        self.assertEqual(len(exp.arms_by_name), 1)
        trial.mark_arm_abandoned(trial.arms[0].name)
        self.assertEqual(exp.num_abandoned_arms, 1)

    def testExperimentWithoutName(self):
        exp = Experiment(
            search_space=get_branin_search_space(),
            tracking_metrics=[BraninMetric(name="b", param_names=["x1", "x2"])],
            runner=SyntheticRunner(),
        )
        self.assertEqual("Experiment(None)", str(exp))
        batch = exp.new_batch_trial()
        batch.add_arms_and_weights(arms=get_branin_arms(n=5, seed=0))
        batch.run()
        self.assertEqual(batch.run_metadata, {"name": "0"})

    def testExperimentRunner(self):
        original_runner = SyntheticRunner()
        self.experiment.runner = original_runner
        batch = self.experiment.new_batch_trial()
        batch.run()
        self.assertEqual(batch.runner, original_runner)

        # Simulate a failed run/deployment, in which the runner is attached
        # but the actual run fails, and so the trial remains CANDIDATE.
        candidate_batch = self.experiment.new_batch_trial()
        candidate_batch.run()
        candidate_batch._status = TrialStatus.CANDIDATE
        self.assertEqual(self.experiment.trials_expecting_data, [batch])
        tbs = self.experiment.trials_by_status  # All statuses should be present
        self.assertEqual(len(tbs), len(TrialStatus))
        self.assertEqual(tbs[TrialStatus.RUNNING], [batch])
        self.assertEqual(tbs[TrialStatus.CANDIDATE], [candidate_batch])
        tibs = self.experiment.trial_indices_by_status
        self.assertEqual(len(tibs), len(TrialStatus))
        self.assertEqual(tibs[TrialStatus.RUNNING], {0})
        self.assertEqual(tibs[TrialStatus.CANDIDATE], {1})

        identifier = {"new_runner": True}
        new_runner = SyntheticRunner(dummy_metadata=identifier)

        self.experiment.reset_runners(new_runner)
        # Don't update trials that have been run.
        self.assertEqual(batch.runner, original_runner)
        # Update default runner
        self.assertEqual(self.experiment.runner, new_runner)
        # Update candidate trial runners.
        self.assertEqual(self.experiment.trials[1].runner, new_runner)

    def testFetchTrialsData(self):
        exp = self._setupBraninExperiment(n=5)
        batch_0 = exp.trials[0]
        batch_1 = exp.trials[1]
        batch_0_data = exp.fetch_trials_data(trial_indices=[0])
        self.assertEqual(set(batch_0_data.df["trial_index"].values), {0})
        self.assertEqual(
            set(batch_0_data.df["arm_name"].values), {a.name for a in batch_0.arms}
        )
        batch_1_data = exp.fetch_trials_data(trial_indices=[1])
        self.assertEqual(set(batch_1_data.df["trial_index"].values), {1})
        self.assertEqual(
            set(batch_1_data.df["arm_name"].values), {a.name for a in batch_1.arms}
        )
        self.assertEqual(
            exp.fetch_trials_data(trial_indices=[0, 1]),
            Data.from_multiple_data([batch_0_data, batch_1_data]),
        )
        with self.assertRaisesRegex(ValueError, ".* not associated .*"):
            exp.fetch_trials_data(trial_indices=[2])
        # Try to fetch data when there are only metrics and no attached data.
        exp.remove_tracking_metric(metric_name="b")  # Remove implemented metric.
        exp.add_tracking_metric(Metric(name="dummy"))  # Add unimplemented metric.
        self.assertTrue(exp.fetch_trials_data(trial_indices=[0]).df.empty)
        # Try fetching attached data.
        exp.attach_data(batch_0_data)
        exp.attach_data(batch_1_data)
        self.assertEqual(exp.fetch_trials_data(trial_indices=[0]), batch_0_data)
        self.assertEqual(exp.fetch_trials_data(trial_indices=[1]), batch_1_data)
        self.assertEqual(set(batch_0_data.df["trial_index"].values), {0})
        self.assertEqual(
            set(batch_0_data.df["arm_name"].values), {a.name for a in batch_0.arms}
        )

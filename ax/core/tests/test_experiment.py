#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.parameter import FixedParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.metrics.branin import BraninMetric
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.fake import (
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

        # Verify update_tracking_metric updates the metric definition
        self.assertIsNone(self.experiment.metrics["m4"].lower_is_better)
        self.experiment.update_tracking_metric(Metric(name="m4", lower_is_better=True))
        self.assertTrue(self.experiment.metrics["m4"].lower_is_better)

        # Verify unable to add existing metric
        with self.assertRaises(ValueError):
            self.experiment.add_tracking_metric(Metric(name="m4"))

        # Verify unable to add metric in optimization config
        with self.assertRaises(ValueError):
            self.experiment.add_tracking_metric(Metric(name="m1"))

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

        # Verify arms_by_signature only contains status_quo
        self.assertEqual(len(self.experiment.arms_by_signature), 1)

        # Change status quo, verify still just 1 arm
        sq_parameters["w"] = 3.6
        self.experiment.status_quo = Arm(sq_parameters)
        self.assertEqual(len(self.experiment.arms_by_signature), 1)

        # Make a batch, then change exp status quo, verify 2 arms
        self.experiment.new_batch_trial()
        sq_parameters["w"] = 3.7
        self.experiment.status_quo = Arm(sq_parameters)
        self.assertEqual(len(self.experiment.arms_by_signature), 2)
        self.assertEqual(self.experiment.status_quo.name, "status_quo_e0")

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

    def testFetchAndStoreData(self):
        n = 10
        exp = self._setupBraninExperiment(n)
        batch = exp.trials[0]

        # Test fetch data
        batch_data = batch.fetch_data()
        self.assertEqual(len(batch_data.df), n)

        exp_data = exp.fetch_data()
        self.assertEqual(len(exp_data.df), 4 * n)
        self.assertEqual(len(exp.arms_by_name), 4 * n)

        # Verify data lookup is empty
        self.assertEqual(len(exp.lookup_data_for_trial(0).df), 0)

        # Test local storage
        t1 = exp.attach_data(batch_data)

        t2 = exp.attach_data(exp_data)

        full_dict = exp.data_by_trial
        self.assertEqual(len(full_dict), 2)  # data for 2 trials
        self.assertEqual(len(full_dict[0]), 2)  # 2 data objs for batch 0

        # Test retrieving original batch 0 data
        self.assertEqual(len(exp.lookup_data_for_ts(t1).df), n)
        self.assertEqual(len(exp.lookup_data_for_trial(0).df), n)

        # Test retrieving full exp data
        self.assertEqual(len(exp.lookup_data_for_ts(t2).df), 4 * n)

        # Verify we don't get the data if the trial is abandoned
        batch._status = TrialStatus.ABANDONED
        self.assertEqual(len(batch.fetch_data().df), 0)
        self.assertEqual(len(exp.fetch_data().df), 3 * n)

        # Verify we do get the data if the trial is a candidate
        batch._status = TrialStatus.CANDIDATE
        self.assertEqual(len(batch.fetch_data().df), n)
        self.assertEqual(len(exp.fetch_data().df), 4 * n)

        # Verify we do get the stored data if there is an unimplemented metric
        batch._status = TrialStatus.RUNNING
        exp.add_tracking_metric(Metric(name="m"))
        self.assertEqual(len(batch.fetch_data().df), n)
        self.assertEqual(len(exp.fetch_data().df), 4 * n)

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

    def testExperimentWithoutName(self) -> Experiment:
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

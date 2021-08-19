#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Type
from unittest.mock import patch

import pandas as pd
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric
from ax.core.parameter import FixedParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.metrics.branin import BraninMetric
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.constants import Keys, EXPERIMENT_IS_TEST_WARNING
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_arm,
    get_branin_arms,
    get_branin_optimization_config,
    get_branin_search_space,
    get_branin_experiment,
    get_branin_experiment_with_timestamp_map_metric,
    get_data,
    get_experiment,
    get_experiment_with_map_data_type,
    get_optimization_config,
    get_search_space,
    get_sobol,
    get_status_quo,
    get_scalarized_outcome_constraint,
)

DUMMY_RUN_METADATA = {"test_run_metadata_key": "test_run_metadata_value"}


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

        batch_2 = exp.new_batch_trial()
        batch_2.add_arms_and_weights(arms=get_branin_arms(n=3 * n, seed=1))
        batch_2.run()
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

    def testDBId(self):
        self.assertIsNone(self.experiment.db_id)
        some_id = 123456789
        self.experiment.db_id = some_id
        self.assertEqual(self.experiment.db_id, some_id)

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

        # Add optimization config with 1 scalarized constraint composed of 2 metrics
        opt_config = get_optimization_config()
        opt_config.outcome_constraints = opt_config.outcome_constraints + [
            get_scalarized_outcome_constraint()
        ]
        self.experiment.optimization_config = opt_config

        # Verify total metrics size is the same.
        self.assertEqual(len(opt_config.metrics) + 1, len(self.experiment.metrics))
        self.assertEqual(
            len(get_optimization_config().metrics) + 3, len(self.experiment.metrics)
        )
        # set back
        self.experiment.optimization_config = get_optimization_config()

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
        batch.mark_completed()

        # Test fetch data
        batch_data = batch.fetch_data()
        self.assertEqual(len(batch_data.df), n)

        exp_data = exp.fetch_data()
        exp_data2 = exp.metrics["b"].fetch_experiment_data(exp)
        self.assertEqual(len(exp_data2.df), 4 * n)
        self.assertEqual(len(exp_data.df), 4 * n)
        self.assertEqual(len(exp.arms_by_name), 4 * n)

        # Verify that `metrics` kwarg to `experiment.fetch_data` is respected.
        exp.add_tracking_metric(Metric(name="not_yet_on_experiment"))
        exp.attach_data(
            Data(
                df=pd.DataFrame.from_records(
                    [
                        {
                            "arm_name": "0_0",
                            "metric_name": "not_yet_on_experiment",
                            "mean": 3,
                            "sem": 0,
                            "trial_index": 0,
                        }
                    ]
                )
            )
        )
        self.assertEqual(
            set(
                exp.fetch_data(metrics=[Metric(name="not_yet_on_experiment")])
                .df["metric_name"]
                .values
            ),
            {"not_yet_on_experiment"},
        )

        # Verify data lookup includes trials attached from `fetch_data`.
        self.assertEqual(len(exp.lookup_data_for_trial(1)[0].df), 30)

        # Test local storage
        t1 = exp.attach_data(batch_data)
        t2 = exp.attach_data(exp_data)

        full_dict = exp.data_by_trial
        self.assertEqual(len(full_dict), 2)  # data for 2 trials
        self.assertEqual(len(full_dict[0]), 5)  # 5 data objs for batch 0

        # Test retrieving original batch 0 data
        self.assertEqual(len(exp.lookup_data_for_ts(t1).df), n)
        self.assertEqual(len(exp.lookup_data_for_trial(0)[0].df), n)

        # Test retrieving full exp data
        self.assertEqual(len(exp.lookup_data_for_ts(t2).df), 4 * n)

        with self.assertRaisesRegex(ValueError, ".* for metric"):
            exp.attach_data(batch_data, combine_with_last_data=True)

        self.assertEqual(len(full_dict[0]), 5)  # 5 data objs for batch 0
        new_data = Data(
            df=pd.DataFrame.from_records(
                [
                    {
                        "arm_name": "0_0",
                        "metric_name": "z",
                        "mean": 3,
                        "sem": 0,
                        "trial_index": 0,
                    }
                ]
            )
        )
        t3 = exp.attach_data(new_data, combine_with_last_data=True)
        # still 5 data objs, since we combined last one
        self.assertEqual(len(full_dict[0]), 5)
        self.assertIn("z", exp.lookup_data_for_ts(t3).df["metric_name"].tolist())

        # Verify we don't get the data if the trial is abandoned
        batch._status = TrialStatus.ABANDONED
        self.assertEqual(len(batch.fetch_data().df), 0)
        self.assertEqual(len(exp.fetch_data().df), 3 * n)

        # Verify we do get the stored data if there are an unimplemented metrics.
        del exp._data_by_trial[0][t3]  # Remove attached data for nonexistent metric.
        # Remove implemented metric that is `available_while_running`
        # (and therefore not pulled from cache).
        exp.remove_tracking_metric(metric_name="b")
        exp.add_tracking_metric(Metric(name="b"))  # Add unimplemented metric.
        batch._status = TrialStatus.COMPLETED
        # Data should be getting looked up now.
        self.assertEqual(batch.fetch_data(), exp.lookup_data_for_ts(t1))
        self.assertEqual(exp.fetch_data(), exp.lookup_data_for_ts(t1))
        metrics_in_data = set(batch.fetch_data().df["metric_name"].values)
        # Data for metric "z" should no longer be present since we removed it.
        self.assertEqual(metrics_in_data, {"b"})

        # Verify that `metrics` kwarg to `experiment.fetch_data` is respected
        # when pulling looked-up data.
        self.assertEqual(
            exp.fetch_data(metrics=[Metric(name="not_on_experiment")]), Data()
        )

    def testOverwriteExistingData(self):
        n = 10
        exp = self._setupBraninExperiment(n)

        # automatically attaches data
        data = exp.fetch_data()

        # can't set both combine_with_last_data and overwrite_existing_data
        with self.assertRaises(UnsupportedError):
            exp.attach_data(
                data, combine_with_last_data=True, overwrite_existing_data=True
            )

        # data exists for two trials
        # data has been attached once for each trial
        self.assertEqual(len(exp._data_by_trial), 2)
        self.assertEqual(len(exp._data_by_trial[0]), 1)
        self.assertEqual(len(exp._data_by_trial[1]), 1)

        exp.attach_data(data)
        # data has been attached twice for each trial
        self.assertEqual(len(exp._data_by_trial), 2)
        self.assertEqual(len(exp._data_by_trial[0]), 2)
        self.assertEqual(len(exp._data_by_trial[1]), 2)

        ts = exp.attach_data(data, overwrite_existing_data=True)
        # previous two attachment are overwritten,
        # now only one data (most recent one) per trial
        self.assertEqual(len(exp._data_by_trial), 2)
        self.assertEqual(len(exp._data_by_trial[0]), 1)
        self.assertEqual(len(exp._data_by_trial[1]), 1)
        self.assertTrue(ts in exp._data_by_trial[0])
        self.assertTrue(ts in exp._data_by_trial[1])

    def testEmptyMetrics(self):
        empty_experiment = Experiment(
            name="test_experiment", search_space=get_search_space()
        )
        self.assertEqual(empty_experiment.num_trials, 0)
        with self.assertRaises(ValueError):
            empty_experiment.fetch_data()
        batch = empty_experiment.new_batch_trial()
        batch.mark_running(no_runner_required=True)
        self.assertEqual(empty_experiment.num_trials, 1)
        with self.assertRaises(ValueError):
            batch.fetch_data()
        empty_experiment.add_tracking_metric(Metric(name="ax_test_metric"))
        self.assertTrue(empty_experiment.fetch_data().df.empty)
        empty_experiment.attach_data(get_data())
        batch.mark_completed()
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
        batch_0.mark_completed()
        batch_1.mark_completed()
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

        # Since NoisyFunction metric has overwrite_existing_data = False,
        # we should have two dfs per trial now
        self.assertEqual(len(exp.data_by_trial[0]), 2)

        with self.assertRaisesRegex(ValueError, ".* not associated .*"):
            exp.fetch_trials_data(trial_indices=[2])
        # Try to fetch data when there are only metrics and no attached data.
        exp.remove_tracking_metric(metric_name="b")  # Remove implemented metric.
        exp.add_tracking_metric(Metric(name="b"))  # Add unimplemented metric.
        self.assertEqual(len(exp.fetch_trials_data(trial_indices=[0]).df), 5)
        # Try fetching attached data.
        exp.attach_data(batch_0_data)
        exp.attach_data(batch_1_data)
        self.assertEqual(exp.fetch_trials_data(trial_indices=[0]), batch_0_data)
        self.assertEqual(exp.fetch_trials_data(trial_indices=[1]), batch_1_data)
        self.assertEqual(set(batch_0_data.df["trial_index"].values), {0})
        self.assertEqual(
            set(batch_0_data.df["arm_name"].values), {a.name for a in batch_0.arms}
        )

    def test_immutable_search_space_and_opt_config(self):
        mutable_exp = self._setupBraninExperiment(n=5)
        self.assertFalse(mutable_exp.immutable_search_space_and_opt_config)
        immutable_exp = Experiment(
            name="test4",
            search_space=get_branin_search_space(),
            tracking_metrics=[BraninMetric(name="b", param_names=["x1", "x2"])],
            optimization_config=get_branin_optimization_config(),
            runner=SyntheticRunner(),
            properties={Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF: True},
        )
        self.assertTrue(immutable_exp.immutable_search_space_and_opt_config)
        immutable_exp.new_batch_trial()
        with self.assertRaises(UnsupportedError):
            immutable_exp.optimization_config = get_branin_optimization_config()
        with self.assertRaises(UnsupportedError):
            immutable_exp.search_space = get_branin_search_space()

        # Check that passing the property as just a string is processed
        # correctly.
        immutable_exp_2 = Experiment(
            name="test4",
            search_space=get_branin_search_space(),
            tracking_metrics=[BraninMetric(name="b", param_names=["x1", "x2"])],
            runner=SyntheticRunner(),
            properties={Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF.value: True},
        )
        self.assertTrue(immutable_exp_2.immutable_search_space_and_opt_config)

    def test_fetch_as_class(self):
        class MyMetric(Metric):
            @property
            def fetch_multi_group_by_metric(self) -> Type[Metric]:
                return Metric

        m = MyMetric(name="test_metric")
        exp = Experiment(
            name="test",
            search_space=get_branin_search_space(),
            tracking_metrics=[m],
            runner=SyntheticRunner(),
        )
        self.assertEqual(exp._metrics_by_class(), {Metric: [m]})

    @patch(
        # No-op mock just to record calls to `fetch_experiment_data_multi`.
        f"{BraninMetric.__module__}.BraninMetric.fetch_experiment_data_multi",
        side_effect=BraninMetric.fetch_experiment_data_multi,
    )
    def test_prefer_lookup_where_possible(self, mock_fetch_exp_data_multi):
        # By default, `BraninMetric` is available while trial is running.
        exp = self._setupBraninExperiment(n=5)
        exp.fetch_data()
        # Since metric is available while trial is running, we should be
        # refetching the data and no data should be attached to experiment.
        mock_fetch_exp_data_multi.assert_called_once()
        self.assertEqual(len(exp._data_by_trial), 2)

        with patch(
            f"{BraninMetric.__module__}.BraninMetric.is_available_while_running",
            return_value=False,
        ):
            exp = self._setupBraninExperiment(n=5)
            exp.fetch_data()
            # 1. No completed trials => no fetch case.
            mock_fetch_exp_data_multi.reset_mock()
            dat = exp.fetch_data()
            mock_fetch_exp_data_multi.assert_not_called()
            # Data should be empty since there are no completed trials.
            self.assertTrue(dat.df.empty)

            # 2. Newly completed trials => fetch case.
            mock_fetch_exp_data_multi.reset_mock()
            exp.trials.get(0).mark_completed()
            exp.trials.get(1).mark_completed()
            dat = exp.fetch_data()
            # `fetch_experiment_data_multi` should be called N=number of trials times.
            self.assertEqual(len(mock_fetch_exp_data_multi.call_args_list), 2)
            # Data should no longer be empty since there are completed trials.
            self.assertFalse(dat.df.empty)
            # Data for two trials should get attached.
            self.assertEqual(len(exp._data_by_trial), 2)

            # 3. Previously fetched => look up in cache case.
            mock_fetch_exp_data_multi.reset_mock()
            # All fetched data should get cached, so no fetch should happen next time.
            exp.fetch_data()
            mock_fetch_exp_data_multi.assert_not_called()

    def testWarmStartFromOldExperiment(self):
        # create old_experiment
        len_old_trials = 5
        i_failed_trial = 3
        old_experiment = get_branin_experiment()
        for i_old_trial in range(len_old_trials):
            sobol_run = get_sobol(search_space=old_experiment.search_space).gen(n=1)
            trial = old_experiment.new_trial(generator_run=sobol_run)
            trial.mark_running(no_runner_required=True)
            if i_old_trial == i_failed_trial:
                trial.mark_failed()
            else:
                trial.mark_completed()
        # make metric noiseless for exact reproducibility
        old_experiment.optimization_config.objective.metric.noise_sd = 0
        old_experiment.fetch_data()

        # should fail if new_experiment has trials
        new_experiment = get_branin_experiment(with_trial=True)
        with self.assertRaisesRegex(ValueError, "Experiment.*has.*trials"):
            new_experiment.warm_start_from_old_experiment(old_experiment=old_experiment)

        # should fail if search spaces are different
        with self.assertRaisesRegex(ValueError, "mismatch in search space parameters"):
            self.experiment.warm_start_from_old_experiment(
                old_experiment=old_experiment
            )

        # check that all non-failed trials are copied to new_experiment
        new_experiment = get_branin_experiment()
        # make metric noiseless for exact reproducibility
        new_experiment.optimization_config.objective.metric.noise_sd = 0
        for _, trial in old_experiment.trials.items():
            trial._run_metadata = DUMMY_RUN_METADATA
        new_experiment.warm_start_from_old_experiment(
            old_experiment=old_experiment, copy_run_metadata=True
        )
        self.assertEqual(len(new_experiment.trials), len(old_experiment.trials) - 1)
        i_old_trial = 0
        for _, trial in new_experiment.trials.items():
            # skip failed trial
            i_old_trial += i_old_trial == i_failed_trial
            self.assertEqual(
                trial.arm.parameters, old_experiment.trials[i_old_trial].arm.parameters
            )
            self.assertRegex(
                trial._properties["source"], "Warm start.*Experiment.*trial"
            )
            self.assertDictEqual(trial.run_metadata, DUMMY_RUN_METADATA)
            i_old_trial += 1

        # Check that the data was attached for correct trials
        old_df = old_experiment.fetch_data().df
        new_df = new_experiment.fetch_data().df

        self.assertEqual(len(new_df), len_old_trials - 1)
        pd.testing.assert_frame_equal(
            old_df.drop(["arm_name", "trial_index"], axis=1),
            new_df.drop(["arm_name", "trial_index"], axis=1),
        )

    def test_is_test_warning(self):
        experiments_module = "ax.core.experiment"
        with self.subTest("it warns on construction for a test"):
            with self.assertLogs(experiments_module, level=logging.INFO) as logger:
                exp = Experiment(
                    search_space=get_search_space(),
                    is_test=True,
                )
                self.assertIn(
                    f"INFO:{experiments_module}:{EXPERIMENT_IS_TEST_WARNING}",
                    logger.output,
                )

        with self.subTest("it does not warn on construction for a non test"):
            with self.assertLogs(experiments_module, level=logging.INFO) as logger:
                logging.getLogger(experiments_module).info(
                    "there must be at least one log or the assertLogs statement fails"
                )
                exp = Experiment(
                    search_space=get_search_space(),
                    is_test=False,
                )
                self.assertNotIn(
                    f"INFO:{experiments_module}:{EXPERIMENT_IS_TEST_WARNING}",
                    logger.output,
                )

        with self.subTest("it warns on setting is_test to True"):
            with self.assertLogs(experiments_module, level=logging.INFO) as logger:
                exp.is_test = True
                self.assertIn(
                    f"INFO:{experiments_module}:{EXPERIMENT_IS_TEST_WARNING}",
                    logger.output,
                )

        with self.subTest("it does not warn on setting is_test to False"):
            with self.assertLogs(experiments_module, level=logging.INFO) as logger:
                logging.getLogger(experiments_module).info(
                    "there must be at least one log or the assertLogs statement fails"
                )
                exp.is_test = False
                self.assertNotIn(
                    f"INFO:{experiments_module}:{EXPERIMENT_IS_TEST_WARNING}",
                    logger.output,
                )


class ExperimentWithMapDataTest(TestCase):
    def setUp(self):
        self.experiment = get_experiment_with_map_data_type()

    def _setupBraninExperiment(self, n: int, incremental: bool = False) -> Experiment:
        exp = get_branin_experiment_with_timestamp_map_metric(incremental=incremental)
        batch = exp.new_batch_trial()
        batch.add_arms_and_weights(arms=get_branin_arms(n=n, seed=0))
        batch.run()

        batch_2 = exp.new_batch_trial()
        batch_2.add_arms_and_weights(arms=get_branin_arms(n=3 * n, seed=1))
        batch_2.run()
        return exp

    def testFetchDataWithMapData(self):
        evaluations = {
            "0_0": [
                ({"epoch": 1}, {"no_fetch_impl_metric": (3.7, 0.5)}),
                ({"epoch": 2}, {"no_fetch_impl_metric": (3.8, 0.5)}),
                ({"epoch": 3}, {"no_fetch_impl_metric": (3.9, 0.5)}),
                ({"epoch": 4}, {"no_fetch_impl_metric": (4.0, 0.5)}),
            ],
        }

        self.experiment.add_tracking_metric(
            metric=MapMetric(name="no_fetch_impl_metric")
        )
        self.experiment.new_trial()
        self.experiment.trials[0].mark_running(no_runner_required=True)
        first_epoch = MapData.from_map_evaluations(
            evaluations={
                arm_name: partial_results[0:1]
                for arm_name, partial_results in evaluations.items()
            },
            trial_index=0,
        )
        self.experiment.attach_data(first_epoch)
        remaining_epochs = MapData.from_map_evaluations(
            evaluations={
                arm_name: partial_results[1:4]
                for arm_name, partial_results in evaluations.items()
            },
            trial_index=0,
        )
        self.experiment.attach_data(remaining_epochs)
        self.experiment.trials[0].mark_completed()

        expected_data = remaining_epochs
        actual_data = self.experiment.lookup_data()
        self.assertEqual(expected_data, actual_data)

    def testFetchTrialsData(self):
        exp = self._setupBraninExperiment(n=5)
        batch_0 = exp.trials[0]
        batch_1 = exp.trials[1]
        batch_0.mark_completed()
        batch_1.mark_completed()
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
            MapData.from_multiple_data([batch_0_data, batch_1_data]),
        )

        # Since NoisyFunctionMap metric has overwrite_existing_data = True,
        # we should only have one df per trial now
        self.assertEqual(len(exp.data_by_trial[0]), 1)

        with self.assertRaisesRegex(ValueError, ".* not associated .*"):
            exp.fetch_trials_data(trial_indices=[2])
        # Try to fetch data when there are only metrics and no attached data.
        exp.remove_tracking_metric(metric_name="b")  # Remove implemented metric.
        exp.add_tracking_metric(MapMetric(name="b"))  # Add unimplemented metric.
        self.assertEqual(len(exp.fetch_trials_data(trial_indices=[0]).df), 30)
        # Try fetching attached data.
        exp.attach_data(batch_0_data)
        exp.attach_data(batch_1_data)
        self.assertEqual(exp.fetch_trials_data(trial_indices=[0]), batch_0_data)
        self.assertEqual(exp.fetch_trials_data(trial_indices=[1]), batch_1_data)
        self.assertEqual(set(batch_0_data.df["trial_index"].values), {0})
        self.assertEqual(
            set(batch_0_data.df["arm_name"].values), {a.name for a in batch_0.arms}
        )

    def testFetchTrialsDataIncremental(self):
        exp = self._setupBraninExperiment(n=5, incremental=True)

        first_data = exp.fetch_trials_data(trial_indices=[0])
        self.assertEqual(set(first_data.df["timestamp"].values), {0})

        more_data = exp.fetch_trials_data(trial_indices=[0])
        self.assertEqual(set(more_data.df["timestamp"].values), {1})

        # Since we're using BraninIncrementalTimestampMetric,
        # which has combine_with_last_data = True,
        # the cached data should be merged and contain both timestamps
        self.assertEqual(len(exp.data_by_trial[0]), 1)
        looked_up_data = exp.lookup_data()
        self.assertEqual(set(looked_up_data.df["timestamp"].values), {0, 1})

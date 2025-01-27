#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from collections import OrderedDict
from enum import unique
from unittest.mock import MagicMock, patch

import pandas as pd
from ax.core import BatchTrial, Experiment, Trial
from ax.core.arm import Arm
from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.exceptions.core import AxError, RunnerNotFoundError, UnsupportedError
from ax.metrics.branin import BraninMetric
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.load import load_experiment
from ax.storage.sqa_store.save import save_experiment
from ax.utils.common.constants import EXPERIMENT_IS_TEST_WARNING, Keys
from ax.utils.common.random import set_rng_seed
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_arm,
    get_branin_arms,
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_branin_experiment_with_timestamp_map_metric,
    get_branin_optimization_config,
    get_branin_search_space,
    get_data,
    get_experiment,
    get_experiment_with_data,
    get_experiment_with_map_data_type,
    get_experiment_with_observations,
    get_optimization_config,
    get_scalarized_outcome_constraint,
    get_search_space,
    get_sobol,
    get_status_quo,
    get_test_map_data_experiment,
)
from ax.utils.testing.mock import mock_botorch_optimize
from pyre_extensions import assert_is_instance

DUMMY_RUN_METADATA_KEY = "test_run_metadata_key"
DUMMY_RUN_METADATA_VALUE = "test_run_metadata_value"
DUMMY_RUN_METADATA: dict[str, str] = {DUMMY_RUN_METADATA_KEY: DUMMY_RUN_METADATA_VALUE}
DUMMY_ABANDONED_REASON = "test abandoned reason"
DUMMY_ARM_NAME = "test_arm_name"


class TestMetric(Metric):
    """Shell metric class for testing."""

    __test__ = False

    pass


class SyntheticRunnerWithMetadataKeys(SyntheticRunner):
    @property
    def run_metadata_report_keys(self) -> list[str]:
        return [DUMMY_RUN_METADATA_KEY]


class ExperimentTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
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

    def test_ExperimentInit(self) -> None:
        self.assertEqual(self.experiment.name, "test")
        self.assertEqual(self.experiment.description, "test description")
        self.assertEqual(self.experiment.name, "test")
        self.assertIsNotNone(self.experiment.time_created)
        self.assertEqual(self.experiment.experiment_type, None)
        self.assertEqual(self.experiment.num_abandoned_arms, 0)

    def test_ExperimentName(self) -> None:
        self.assertTrue(self.experiment.has_name)
        self.experiment.name = None
        self.assertFalse(self.experiment.has_name)
        with self.assertRaises(ValueError):
            self.experiment.name
        self.experiment.name = "test"

    def test_ExperimentType(self) -> None:
        self.experiment.experiment_type = "test"
        self.assertEqual(self.experiment.experiment_type, "test")

    def test_Eq(self) -> None:
        self.assertEqual(self.experiment, self.experiment)

        experiment2 = Experiment(
            name="test2",
            search_space=get_search_space(),
            optimization_config=get_optimization_config(),
            status_quo=get_arm(),
            description="test description",
        )
        self.assertNotEqual(self.experiment, experiment2)

    def test_DBId(self) -> None:
        self.assertIsNone(self.experiment.db_id)
        some_id = 123456789
        self.experiment.db_id = some_id
        self.assertEqual(self.experiment.db_id, some_id)

    def test_TrackingMetricsMerge(self) -> None:
        # Tracking and optimization metrics should get merged
        # m1 is on optimization_config while m3 is not
        exp = Experiment(
            name="test2",
            search_space=get_search_space(),
            optimization_config=get_optimization_config(),
            tracking_metrics=[Metric(name="m1"), Metric(name="m3")],
        )
        # pyre-fixme[16]: Optional type has no attribute `metrics`.
        self.assertEqual(len(exp.optimization_config.metrics) + 1, len(exp.metrics))

    def test_BasicBatchCreation(self) -> None:
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

    def test_Repr(self) -> None:
        self.assertEqual("Experiment(test)", str(self.experiment))

    def test_BasicProperties(self) -> None:
        self.assertEqual(self.experiment.status_quo, get_status_quo())
        self.assertEqual(self.experiment.search_space, get_search_space())
        self.assertEqual(self.experiment.optimization_config, get_optimization_config())
        self.assertEqual(self.experiment.is_test, True)

    def test_OnlyRangeParameterConstraints(self) -> None:
        self.assertEqual(0, 0)
        self.assertTrue(True)

        ax_client = AxClient()

        # Create an experiment with valid parameter constraints
        ax_client.create_experiment(
            name="experiment",
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                },
                {
                    "name": "x2",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                },
            ],
            objectives={"objective": ObjectiveProperties(minimize=False)},
            parameter_constraints=["x1 + x2 <= 1"],
        )

        # Try (and fail) to create an experiment with constraints on choice
        # paramaters
        with self.assertRaises(ValueError):
            ax_client.create_experiment(
                name="experiment",
                parameters=[
                    {
                        "name": "x1",
                        "type": "choice",
                        "values": [0.0, 1.0],
                    },
                    {
                        "name": "x2",
                        "type": "range",
                        "bounds": [0.0, 1.0],
                    },
                ],
                objectives={"objective": ObjectiveProperties(minimize=False)},
                parameter_constraints=["x1 + x2 <= 1"],
            )

        # Try (and fail) to create an experiment with constraints on fixed
        # parameters
        with self.assertRaises(ValueError):
            ax_client.create_experiment(
                name="experiment",
                parameters=[
                    {"name": "x1", "type": "fixed", "value": 0.0},
                    {
                        "name": "x2",
                        "type": "range",
                        "bounds": [0.0, 1.0],
                    },
                ],
                objectives={"objective": ObjectiveProperties(minimize=False)},
                parameter_constraints=["x1 + x2 <= 1"],
            )

    def test_MetricSetters(self) -> None:
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

    def test_SearchSpaceSetter(self) -> None:
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

    def test_StatusQuoSetter(self) -> None:
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
        with patch("ax.core.experiment.logger.warning") as mock_logger:
            sq_parameters["w"] = 3.6
            self.experiment.status_quo = Arm(sq_parameters)
        mock_logger.assert_called_once()
        self.assertIn("status_quo is updated", mock_logger.call_args.args[0])
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
        # pyre-fixme[16]: Optional type has no attribute `name`.
        self.assertEqual(exp.status_quo.name, "0_0")

        # Try setting sq to existing arm with different name
        with self.assertRaises(ValueError):
            exp.status_quo = Arm(arms[0].parameters, name="new_name")

    def test_RegisterArm(self) -> None:
        # Create a new arm, register on experiment
        parameters = self.experiment.status_quo.parameters
        parameters["w"] = 3.5
        arm = Arm(name="my_arm_name", parameters=parameters)
        self.experiment._register_arm(arm)
        self.assertEqual(self.experiment.arms_by_name[arm.name], arm)
        self.assertEqual(self.experiment.arms_by_signature[arm.signature], arm)

    def test_FetchAndStoreData(self) -> None:
        n = 10
        exp = self._setupBraninExperiment(n)
        batch = exp.trials[0]
        batch.mark_completed()
        self.assertEqual(exp.completed_trials, [batch])

        # Test fetch data
        batch_data = batch.fetch_data()
        self.assertEqual(len(batch_data.df), n)

        exp_data = exp.fetch_data()
        res = exp.fetch_data_results(metrics=[exp.metrics["b"]])
        res_one_metric = {k: v["b"] for k, v in res.items()}
        exp_data2 = Metric._unwrap_experiment_data(results=res_one_metric)
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
        self.assertEqual(len(full_dict[0]), 6)  # 6 data objs for batch 0

        # Test retrieving original batch 0 data
        self.assertEqual(len(exp.lookup_data_for_ts(t1).df), n)
        self.assertEqual(len(exp.lookup_data_for_trial(0)[0].df), n)

        # Test retrieving full exp data
        self.assertEqual(len(exp.lookup_data_for_ts(t2).df), 4 * n)

        self.assertEqual(len(full_dict[0]), 6)  # 6 data objs for batch 0
        new_data = Data(
            df=pd.DataFrame.from_records(
                [
                    {
                        "arm_name": "0_0",
                        # but now it is
                        "metric_name": "not_yet_on_experiment",
                        "mean": 3,
                        "sem": 0,
                        "trial_index": 0,
                    },
                    {
                        "arm_name": "0_0",
                        "metric_name": "z",
                        "mean": 3,
                        "sem": 0,
                        "trial_index": 0,
                    },
                ]
            )
        )
        t3 = exp.attach_data(new_data, combine_with_last_data=True)
        # still 6 data objs, since we combined last one
        self.assertEqual(len(full_dict[0]), 6)
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

    def test_OverwriteExistingData(self) -> None:
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

        with self.assertRaisesRegex(
            ValueError, "new data contains only a subset of the metrics"
        ):
            # prevent users from overwriting metrics
            data.df["metric_name"] = "a"
            exp.attach_data(data, overwrite_existing_data=True)

    def test_EmptyMetrics(self) -> None:
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

    def test_NumArmsNoDeduplication(self) -> None:
        exp = Experiment(name="test_experiment", search_space=get_search_space())
        arm = get_arm()
        exp.new_batch_trial().add_arm(arm)
        trial = exp.new_batch_trial().add_arm(arm)
        self.assertEqual(exp.sum_trial_sizes, 2)
        self.assertEqual(len(exp.arms_by_name), 1)
        trial.mark_arm_abandoned(trial.arms[0].name)
        self.assertEqual(exp.num_abandoned_arms, 1)

    def test_ExperimentWithoutName(self) -> None:
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

    def test_ExperimentRunner(self) -> None:
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
        # pyre-fixme[6]: For 1st param expected `Optional[str]` but got `Dict[str,
        #  bool]`.
        new_runner = SyntheticRunner(dummy_metadata=identifier)

        self.experiment.reset_runners(new_runner)
        # Don't update trials that have been run.
        self.assertEqual(batch.runner, original_runner)
        # Update default runner
        self.assertEqual(self.experiment.runner, new_runner)
        # Update candidate trial runners.
        self.assertEqual(self.experiment.trials[1].runner, new_runner)

    def test_FetchTrialsData(self) -> None:
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

    def test_immutable_search_space_and_opt_config(self) -> None:
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

    def test_AttachBatchTrialNoArmNames(self) -> None:
        num_trials = len(self.experiment.trials)

        attached_parameterizations, trial_index = self.experiment.attach_trial(
            parameterizations=[
                {"w": 5.3, "x": 5, "y": "baz", "z": True},
                {"w": 5.2, "x": 5, "y": "foo", "z": True},
                {"w": 5.1, "x": 5, "y": "bar", "z": True},
            ],
            ttl_seconds=3600,
            run_metadata={"test_metadata_field": 1},
            optimize_for_power=True,
        )

        self.assertEqual(len(self.experiment.trials), num_trials + 1)
        self.assertEqual(
            len(set(self.experiment.trials[trial_index].arms_by_name) - {"status_quo"}),
            3,
        )
        self.assertEqual(type(self.experiment.trials[trial_index]), BatchTrial)

    def test_AttachBatchTrialWithArmNames(self) -> None:
        num_trials = len(self.experiment.trials)

        attached_parameterizations, trial_index = self.experiment.attach_trial(
            parameterizations=[
                {"w": 5.3, "x": 5, "y": "baz", "z": True},
                {"w": 5.2, "x": 5, "y": "foo", "z": True},
                {"w": 5.1, "x": 5, "y": "bar", "z": True},
            ],
            arm_names=["arm1", "arm2", "arm3"],
            ttl_seconds=3600,
            run_metadata={"test_metadata_field": 1},
            optimize_for_power=True,
        )

        self.assertEqual(len(self.experiment.trials), num_trials + 1)
        self.assertEqual(
            len(set(self.experiment.trials[trial_index].arms_by_name) - {"status_quo"}),
            3,
        )
        self.assertEqual(type(self.experiment.trials[trial_index]), BatchTrial)
        self.assertEqual(
            {"arm1", "arm2", "arm3"},
            set(self.experiment.trials[trial_index].arms_by_name) - {"status_quo"},
        )

    def test_AttachSingleArmTrialNoArmName(self) -> None:
        num_trials = len(self.experiment.trials)

        attached_parameterization, trial_index = self.experiment.attach_trial(
            parameterizations=[{"w": 5.3, "x": 5, "y": "baz", "z": True}],
            ttl_seconds=3600,
            run_metadata={"test_metadata_field": 1},
            optimize_for_power=True,
        )

        self.assertEqual(len(self.experiment.trials), num_trials + 1)
        self.assertEqual(type(self.experiment.trials[trial_index]), Trial)

    def test_AttachSingleArmTrialWithArmName(self) -> None:
        num_trials = len(self.experiment.trials)

        attached_parameterization, trial_index = self.experiment.attach_trial(
            parameterizations=[{"w": 5.3, "x": 5, "y": "baz", "z": True}],
            arm_names=["arm1"],
            ttl_seconds=3600,
            run_metadata={"test_metadata_field": 1},
            optimize_for_power=True,
        )

        self.assertEqual(len(self.experiment.trials), num_trials + 1)
        self.assertEqual(type(self.experiment.trials[trial_index]), Trial)
        self.assertEqual(
            "arm1",
            self.experiment.trials[trial_index].arm.name,
        )

    def test_fetch_as_class(self) -> None:
        class MyMetric(Metric):
            @property
            def fetch_multi_group_by_metric(self) -> type[Metric]:
                return Metric

        m = MyMetric(name="test_metric")
        exp = Experiment(
            name="test",
            search_space=get_branin_search_space(),
            tracking_metrics=[m],
            runner=SyntheticRunner(),
        )
        self.assertEqual(exp._metrics_by_class(), {Metric: [m]})

    @patch(  # pyre-ignore[56]: Cannot infer function type
        # No-op mock just to record calls to `bulk_fetch_experiment_data`.
        f"{BraninMetric.__module__}.BraninMetric.bulk_fetch_experiment_data",
        side_effect=BraninMetric(
            name="branin", param_names=["x1", "x2"]
        ).bulk_fetch_experiment_data,
    )
    def test_prefer_lookup_where_possible(
        self, mock_bulk_fetch_experiment_data: MagicMock
    ) -> None:
        # By default, `BraninMetric` is available while trial is running.
        exp = self._setupBraninExperiment(n=5)
        exp.fetch_data()
        # Since metric is available while trial is running, we should be
        # refetching the data and no data should be attached to experiment.
        mock_bulk_fetch_experiment_data.assert_called_once()
        self.assertEqual(len(exp._data_by_trial), 2)

        with patch(
            f"{BraninMetric.__module__}.BraninMetric.is_available_while_running",
            return_value=False,
        ):
            exp = self._setupBraninExperiment(n=5)
            exp.fetch_data()
            # 1. No completed trials => no fetch case.
            mock_bulk_fetch_experiment_data.reset_mock()
            dat = exp.fetch_data()
            mock_bulk_fetch_experiment_data.assert_not_called()
            # Data should be empty since there are no completed trials.
            self.assertTrue(dat.df.empty)

            # 2. Newly completed trials => fetch case.
            mock_bulk_fetch_experiment_data.reset_mock()
            # pyre-fixme[16]: Optional type has no attribute `mark_completed`.
            exp.trials.get(0).mark_completed()
            exp.trials.get(1).mark_completed()
            dat = exp.fetch_data()
            # `bulk_fetch_experiment_data` should be called N=number of trials times.
            self.assertEqual(len(mock_bulk_fetch_experiment_data.call_args_list), 2)
            # Data should no longer be empty since there are completed trials.
            self.assertFalse(dat.df.empty)
            # Data for two trials should get attached.
            self.assertEqual(len(exp._data_by_trial), 2)

            # 3. Previously fetched => look up in cache case.
            mock_bulk_fetch_experiment_data.reset_mock()
            # All fetched data should get cached, so no fetch should happen next time.
            exp.fetch_data()
            mock_bulk_fetch_experiment_data.assert_not_called()
            # No new data should be attached to the experiment
            self.assertEqual(len(exp._data_by_trial), 2)

    def test_WarmStartFromOldExperiment(self) -> None:
        # create old_experiment
        len_old_trials = 7
        i_failed_trial = 1
        i_abandoned_trial = 3
        i_running_trial = 5
        old_experiment = get_branin_experiment()
        old_experiment.runner = SyntheticRunnerWithMetadataKeys()
        for i_old_trial in range(len_old_trials):
            sobol_run = get_sobol(search_space=old_experiment.search_space).gen(n=1)
            trial = old_experiment.new_trial(generator_run=sobol_run)
            trial.mark_running(no_runner_required=True)
            if i_old_trial == i_failed_trial:
                trial.mark_failed()
            elif i_old_trial == i_abandoned_trial:
                trial.mark_abandoned(reason=DUMMY_ABANDONED_REASON)
            elif i_old_trial == i_running_trial:
                pass
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
        # name one arm to test name-preserving logic.
        old_experiment.trials[0].arm._name = DUMMY_ARM_NAME
        new_experiment.warm_start_from_old_experiment(
            old_experiment=old_experiment,
        )
        self.assertEqual(len(new_experiment.trials), len(old_experiment.trials) - 1)
        i_old_trial = 0
        for idx, trial in new_experiment.trials.items():
            # skip failed trial
            i_old_trial += i_old_trial == i_failed_trial
            # pyre-fixme[16]: `BaseTrial` has no attribute `arm`.
            old_arm = old_experiment.trials[i_old_trial].arm
            self.assertEqual(
                trial.arm.parameters,
                old_arm.parameters,
            )
            self.assertRegex(
                trial._properties["source"], "Warm start.*Experiment.*trial"
            )
            self.assertDictEqual(trial.run_metadata, DUMMY_RUN_METADATA)
            i_old_trial += 1

            # Check naming logic.
            if idx == 0:
                self.assertEqual(trial.arm.name, DUMMY_ARM_NAME)
            else:
                self.assertEqual(
                    trial.arm.name, f"{old_arm.name}_{old_experiment.name}"
                )

        # Check that the data was attached for correct trials
        old_df = old_experiment.fetch_data().df
        new_df = new_experiment.fetch_data().df

        self.assertEqual(len(new_df), len_old_trials - 2)
        pd.testing.assert_frame_equal(
            old_df.drop(["arm_name", "trial_index"], axis=1),
            new_df.drop(["arm_name", "trial_index"], axis=1),
        )

        # check that all non-failed/abandoned trials are copied to new_experiment
        new_experiment = get_branin_experiment()
        # make metric noiseless for exact reproducibility
        new_experiment.optimization_config.objective.metric.noise_sd = 0
        new_experiment.warm_start_from_old_experiment(
            old_experiment=old_experiment,
            copy_run_metadata_keys=[DUMMY_RUN_METADATA_KEY],
            trial_statuses_to_copy=[TrialStatus.COMPLETED],
        )
        self.assertEqual(len(new_experiment.trials), len(old_experiment.trials) - 3)

        # Warm start from an experiment with only a subset of metrics
        map_data_experiment = get_branin_experiment_with_timestamp_map_metric()
        map_data_experiment.warm_start_from_old_experiment(
            old_experiment=old_experiment
        )
        self.assertEqual(
            len(map_data_experiment.trials), len(old_experiment.trials) - 1
        )

    def test_is_test_warning(self) -> None:
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

    def test_clone_with(self) -> None:
        init_test_engine_and_session_factory(force_init=True)
        experiment = get_branin_experiment(
            with_batch=True,
            with_completed_trial=True,
            with_status_quo=True,
            with_choice_parameter=True,
            num_batch_trial=3,
            with_completed_batch=True,
        )
        # Save the experiment to set db_ids.
        save_experiment(experiment)

        larger_search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x1",
                    parameter_type=ParameterType.FLOAT,
                    lower=-10.0,
                    upper=10.0,
                ),
                ChoiceParameter(
                    name="x2",
                    parameter_type=ParameterType.FLOAT,
                    values=[float(x) for x in range(0, 16)],
                ),
            ],
        )

        new_status_quo = Arm({"x1": 1.0, "x2": 1.0})

        cloned_experiment = experiment.clone_with(
            search_space=larger_search_space,
            status_quo=new_status_quo,
        )
        self.assertEqual(cloned_experiment._data_by_trial, experiment._data_by_trial)
        self.assertEqual(len(cloned_experiment.trials), 4)
        for trial_index in cloned_experiment.trials.keys():
            cloned_trial = cloned_experiment.trials[trial_index]
            original_trial = experiment.trials[trial_index]
            self.assertEqual(cloned_trial.status, original_trial.status)
        x1 = assert_is_instance(
            cloned_experiment.search_space.parameters["x1"], RangeParameter
        )
        self.assertEqual(x1.lower, -10.0)
        self.assertEqual(x1.upper, 10.0)
        x2 = assert_is_instance(
            cloned_experiment.search_space.parameters["x2"], ChoiceParameter
        )
        self.assertEqual(len(x2.values), 16)
        self.assertEqual(
            assert_is_instance(cloned_experiment.status_quo, Arm).parameters,
            {"x1": 1.0, "x2": 1.0},
        )
        # make sure the sq of the original experiment is unchanged
        self.assertEqual(
            assert_is_instance(experiment.status_quo, Arm).parameters,
            {"x1": 0.0, "x2": 0.0},
        )
        self.assertEqual(len(cloned_experiment.trials[0].arms), 16)

        self.assertEqual(
            cloned_experiment.lookup_data_for_trial(1)[0].df["trial_index"].iloc[0], 1
        )

        # make sure updating cloned experiment doesn't change the original experiment
        cloned_experiment.status_quo = Arm({"x1": -1.0, "x2": 1.0})
        self.assertEqual(
            assert_is_instance(cloned_experiment.status_quo, Arm).parameters,
            {"x1": -1.0, "x2": 1.0},
        )
        self.assertEqual(
            assert_is_instance(experiment.status_quo, Arm).parameters,
            {"x1": 0.0, "x2": 0.0},
        )

        # Save the cloned experiment to db and make sure the original
        # experiment is unchanged in the db.
        save_experiment(cloned_experiment)
        reloaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, reloaded_experiment)

        # Clone specific trials.
        # With existing data.
        experiment._data_by_trial
        cloned_experiment = experiment.clone_with(trial_indices=[1])
        self.assertEqual(len(cloned_experiment.trials), 1)
        cloned_df = cloned_experiment.lookup_data_for_trial(0)[0].df
        self.assertEqual(cloned_df["trial_index"].iloc[0], 0)

        # With new data.
        df = pd.DataFrame(
            {
                "arm_name": ["1_0"],
                "metric_name": ["branin"],
                "mean": [100.0],
                "sem": [1.0],
                "trial_index": [1],
            },
        )
        cloned_experiment = experiment.clone_with(trial_indices=[1], data=Data(df=df))
        self.assertEqual(len(cloned_experiment.trials), 1)
        self.assertEqual(len(experiment.trials), 4)
        cloned_df = cloned_experiment.lookup_data_for_trial(1)[0].df
        self.assertEqual(cloned_df.shape[0], 1)
        self.assertEqual(cloned_df["mean"].iloc[0], 100.0)
        self.assertEqual(cloned_df["sem"].iloc[0], 1.0)
        # make sure the original experiment data is unchanged
        df = experiment.lookup_data_for_trial(1)[0].df
        self.assertEqual(df["sem"].iloc[0], 0.1)

        # Clone with MapData.
        experiment = get_test_map_data_experiment(
            num_trials=5, num_fetches=3, num_complete=4
        )
        cloned_experiment = experiment.clone_with(
            search_space=larger_search_space,
            status_quo=new_status_quo,
        )
        new_data = cloned_experiment.lookup_data()
        self.assertNotEqual(cloned_experiment._data_by_trial, experiment._data_by_trial)
        self.assertIsInstance(new_data, MapData)
        expected_data_by_trial = {}
        for trial_index in experiment.trials:
            if original_trial_data := experiment._data_by_trial.get(trial_index, None):
                expected_data_by_trial[trial_index] = OrderedDict(
                    list(original_trial_data.items())[-1:]
                )
        self.assertEqual(cloned_experiment.data_by_trial, expected_data_by_trial)

        experiment = get_experiment()
        cloned_experiment = experiment.clone_with()
        self.assertEqual(cloned_experiment.name, "cloned_experiment_" + experiment.name)
        cloned_experiment._name = experiment.name

        # the clone_experiment._time_created field is set as datetime.now().
        # for it to be equal we need to update it to match experiment.
        cloned_experiment._time_created = experiment._time_created
        self.assertEqual(cloned_experiment, experiment)

        # test clear_trial_type
        experiment = get_branin_experiment(
            with_batch=True,
            num_batch_trial=1,
            with_completed_batch=True,
        )
        experiment.trials[0]._trial_type = "foo"
        with self.assertRaisesRegex(
            ValueError, "Experiment does not support trial_type foo."
        ):
            experiment.clone_with()
        cloned_experiment = experiment.clone_with(clear_trial_type=True)
        self.assertIsNone(cloned_experiment.trials[0].trial_type)

    def test_metric_summary_df(self) -> None:
        experiment = Experiment(
            name="test_experiment",
            search_space=SearchSpace(parameters=[]),
            optimization_config=MultiObjectiveOptimizationConfig(
                objective=MultiObjective(
                    objectives=[
                        Objective(
                            metric=Metric(name="my_objective_1", lower_is_better=True),
                            minimize=True,
                        ),
                        Objective(
                            metric=TestMetric(name="my_objective_2"), minimize=False
                        ),
                    ]
                ),
                objective_thresholds=[
                    ObjectiveThreshold(
                        metric=TestMetric(name="my_objective_2"),
                        bound=5.1,
                        relative=False,
                        op=ComparisonOp.GEQ,
                    )
                ],
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=Metric(name="my_constraint_1", lower_is_better=False),
                        bound=1,
                        relative=True,
                        op=ComparisonOp.GEQ,
                    ),
                    OutcomeConstraint(
                        metric=TestMetric(name="my_constraint_2"),
                        bound=-7.8,
                        relative=False,
                        op=ComparisonOp.LEQ,
                    ),
                ],
            ),
            tracking_metrics=[
                Metric(name="my_tracking_metric_1", lower_is_better=True),
                TestMetric(name="my_tracking_metric_2", lower_is_better=False),
                Metric(name="my_tracking_metric_3"),
            ],
        )
        df = experiment.metric_config_summary_df
        expected_df = pd.DataFrame(
            data={
                "Name": [
                    "my_objective_1",
                    "my_objective_2",
                    "my_constraint_1",
                    "my_constraint_2",
                    "my_tracking_metric_1",
                    "my_tracking_metric_2",
                    "my_tracking_metric_3",
                ],
                "Type": [
                    "Metric",
                    "TestMetric",
                    "Metric",
                    "TestMetric",
                    "Metric",
                    "TestMetric",
                    "Metric",
                ],
                "Goal": [
                    "minimize",
                    "maximize",
                    "constrain",
                    "constrain",
                    "track",
                    "track",
                    "track",
                ],
                "Bound": ["None", ">= 5.1", ">= 1%", "<= -7.8", "None", "None", "None"],
                "Lower is Better": [True, "None", False, "None", True, False, "None"],
            }
        )
        expected_df["Goal"] = pd.Categorical(
            df["Goal"],
            categories=["minimize", "maximize", "constrain", "track", "None"],
            ordered=True,
        )
        pd.testing.assert_frame_equal(df, expected_df)

    def test_arms_by_signature_for_deduplication(self) -> None:
        experiment = self.experiment
        trial = experiment.new_trial()
        arm = Arm({"w": 1, "x": 2, "y": "foo", "z": True})
        trial.add_arm(arm)
        expected_with_failed = {
            experiment.status_quo.signature: experiment.status_quo,
        }
        expected_with_other = {
            experiment.status_quo.signature: experiment.status_quo,
            arm.signature: arm,
        }
        for status in TrialStatus:
            trial._status = status
            if status == TrialStatus.FAILED:
                self.assertEqual(
                    experiment.arms_by_signature_for_deduplication, expected_with_failed
                )
            else:
                self.assertEqual(
                    experiment.arms_by_signature_for_deduplication, expected_with_other
                )

    def test_trial_indices(self) -> None:
        experiment = self.experiment
        for _ in range(6):
            experiment.new_trial()
        self.assertEqual(experiment.trial_indices_expecting_data, set())
        experiment.trials[0].mark_staged()
        experiment.trials[1].mark_running(no_runner_required=True)
        experiment.trials[2].mark_running(no_runner_required=True).mark_completed()
        self.assertEqual(experiment.trial_indices_expecting_data, {1, 2})
        experiment.trials[1].mark_abandoned()
        self.assertEqual(experiment.trial_indices_expecting_data, {2})
        experiment.trials[4].mark_running(no_runner_required=True)
        self.assertEqual(experiment.trial_indices_expecting_data, {2, 4})
        experiment.trials[4].mark_failed()
        self.assertEqual(experiment.trial_indices_expecting_data, {2})
        experiment.trials[5].mark_running(no_runner_required=True).mark_early_stopped()
        self.assertEqual(experiment.trial_indices_expecting_data, {2, 5})

    def test_stop_trial(self) -> None:
        self.experiment.new_trial()
        with patch.object(self.experiment, "runner"), patch.object(
            self.experiment.runner, "stop", return_value=None
        ) as mock_runner_stop, patch.object(
            BaseTrial, "mark_early_stopped"
        ) as mock_mark_stopped:
            self.experiment.stop_trial_runs(trials=[self.experiment.trials[0]])
            mock_runner_stop.assert_called_once()
            mock_mark_stopped.assert_called_once()

    def test_stop_trial_without_runner(self) -> None:
        self.experiment.new_trial()
        with self.assertRaisesRegex(
            RunnerNotFoundError,
            "Unable to stop trial runs: Runner not configured for experiment or trial.",
        ):
            self.experiment.stop_trial_runs(trials=[self.experiment.trials[0]])

    def test_to_df(self) -> None:
        experiment = get_experiment_with_observations(
            observations=[[1.0, 2.0], [3.0, 4.0]]
        )
        df = experiment.to_df()
        xs = [
            experiment.trials[0].arms[0].parameters["x"],
            experiment.trials[1].arms[0].parameters["x"],
        ]
        ys = [
            experiment.trials[0].arms[0].parameters["y"],
            experiment.trials[1].arms[0].parameters["y"],
        ]
        expected_df = pd.DataFrame.from_dict(
            {
                "trial_index": [0, 1],
                "arm_name": ["0_0", "1_0"],
                "trial_status": ["COMPLETED", "COMPLETED"],
                "generation_method": ["Sobol", "Sobol"],
                "name": ["0", "1"],  # the metadata
                "m1": [1.0, 3.0],
                "m2": [2.0, 4.0],
                "x": xs,
                "y": ys,
            }
        )
        self.assertTrue(df.equals(expected_df))
        # Check that empty columns are included when omit=False.
        df = experiment.to_df(omit_empty_columns=False)
        self.assertEqual(
            df.columns.tolist(),
            [
                "trial_index",
                "arm_name",
                "trial_status",
                "fail_reason",
                "generation_method",
                "generation_node",
                "name",
                "m1",
                "m2",
                "x",
                "y",
            ],
        )


class ExperimentWithMapDataTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_experiment_with_map_data_type()

    def _setupBraninExperiment(self, n: int, incremental: bool = False) -> Experiment:
        exp = get_branin_experiment_with_timestamp_map_metric()
        batch = exp.new_batch_trial()
        batch.add_arms_and_weights(arms=get_branin_arms(n=n, seed=0))
        batch.run()

        batch_2 = exp.new_batch_trial()
        batch_2.add_arms_and_weights(arms=get_branin_arms(n=3 * n, seed=1))
        batch_2.run()
        return exp

    def test_FetchDataWithMapData(self) -> None:
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
            # pyre-fixme[6]: For 1st param expected `Dict[str, List[Tuple[Dict[str, H...
            evaluations={
                arm_name: partial_results[0:1]
                for arm_name, partial_results in evaluations.items()
            },
            trial_index=0,
        )
        self.experiment.attach_data(first_epoch)
        remaining_epochs = MapData.from_map_evaluations(
            # pyre-fixme[6]: For 1st param expected `Dict[str, List[Tuple[Dict[str, H...
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

    def test_FetchDataWithMixedData(self) -> None:
        with patch(
            f"{BraninMetric.__module__}.BraninMetric.is_available_while_running",
            return_value=False,
        ):
            exp = self._setupBraninExperiment(n=5)
            [exp.trials[i].mark_completed() for i in range(len(exp.trials))]

            # Fill cache with MapData
            map_data = exp.fetch_data(metrics=[exp.metrics["branin_map"]])

            # Fetch other metrics and merge Data into the cached MapData
            full_data = exp.fetch_data()

            self.assertEqual(len(full_data.true_df), len(map_data.true_df) + 20)

    def test_FetchTrialsData(self) -> None:
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
            exp.fetch_trials_data(trial_indices=[0, 1]).df.shape[0],
            len(exp.arms_by_name) * 2,
        )

        with self.assertRaisesRegex(ValueError, ".* not associated .*"):
            exp.fetch_trials_data(trial_indices=[2])
        # Try to fetch data when there are only metrics and no attached data.
        exp.remove_tracking_metric(metric_name="branin")  # Remove implemented metric.
        exp.add_tracking_metric(
            BraninMetric(name="branin", param_names=["x1", "x2"])
        )  # Add unimplemented metric.
        # pyre-fixme[16]: `Data` has no attribute `map_df`.
        self.assertEqual(len(exp.fetch_trials_data(trial_indices=[0]).map_df), 10)
        # Try fetching attached data.
        exp.attach_data(batch_0_data)
        exp.attach_data(batch_1_data)
        pd.testing.assert_frame_equal(
            exp.fetch_trials_data(trial_indices=[0]).df, batch_0_data.df
        )
        pd.testing.assert_frame_equal(
            exp.fetch_trials_data(trial_indices=[1]).df, batch_1_data.df
        )
        self.assertEqual(set(batch_0_data.df["trial_index"].values), {0})
        self.assertEqual(
            set(batch_0_data.df["arm_name"].values), {a.name for a in batch_0.arms}
        )

    def test_is_moo_problem(self) -> None:
        exp = get_branin_experiment()
        self.assertFalse(exp.is_moo_problem)
        exp = get_branin_experiment_with_multi_objective()
        self.assertTrue(exp.is_moo_problem)
        exp._optimization_config = None
        self.assertFalse(exp.is_moo_problem)

    def test_WarmStartMapData(self) -> None:
        # create old_experiment
        len_old_trials = 7
        i_failed_trial = 1
        i_abandoned_trial = 3
        i_running_trial = 5
        old_experiment = get_branin_experiment_with_timestamp_map_metric()
        for i_old_trial in range(len_old_trials):
            sobol_run = get_sobol(search_space=old_experiment.search_space).gen(n=1)
            trial = old_experiment.new_trial(generator_run=sobol_run)
            trial.mark_running(no_runner_required=True)
            if i_old_trial == i_failed_trial:
                trial.mark_failed()
            elif i_old_trial == i_abandoned_trial:
                trial.mark_abandoned(reason=DUMMY_ABANDONED_REASON)
            elif i_old_trial == i_running_trial:
                pass
            else:
                trial.mark_completed()
        # make metric noiseless for exact reproducibility
        old_experiment.optimization_config.objective.metric.noise_sd = 0
        old_experiment.fetch_data()

        # check that all non-failed trials are copied to new_experiment
        new_experiment = get_branin_experiment_with_timestamp_map_metric()
        # make metric noiseless for exact reproducibility
        new_experiment.optimization_config.objective.metric.noise_sd = 0
        for _, trial in old_experiment.trials.items():
            trial._run_metadata = DUMMY_RUN_METADATA
        new_experiment.warm_start_from_old_experiment(
            old_experiment=old_experiment,
            copy_run_metadata_keys=[DUMMY_RUN_METADATA_KEY],
        )
        self.assertEqual(len(new_experiment.trials), len(old_experiment.trials) - 1)
        i_old_trial = 0
        for _, trial in new_experiment.trials.items():
            # skip failed trial
            i_old_trial += i_old_trial == i_failed_trial
            self.assertEqual(
                # pyre-fixme[16]: `BaseTrial` has no attribute `arm`.
                trial.arm.parameters,
                old_experiment.trials[i_old_trial].arm.parameters,
            )
            self.assertRegex(
                trial._properties["source"], "Warm start.*Experiment.*trial"
            )
            self.assertEqual(
                trial._properties["generation_model_key"], Models.SOBOL.value
            )
            self.assertDictEqual(trial.run_metadata, DUMMY_RUN_METADATA)
            i_old_trial += 1

        # Check that the data was attached for correct trials

        # Old experiment has already been fetched, and re-fetching will add readings to
        # still-running map metrics.
        old_df = old_experiment.lookup_data().df
        new_df = new_experiment.fetch_data().df

        old_df = old_df.sort_values(by=["arm_name", "metric_name"], ignore_index=True)
        new_df = new_df.sort_values(by=["arm_name", "metric_name"], ignore_index=True)

        # Factor 2 comes from 2 rows per trial in this test experiment
        self.assertEqual(len(new_df), (len_old_trials - 2) * 2)
        pd.testing.assert_frame_equal(
            old_df.drop(["arm_name", "trial_index"], axis=1),
            new_df.drop(["arm_name", "trial_index"], axis=1),
        )

    @mock_botorch_optimize
    def test_batch_with_multiple_generator_runs(self) -> None:
        exp = get_branin_experiment()
        # set seed to avoid transient errors caused by duplicate arms,
        # which leads to fewer arms in the trial than expected.
        seed = 0
        sobol = Models.SOBOL(experiment=exp, search_space=exp.search_space, seed=seed)
        exp.new_batch_trial(generator_runs=[sobol.gen(n=7)]).run().complete()

        data = exp.fetch_data()
        set_rng_seed(seed)
        gp = Models.BOTORCH_MODULAR(
            experiment=exp, search_space=exp.search_space, data=data
        )
        ts = Models.EMPIRICAL_BAYES_THOMPSON(
            experiment=exp, search_space=exp.search_space, data=data
        )
        exp.new_batch_trial(generator_runs=[gp.gen(n=3), ts.gen(n=1)]).run().complete()

        self.assertEqual(len(exp.trials), 2)
        self.assertEqual(len(exp.trials[0].generator_runs), 1)
        self.assertEqual(len(exp.trials[0].arms), 7)
        self.assertEqual(len(exp.trials[1].generator_runs), 2)
        self.assertEqual(len(exp.trials[1].arms), 4)

    def test_it_does_not_take_both_single_and_multiple_gr_ars(self) -> None:
        exp = get_branin_experiment()
        sobol = Models.SOBOL(experiment=exp, search_space=exp.search_space)
        gr1 = sobol.gen(n=7)
        gr2 = sobol.gen(n=7)
        with self.assertRaisesRegex(
            UnsupportedError,
            "Cannot specify both `generator_run` and `generator_runs`.",
        ):
            exp.new_batch_trial(
                generator_run=gr1,
                generator_runs=[gr2],
            )

    def test_experiment_with_aux_experiments(self) -> None:
        @unique
        class TestAuxiliaryExperimentPurpose(AuxiliaryExperimentPurpose):
            MyAuxExpPurpose = "my_auxiliary_experiment_purpose"
            MyOtherAuxExpPurpose = "my_other_auxiliary_experiment_purpose"

        for get_exp_func in [get_experiment, get_experiment_with_data]:
            exp = get_exp_func()
            data = exp.lookup_data()

            aux_exp = AuxiliaryExperiment(experiment=exp)
            another_aux_exp = AuxiliaryExperiment(experiment=exp, data=data)

            # init experiment with auxiliary experiments
            exp_w_aux_exp = Experiment(
                name="test",
                search_space=get_search_space(),
                auxiliary_experiments_by_purpose={
                    TestAuxiliaryExperimentPurpose.MyAuxExpPurpose: [aux_exp],
                },
            )

            # in-place modification of auxiliary experiments
            exp_w_aux_exp.auxiliary_experiments_by_purpose[
                TestAuxiliaryExperimentPurpose.MyOtherAuxExpPurpose
            ] = [aux_exp]
            self.assertEqual(
                exp_w_aux_exp.auxiliary_experiments_by_purpose,
                {
                    TestAuxiliaryExperimentPurpose.MyAuxExpPurpose: [aux_exp],
                    TestAuxiliaryExperimentPurpose.MyOtherAuxExpPurpose: [aux_exp],
                },
            )

            # test setter
            exp_w_aux_exp.auxiliary_experiments_by_purpose = {
                TestAuxiliaryExperimentPurpose.MyAuxExpPurpose: [aux_exp],
                TestAuxiliaryExperimentPurpose.MyOtherAuxExpPurpose: [
                    aux_exp,
                    another_aux_exp,
                ],
            }
            self.assertEqual(
                exp_w_aux_exp.auxiliary_experiments_by_purpose,
                {
                    TestAuxiliaryExperimentPurpose.MyAuxExpPurpose: [aux_exp],
                    TestAuxiliaryExperimentPurpose.MyOtherAuxExpPurpose: [
                        aux_exp,
                        another_aux_exp,
                    ],
                },
            )

    def test_name_and_store_arm_if_not_exists_same_name_different_signature(
        self,
    ) -> None:
        experiment = self.experiment
        shared_name = "shared_name"

        arm_1 = Arm({"x1": -1.0, "x2": 1.0}, name=shared_name)
        arm_2 = Arm({"x1": -1.7, "x2": 0.2, "x3": 1})
        self.assertNotEqual(arm_1.signature, arm_2.signature)

        experiment._register_arm(arm=arm_1)
        with self.assertRaisesRegex(
            AxError,
            f"Arm with name {shared_name} already exists on experiment "
            f"with different signature.",
        ):
            experiment._name_and_store_arm_if_not_exists(
                arm=arm_2, proposed_name=shared_name
            )

    def test_name_and_store_arm_if_not_exists_same_proposed_name_different_signature(
        self,
    ) -> None:
        experiment = self.experiment
        shared_name = "shared_name"

        arm_1 = Arm({"x1": -1.0, "x2": 1.0}, name=shared_name)
        arm_2 = Arm({"x1": -1.7, "x2": 0.2, "x3": 1}, name=shared_name)
        self.assertNotEqual(arm_1.signature, arm_2.signature)

        experiment._register_arm(arm=arm_1)
        with self.assertRaisesRegex(
            AxError,
            f"Arm with name {shared_name} already exists on experiment "
            f"with different signature.",
        ):
            experiment._name_and_store_arm_if_not_exists(
                arm=arm_2, proposed_name="different proposed name"
            )

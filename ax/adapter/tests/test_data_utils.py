#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import mock

import numpy as np
from ax.adapter.data_utils import DataLoaderConfig, extract_experiment_data
from ax.adapter.registry import Generators
from ax.core.data import Data
from ax.core.trial_status import NON_ABANDONED_STATUSES, TrialStatus
from ax.exceptions.core import UnsupportedError
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_timestamp_map_metric,
    get_experiment_with_observations,
)
from pandas import DataFrame, MultiIndex
from pandas.testing import assert_frame_equal


class TestDataUtils(TestCase):
    def test_data_loader_config(self) -> None:
        # Defaults
        config = DataLoaderConfig()
        self.assertFalse(config.fit_out_of_design)
        self.assertFalse(config.fit_abandoned)
        self.assertTrue(config.fit_only_completed_map_metrics)
        self.assertEqual(config.latest_rows_per_group, 1)
        self.assertIsNone(config.limit_rows_per_group)
        self.assertIsNone(config.limit_rows_per_metric)
        self.assertEqual(config.statuses_to_fit, NON_ABANDONED_STATUSES)
        self.assertEqual(config.statuses_to_fit_map_metric, {TrialStatus.COMPLETED})
        # Validation for latest / limit rows.
        with self.assertRaisesRegex(UnsupportedError, "must be None if either of"):
            DataLoaderConfig(latest_rows_per_group=1, limit_rows_per_metric=5)
        # With a bunch of modifications.
        config = DataLoaderConfig(
            fit_out_of_design=True,
            fit_abandoned=True,
            fit_only_completed_map_metrics=False,
            latest_rows_per_group=None,
            limit_rows_per_metric=10,
            limit_rows_per_group=20,
        )
        self.assertTrue(config.fit_out_of_design)
        self.assertTrue(config.fit_abandoned)
        self.assertFalse(config.fit_only_completed_map_metrics)
        self.assertIsNone(config.latest_rows_per_group)
        self.assertEqual(config.limit_rows_per_metric, 10)
        self.assertEqual(config.limit_rows_per_group, 20)
        self.assertEqual(config.statuses_to_fit, set(TrialStatus))
        self.assertEqual(config.statuses_to_fit_map_metric, set(TrialStatus))

    def test_extract_experiment_data_empty(self) -> None:
        # Tests extraction of experiment data from experiments with no data.
        empty_exp = get_branin_experiment()
        for empty_exp in [
            get_branin_experiment(),
            get_branin_experiment_with_timestamp_map_metric(),
        ]:
            experiment_data = extract_experiment_data(
                experiment=empty_exp, data_loader_config=DataLoaderConfig()
            )
            for df in (experiment_data.arm_data, experiment_data.observation_data):
                self.assertEqual(len(df), 0)
                self.assertEqual(df.index.names, ["trial_index", "arm_name"])
            self.assertEqual(
                experiment_data.arm_data.columns.to_list(),
                list(empty_exp.parameters) + ["metadata"],
            )
            self.assertEqual(experiment_data, experiment_data)

    def test_extract_experiment_data_non_map(self) -> None:
        # This is a 2 objective experiment with 2 trials, 1 arm each.
        observations = [[0.1, 1.0], [0.2, 2.0]]
        exp = get_experiment_with_observations(observations=observations)
        # Add another trial but fail it.
        sobol = Generators.SOBOL(experiment=exp)
        exp.new_trial(generator_run=sobol.gen(1)).run().mark_failed()
        # Add an abandoned trial but include data for one metric.
        # Also add some custom arm metadata.
        gr = sobol.gen(1)
        gr._candidate_metadata_by_arm_signature = {
            gr.arms[0].signature: {"test": "test_metadata"}
        }
        t = exp.new_trial(generator_run=gr).mark_abandoned()
        data = Data(
            df=DataFrame.from_records(
                [
                    {
                        "arm_name": t.arms[0].name,
                        "metric_name": "m1",
                        "mean": 0.4,
                        "sem": 0.2,
                        "trial_index": t.index,
                    }
                ]
            )
        )
        exp.attach_data(data)

        # Test with default config.
        experiment_data = extract_experiment_data(
            experiment=exp, data_loader_config=DataLoaderConfig()
        )
        # Arm data: All trials with data should be included.
        # Filtering happens when constructing datasets.
        arms_with_data = ["0_0", "1_0", "3_0"]
        expected_arm_df = DataFrame(
            [exp.arms_by_name[arm_name].parameters for arm_name in arms_with_data],
            index=MultiIndex.from_tuples(
                [(0, "0_0"), (1, "1_0"), (3, "3_0")],
                names=["trial_index", "arm_name"],
            ),
        )
        assert_frame_equal(
            experiment_data.arm_data.drop("metadata", axis=1), expected_arm_df
        )
        # Check metadata. It only includes info about trial completion etc.
        metadata = experiment_data.arm_data["metadata"].tolist()
        self.assertEqual(
            metadata,
            [
                {Keys.TRIAL_COMPLETION_TIMESTAMP: mock.ANY},
                {Keys.TRIAL_COMPLETION_TIMESTAMP: mock.ANY},
                {Keys.TRIAL_COMPLETION_TIMESTAMP: mock.ANY, "test": "test_metadata"},
            ],
        )
        # Observation data: Only completed trials should be included.
        # First 4 rows correspond to 2 metrics from the 2 completed trials.
        data_df = exp.lookup_data().df[:4]
        expected_obs_df = data_df.pivot(
            columns="metric_name",
            index=["trial_index", "arm_name"],
            values=["mean", "sem"],
        )
        assert_frame_equal(experiment_data.observation_data, expected_obs_df)
        self.assertTrue(
            experiment_data.observation_data.index.equals(
                MultiIndex.from_tuples(
                    [(0, "0_0"), (1, "1_0")],
                    names=["trial_index", "arm_name"],
                )
            )
        )
        self.assertTrue(
            experiment_data.observation_data.columns.equals(
                MultiIndex.from_tuples(
                    [("mean", "m1"), ("mean", "m2"), ("sem", "m1"), ("sem", "m2")],
                    names=[None, "metric_name"],
                )
            )
        )
        # The successful trials don't have SEMs.
        self.assertTrue(experiment_data.observation_data["sem"].isnull().all().all())
        # Check the metric values.
        self.assertEqual(
            experiment_data.observation_data["mean"].to_numpy().tolist(), observations
        )
        # Check equality with self.
        self.assertEqual(experiment_data, experiment_data)

        # Test with config that includes abandoned trials.
        experiment_data = extract_experiment_data(
            experiment=exp, data_loader_config=DataLoaderConfig(fit_abandoned=True)
        )
        assert_frame_equal(
            experiment_data.arm_data.drop("metadata", axis=1), expected_arm_df
        )
        # All data should be included.
        data_df = exp.lookup_data().df
        expected_obs_df = data_df.pivot(
            columns="metric_name",
            index=["trial_index", "arm_name"],
            values=["mean", "sem"],
        )
        assert_frame_equal(experiment_data.observation_data, expected_obs_df)
        self.assertEqual(len(experiment_data.observation_data), 3)
        # Check the metric & sem values.
        NAN = float("nan")
        self.assertTrue(
            np.array_equal(
                experiment_data.observation_data["mean"].to_numpy(),
                np.asarray(observations + [[0.4, NAN]]),
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.array_equal(
                experiment_data.observation_data["sem"].to_numpy(),
                np.asarray([[NAN, NAN], [NAN, NAN], [0.2, NAN]]),
                equal_nan=True,
            )
        )

    def test_extract_experiment_data_map(self) -> None:
        exp = get_branin_experiment_with_timestamp_map_metric(with_trials_and_data=True)
        t_0_metric = 55.602112642270264
        t_1_metric = 27.702905548512433
        # Test with default config.
        experiment_data = extract_experiment_data(
            experiment=exp, data_loader_config=DataLoaderConfig()
        )
        # Arm data: First two trials should be included, since they have data.
        expected_arm_df = DataFrame(
            [{"x1": 0.0, "x2": 0.0}, {"x1": 1.0, "x2": 1.0}],
            index=MultiIndex.from_tuples(
                [(0, "0_0"), (1, "1_0")], names=["trial_index", "arm_name"]
            ),
        )
        assert_frame_equal(
            experiment_data.arm_data.drop("metadata", axis=1), expected_arm_df
        )
        # Observation data: By default only includes completed map metrics.
        # There is none, so map metrics are not included.
        metrics = set(experiment_data.observation_data["mean"])
        self.assertEqual(metrics, {"branin"})
        self.assertEqual(len(experiment_data.observation_data), 2)
        # Complete a trial to include map metrics.
        exp.trials[0].complete()
        experiment_data = extract_experiment_data(
            experiment=exp, data_loader_config=DataLoaderConfig()
        )
        # Arm data is not changed.
        assert_frame_equal(
            experiment_data.arm_data.drop("metadata", axis=1), expected_arm_df
        )
        # Observation data: Map metrics should be included but only with latest
        # timestamp for trial 0.
        metrics = set(experiment_data.observation_data["mean"])
        self.assertEqual(metrics, {"branin", "branin_map"})
        index = MultiIndex.from_tuples(
            [(0, "0_0", 0.0), (0, "0_0", 3.0), (1, "1_0", 0.0)],
            names=["trial_index", "arm_name", "timestamp"],
        )
        expected_mean_df = DataFrame(
            [
                {"branin": t_0_metric, "branin_map": None},
                {"branin": None, "branin_map": t_0_metric},
                {"branin": t_1_metric, "branin_map": None},
            ],
            index=index,
        )
        self.assertTrue(
            experiment_data.observation_data["mean"].equals(expected_mean_df)
        )
        expected_sem_df = DataFrame(
            [
                {"branin": 0.0, "branin_map": None},
                {"branin": None, "branin_map": 0.0},
                {"branin": 0.0, "branin_map": None},
            ],
            index=index,
        )
        self.assertTrue(experiment_data.observation_data["sem"].equals(expected_sem_df))

        # Test with config that includes all map data.
        experiment_data = extract_experiment_data(
            experiment=exp,
            data_loader_config=DataLoaderConfig(
                fit_only_completed_map_metrics=False,
                latest_rows_per_group=None,
            ),
        )
        # Arm data is not changed.
        assert_frame_equal(
            experiment_data.arm_data.drop("metadata", axis=1), expected_arm_df
        )
        # Observation data: Map metrics should be included for all timestamps.
        metrics = set(experiment_data.observation_data["mean"])
        self.assertEqual(metrics, {"branin", "branin_map"})
        index = MultiIndex.from_tuples(
            [
                (0, "0_0", 0.0),
                (0, "0_0", 1.0),
                (0, "0_0", 2.0),
                (0, "0_0", 3.0),
                (1, "1_0", 0.0),
                (1, "1_0", 1.0),
            ],
            names=["trial_index", "arm_name", "timestamp"],
        )
        expected_mean_df = DataFrame(
            [
                {"branin": t_0_metric, "branin_map": t_0_metric},
                {"branin": None, "branin_map": t_0_metric},
                {"branin": None, "branin_map": t_0_metric},
                {"branin": None, "branin_map": t_0_metric},
                {"branin": t_1_metric, "branin_map": t_1_metric},
                {"branin": None, "branin_map": t_1_metric},
            ],
            index=index,
        )
        self.assertTrue(
            experiment_data.observation_data["mean"].equals(expected_mean_df)
        )
        # Check equality with self.
        self.assertEqual(experiment_data, experiment_data)

    def test_filter_by_arm_name(self) -> None:
        # This is a 2 objective experiment with 5 trials, 1 arm each.
        observations = [[0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0], [0.5, 5.0]]
        exp = get_experiment_with_observations(observations=observations)
        experiment_data = extract_experiment_data(
            experiment=exp, data_loader_config=DataLoaderConfig()
        )
        self.assertEqual(len(experiment_data.arm_data), 5)
        self.assertEqual(len(experiment_data.observation_data), 5)
        # Filter to only include arms 1 & 3.
        arm_names = ["1_0", "3_0"]
        filtered = experiment_data.filter_by_arm_names(arm_names=arm_names)
        self.assertEqual(
            list(filtered.arm_data.index.get_level_values("arm_name")), arm_names
        )
        self.assertEqual(
            list(filtered.observation_data.index.get_level_values("arm_name")),
            arm_names,
        )
        # Check that filtering was applied correctly.
        mask = [False, True, False, True, False]
        assert_frame_equal(filtered.arm_data, experiment_data.arm_data.loc[mask])
        assert_frame_equal(
            filtered.observation_data, experiment_data.observation_data.loc[mask]
        )

    def test_filter_latest_observations(self) -> None:
        exp = get_branin_experiment_with_timestamp_map_metric(with_trials_and_data=True)
        experiment_data = extract_experiment_data(
            experiment=exp,
            data_loader_config=DataLoaderConfig(
                fit_only_completed_map_metrics=False,
                latest_rows_per_group=None,
            ),
        )
        filtered_data = experiment_data.filter_latest_observations()
        self.assertNotEqual(filtered_data, experiment_data)
        # Arm data is the same.
        assert_frame_equal(experiment_data.arm_data, filtered_data.arm_data)
        # Observation data is filtered to only include one row for each arm
        # and no timestamp on the index.
        # In this case, the data is identical to timeframe 0 for both arms.
        expected_obs_data = experiment_data.observation_data.loc[
            [(0, "0_0", 0.0), (1, "1_0", 0.0)]
        ].droplevel(2)
        assert_frame_equal(filtered_data.observation_data, expected_obs_data)

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import namedtuple
from logging import WARN
from typing import Dict, List

from unittest.mock import patch

import pandas as pd

from ax.analysis.utils.analysis_utils import (
    compute_maximum_map_values,
    exp_to_df,
    Experiment,
    FEASIBLE_COL_NAME,
)
from ax.core.arm import Arm
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_experiment_with_observations,
    get_multi_type_experiment,
    get_test_map_data_experiment,
)

OBJECTIVE_NAME = "branin"
PARAMETER_COLUMNS = ["x1", "x2"]
FLOAT_COLUMNS: List[str] = [OBJECTIVE_NAME] + PARAMETER_COLUMNS
EXPECTED_COLUMNS: List[str] = [
    "trial_index",
    "arm_name",
    "trial_status",
    "generation_method",
] + FLOAT_COLUMNS
DUMMY_OBJECTIVE_MEAN = 1.2345
DUMMY_SOURCE = "test_source"


DUMMY_MSG = "test_message"

ANALYSIS_UTIL_PATH = "ax.analysis.utils.analysis_utils"


class AnalysisUtilsTest(TestCase):
    @patch(
        f"{ANALYSIS_UTIL_PATH}._merge_results_if_no_duplicates",
        autospec=True,
        return_value=pd.DataFrame(
            [
                # Trial indexes are out-of-order.
                {"arm_name": "a", "trial_index": 1},
                {"arm_name": "b", "trial_index": 2},
                {"arm_name": "c", "trial_index": 0},
            ]
        ),
    )
    def test_exp_to_df_row_ordering(self, _) -> None:
        """
        This test verifies that the returned data frame indexes are
        in the same order as trial index. It mocks _merge_results_if_no_duplicates
        to verify just the ordering of items in the final data frame.
        """
        exp = get_branin_experiment(with_trial=True)
        df = exp_to_df(exp)
        # Check that all 3 rows are in order
        self.assertEqual(len(df), 3)
        for idx, row in df.iterrows():
            self.assertEqual(row["trial_index"], idx)

    @patch(
        f"{ANALYSIS_UTIL_PATH}._merge_results_if_no_duplicates",
        autospec=True,
        return_value=pd.DataFrame(
            [
                # Trial indexes are out-of-order.
                {
                    "col1": 1,
                    "arm_name": "a",
                    "trial_status": "FAILED",
                    "generation_method": "Manual",
                    "trial_index": 1,
                },
                {
                    "col1": 2,
                    "arm_name": "b",
                    "trial_status": "COMPLETED",
                    "generation_method": "BO",
                    "trial_index": 2,
                },
                {
                    "col1": 3,
                    "arm_name": "c",
                    "trial_status": "COMPLETED",
                    "generation_method": "Manual",
                    "trial_index": 0,
                },
            ]
        ),
    )
    def test_exp_to_df_col_ordering(self, _) -> None:
        """
        This test verifies that the returned data frame indexes are
        in the same order as trial index. It mocks _merge_results_if_no_duplicates
        to verify just the ordering of items in the final data frame.
        """
        exp = get_branin_experiment(with_trial=True)
        df = exp_to_df(exp)
        self.assertListEqual(
            list(df.columns),
            ["trial_index", "arm_name", "trial_status", "generation_method", "col1"],
        )

    def test_exp_to_df_max_map_value(self) -> None:
        exp = get_test_map_data_experiment(num_trials=3, num_fetches=5, num_complete=0)

        def compute_maximum_map_values_timestamp(
            experiment: Experiment,
        ) -> Dict[int, float]:
            return compute_maximum_map_values(
                experiment=experiment, map_key="timestamp"
            )

        df = exp_to_df(
            exp=exp,
            additional_fields_callables={  # pyre-ignore
                "timestamp": compute_maximum_map_values_timestamp
            },
        )
        self.assertEqual(df["timestamp"].tolist(), [5.0, 5.0, 5.0])

    def test_exp_to_df_trial_timing(self) -> None:
        # 1. test all have started, none have completed
        exp = get_test_map_data_experiment(num_trials=3, num_fetches=5, num_complete=0)
        df = exp_to_df(
            exp=exp,
            trial_attribute_fields=["time_run_started", "time_completed"],
            always_include_field_columns=True,
        )
        self.assertTrue("time_run_started" in list(df.columns))
        self.assertTrue("time_completed" in list(df.columns))
        # since all trials started, all should have values
        self.assertFalse(any(df["time_run_started"].isnull()))
        # since no trials are complete, all should be None
        self.assertTrue(all(df["time_completed"].isnull()))

        # 2. test some trials not started yet
        exp.trials[0]._time_run_started = None
        df = exp_to_df(
            exp=exp, trial_attribute_fields=["time_run_started", "time_completed"]
        )
        # the first trial should have NaN for rel_time_run_started
        self.assertTrue(df["time_run_started"].isnull().iloc[0])

        # 3. test all trials not started yet
        for t in exp.trials.values():
            t._time_run_started = None
        df = exp_to_df(
            exp=exp,
            trial_attribute_fields=["time_run_started", "time_completed"],
            always_include_field_columns=True,
        )
        self.assertTrue(all(df["time_run_started"].isnull()))

        # 4. test some trials are completed
        exp = get_test_map_data_experiment(num_trials=3, num_fetches=5, num_complete=2)
        df = exp_to_df(
            exp=exp, trial_attribute_fields=["time_run_started", "time_completed"]
        )
        # the last trial should have NaN for rel_time_completed
        self.assertTrue(df["time_completed"].isnull().iloc[2])

    def test_exp_to_df_with_failure(self) -> None:
        fail_reason = "test reason"

        # Set up experiment with a failed trial
        exp = get_branin_experiment(with_trial=True)
        exp.trials[0].run()
        exp.trials[0].mark_failed(reason=fail_reason)

        df = exp_to_df(exp)
        self.assertEqual(
            set(EXPECTED_COLUMNS + ["reason"]) - set(df.columns), {OBJECTIVE_NAME}
        )
        self.assertEqual(f"{fail_reason}...", df["reason"].iloc[0])

    def test_exp_to_df(self) -> None:
        # MultiTypeExperiment should fail
        exp = get_multi_type_experiment()
        with self.assertRaisesRegex(ValueError, "MultiTypeExperiment"):
            exp_to_df(exp=exp)

        # exp with no trials should return empty results
        exp = get_branin_experiment()
        df = exp_to_df(exp=exp)
        self.assertEqual(len(df), 0)

        # set up experiment
        exp = get_branin_experiment(with_batch=True)

        # check that pre-run experiment returns all columns except objective
        df = exp_to_df(exp)
        self.assertEqual(set(EXPECTED_COLUMNS) - set(df.columns), {OBJECTIVE_NAME})
        self.assertEqual(len(df.index), len(exp.arms_by_name))

        exp.trials[0].run()
        exp.fetch_data()

        # assert result is df with expected columns and length
        df = exp_to_df(exp=exp)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(sorted(df.columns), sorted(EXPECTED_COLUMNS))
        self.assertEqual(len(df.index), len(exp.arms_by_name))

        # test with run_metadata_fields and trial_properties_fields not empty
        # add source to properties
        for _, trial in exp.trials.items():
            trial._properties["source"] = DUMMY_SOURCE
        df = exp_to_df(
            exp, run_metadata_fields=["name"], trial_properties_fields=["source"]
        )
        self.assertIn("name", df.columns)
        self.assertIn("trial_properties_source", df.columns)

        # test column values or types
        self.assertTrue(all(x == 0 for x in df.trial_index))
        self.assertTrue(all(x == "RUNNING" for x in df.trial_status))
        self.assertTrue(all(x == "Sobol" for x in df.generation_method))
        self.assertTrue(all(x == DUMMY_SOURCE for x in df.trial_properties_source))
        self.assertTrue(all(x == "branin_test_experiment_0" for x in df.name))
        for float_column in FLOAT_COLUMNS:
            self.assertTrue(all(isinstance(x, float) for x in df[float_column]))

        # works correctly for failed trials (will need to mock)
        dummy_struct = namedtuple("dummy_struct", "df")
        mock_results = dummy_struct(
            df=pd.DataFrame(
                {
                    "arm_name": ["0_0", "1_0"],
                    "metric_name": [OBJECTIVE_NAME] * 2,
                    "mean": [DUMMY_OBJECTIVE_MEAN] * 2,
                    "sem": [0] * 2,
                    "trial_index": [0, 1],
                    "n": [123] * 2,
                    "frac_nonnull": [1] * 2,
                }
            )
        )
        mock_results.df.index = range(len(exp.trials) * 2, len(exp.trials) * 2 + 2)
        with patch.object(Experiment, "lookup_data", lambda self: mock_results):
            df = exp_to_df(exp=exp)
        # all but two rows should have a metric value of NaN
        self.assertEqual(pd.isna(df[OBJECTIVE_NAME]).sum(), len(df.index) - 2)

        # an experiment with more results than arms raises an error
        with patch.object(
            Experiment, "lookup_data", lambda self: mock_results
        ), self.assertRaisesRegex(ValueError, "inconsistent experimental state"):
            exp_to_df(exp=get_branin_experiment())

        # custom added trial has a generation_method of Manual
        custom_arm = Arm(name="custom", parameters={"x1": 0, "x2": 0})
        exp.new_trial().add_arm(custom_arm)
        df = exp_to_df(exp)
        self.assertEqual(
            df[df.arm_name == "custom"].iloc[0].generation_method, "Manual"
        )
        # failing feasibility calculation doesn't warns and suppresses error
        observations = [[1.0, 2.0, 3.0], [4.0, 5.0, -6.0], [7.0, 8.0, 9.0]]
        exp = get_experiment_with_observations(
            observations=observations,
            constrained=True,
        )
        with patch(
            f"{exp_to_df.__module__}._is_row_feasible", side_effect=KeyError(DUMMY_MSG)
        ), self.assertLogs(logger="ax", level=WARN) as log:
            exp_to_df(exp)
            self.assertIn(
                f"Feasibility calculation failed with error: '{DUMMY_MSG}'",
                log.output[0],
            )

        # infeasible arm has `is_feasible = False`.
        df = exp_to_df(exp)
        self.assertListEqual(list(df[FEASIBLE_COL_NAME]), [True, False, True])

        # all rows infeasible.
        observations = [[1.0, 2.0, -3.0], [4.0, 5.0, -6.0], [7.0, 8.0, -9.0]]
        exp = get_experiment_with_observations(
            observations=observations,
            constrained=True,
        )
        df = exp_to_df(exp)
        self.assertListEqual(list(df[FEASIBLE_COL_NAME]), [False, False, False])

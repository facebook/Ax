#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from ax.core.data import (
    REQUIRED_COLUMNS,
    Data,
    clone_without_metrics,
    custom_data_class,
    set_single_trial,
)
from ax.utils.common.testutils import TestCase
from ax.utils.common.timeutils import current_timestamp_in_millis


class DataTest(TestCase):
    def setUp(self):
        self.df_hash = "3dd7ab8c67942d43c78ea4af05bbb1c4"
        self.df = pd.DataFrame(
            [
                {
                    "arm_name": "0_0",
                    "mean": 2.0,
                    "sem": 0.2,
                    "trial_index": 1,
                    "metric_name": "a",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_0",
                    "mean": 1.8,
                    "sem": 0.3,
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_1",
                    "mean": 4.0,
                    "sem": 0.6,
                    "trial_index": 1,
                    "metric_name": "a",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_1",
                    "mean": 3.7,
                    "sem": 0.5,
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_2",
                    "mean": 0.5,
                    "sem": None,
                    "trial_index": 1,
                    "metric_name": "a",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_2",
                    "mean": 3.0,
                    "sem": None,
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
            ]
        )

    def testData(self):
        self.assertEqual(Data(), Data())
        data = Data(df=self.df)
        self.assertEqual(data, data)
        self.assertEqual(data.df_hash, self.df_hash)

        df = data.df
        self.assertEqual(
            float(df[df["arm_name"] == "0_0"][df["metric_name"] == "a"]["mean"]), 2.0
        )
        self.assertEqual(
            float(df[df["arm_name"] == "0_1"][df["metric_name"] == "b"]["sem"]), 0.5
        )

    def testBadData(self):
        df = pd.DataFrame([{"bad_field": "0_0", "bad_field_2": {"x": 0, "y": "a"}}])
        with self.assertRaises(ValueError):
            Data(df=df)

    def testEmptyData(self):
        df = Data().df
        self.assertTrue(df.empty)
        self.assertTrue(set(df.columns == REQUIRED_COLUMNS))
        self.assertTrue(Data.from_multiple_data([]).df.empty)

    def testSetSingleBatch(self):
        data = Data(df=self.df)
        merged_data = set_single_trial(data)
        self.assertTrue((merged_data.df["trial_index"] == 0).all())

        data = Data(
            df=pd.DataFrame(
                [{"arm_name": "0_1", "mean": 3.7, "sem": 0.5, "metric_name": "b"}]
            )
        )
        merged_data = set_single_trial(data)
        self.assertTrue("trial_index" not in merged_data.df)

    def testCustomData(self):
        CustomData = custom_data_class(
            column_data_types={"metadata": str, "created_time": pd.Timestamp},
            required_columns={"metadata"},
        )
        data_entry = {
            "arm_name": "0_1",
            "mean": 3.7,
            "sem": 0.5,
            "metric_name": "b",
            "metadata": "42",
            "created_time": "2018-09-20",
        }
        data = CustomData(df=pd.DataFrame([data_entry]))
        self.assertNotEqual(data, Data(self.df))
        self.assertTrue(isinstance(data.df.iloc[0]["created_time"], pd.Timestamp))

        data_entry2 = {
            "arm_name": "0_1",
            "mean": 3.7,
            "sem": 0.5,
            "metric_name": "b",
            "created_time": "2018-09-20",
        }

        # Test without required column
        with self.assertRaises(ValueError):
            CustomData(df=pd.DataFrame([data_entry2]))

        # Try making regular data with extra column
        with self.assertRaises(ValueError):
            Data(df=pd.DataFrame([data_entry2]))

    def testFromEvaluations(self):
        data = Data.from_evaluations(
            evaluations={"0_1": {"b": (3.7, 0.5)}},
            trial_index=0,
            sample_sizes={"0_1": 2},
            start_time=current_timestamp_in_millis(),
            end_time=current_timestamp_in_millis(),
        )
        self.assertEqual(len(data.df), 1)
        self.assertNotEqual(data, Data(self.df))
        self.assertIn("start_time", data.df)
        self.assertIn("end_time", data.df)

    def testFromFidelityEvaluations(self):
        data = Data.from_fidelity_evaluations(
            evaluations={
                "0_1": [
                    ({"f1": 1.0, "f2": 0.5}, {"b": (3.7, 0.5)}),
                    ({"f1": 1.0, "f2": 0.75}, {"b": (3.8, 0.5)}),
                ]
            },
            trial_index=0,
            sample_sizes={"0_1": 2},
            start_time=current_timestamp_in_millis(),
            end_time=current_timestamp_in_millis(),
        )
        self.assertEqual(len(data.df), 2)
        self.assertIn("start_time", data.df)
        self.assertIn("end_time", data.df)

    def testCloneWithoutMetrics(self):
        data = Data(df=self.df)
        expected = Data(
            df=pd.DataFrame(
                [
                    {
                        "arm_name": "0_0",
                        "mean": 1.8,
                        "sem": 0.3,
                        "trial_index": 1,
                        "metric_name": "b",
                        "start_time": "2018-01-01",
                        "end_time": "2018-01-02",
                    },
                    {
                        "arm_name": "0_1",
                        "mean": 3.7,
                        "sem": 0.5,
                        "trial_index": 1,
                        "metric_name": "b",
                        "start_time": "2018-01-01",
                        "end_time": "2018-01-02",
                    },
                    {
                        "arm_name": "0_2",
                        "mean": 3.0,
                        "sem": None,
                        "trial_index": 1,
                        "metric_name": "b",
                        "start_time": "2018-01-01",
                        "end_time": "2018-01-02",
                    },
                ]
            )
        )
        self.assertEqual(clone_without_metrics(data, {"a"}), expected)

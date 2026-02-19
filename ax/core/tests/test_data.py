#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

import numpy as np
import pandas as pd
from ax.core.data import (
    _filter_df,
    _subsample_rate,
    combine_dfs_favoring_recent,
    Data,
    MAP_KEY,
)
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.utils.common.testutils import TestCase

REPR_500: str = (
    "Data(df=\n"
    "|    |   trial_index |   arm_name | metric_name   | metric_signature   |   mean "
    "|   sem | start_time          | end_time            |\n"
    "|---:|--------------:|-----------:|:--------------|:-------------------|-------:"
    "|------:|:--------------------|:--------------------|\n"
    "|  0 |             1 |        0_0 | a             | a_signature        |    2   "
    "|   0.2 | 2018-01-01 00:00:00 | 2018-01-02 00:00:00 |\n"
    "|  1 |             1 |        0_0 | b             | b_signature        |    1.8 "
    "|   0.3 | 2018-01-...)"
)

REPR_1000: str = (
    "Data(df=\n"
    "|    |   trial_index |   arm_name | metric_name   | metric_signature   |   mean "
    "|   sem | start_time          | end_time            |\n"
    "|---:|--------------:|-----------:|:--------------|:-------------------|-------:"
    "|------:|:--------------------|:--------------------|\n"
    "|  0 |             1 |        0_0 | a             | a_signature        |    2   "
    "|   0.2 | 2018-01-01 00:00:00 | 2018-01-02 00:00:00 |\n"
    "|  1 |             1 |        0_0 | b             | b_signature        |    1.8 "
    "|   0.3 | 2018-01-01 00:00:00 | 2018-01-02 00:00:00 |\n"
    "|  2 |             1 |        0_1 | a             | a_signature        |    4   "
    "|   0.6 | 2018-01-01 00:00:00 | 2018-01-02 00:00:00 |\n"
    "|  3 |             1 |        0_1 | b             | b_signature        |    3.7 "
    "|   0.5 | 2018-01-01 00:00:00 | 2018-01-02 00:00:00 |\n"
    "|  4 |             1 |        0_2 | a             | a_signature        |    0.5 "
    "| nan   | 2018-01-01 00:00:00 | 2018-01-02 00:00:00 |\n"
    "|  5 |             1 |        0_2 | b             | b_signatur...)"
)


def get_test_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "arm_name": "0_0",
                "mean": 2.0,
                "sem": 0.2,
                "trial_index": 1,
                "metric_name": "a",
                "start_time": "2018-01-01",
                "end_time": "2018-01-02",
                "metric_signature": "a_signature",
            },
            {
                "arm_name": "0_0",
                "mean": 1.8,
                "sem": 0.3,
                "trial_index": 1,
                "metric_name": "b",
                "start_time": "2018-01-01",
                "end_time": "2018-01-02",
                "metric_signature": "b_signature",
            },
            {
                "arm_name": "0_1",
                "mean": 4.0,
                "sem": 0.6,
                "trial_index": 1,
                "metric_name": "a",
                "start_time": "2018-01-01",
                "end_time": "2018-01-02",
                "metric_signature": "a_signature",
            },
            {
                "arm_name": "0_1",
                "mean": 3.7,
                "sem": 0.5,
                "trial_index": 1,
                "metric_name": "b",
                "start_time": "2018-01-01",
                "end_time": "2018-01-02",
                "metric_signature": "b_signature",
            },
            {
                "arm_name": "0_2",
                "mean": 0.5,
                "sem": None,
                "trial_index": 1,
                "metric_name": "a",
                "start_time": "2018-01-01",
                "end_time": "2018-01-02",
                "metric_signature": "a_signature",
            },
            {
                "arm_name": "0_2",
                "mean": 3.0,
                "sem": None,
                "trial_index": 1,
                "metric_name": "b",
                "start_time": "2018-01-01",
                "end_time": "2018-01-02",
                "metric_signature": "b_signature",
            },
        ]
    )


class DataTest(TestCase):
    """Tests for Data without a "step" column."""

    def setUp(self) -> None:
        super().setUp()
        self.data_without_df = Data()
        self.df = get_test_dataframe()
        self.data_with_df = Data(df=self.df)
        self.metric_name_to_signature = {"a": "a_signature", "b": "b_signature"}

    def test_init(self) -> None:
        # Test equality
        self.assertEqual(self.data_without_df, self.data_without_df)
        self.assertEqual(self.data_with_df, self.data_with_df)

        # Test accessing values
        df = self.data_with_df.df
        self.assertEqual(
            float(df[df["arm_name"] == "0_0"][df["metric_name"] == "a"]["mean"].item()),
            2.0,
        )
        self.assertEqual(
            float(df[df["arm_name"] == "0_1"][df["metric_name"] == "b"]["sem"].item()),
            0.5,
        )

        # Test has_step_column is False
        self.assertFalse(self.data_with_df.has_step_column)

        # Test empty initialization
        empty = Data()
        self.assertTrue(empty.empty)
        self.assertEqual(set(empty.full_df.columns), empty.REQUIRED_COLUMNS)
        self.assertFalse(empty.has_step_column)

    def test_clone(self) -> None:
        data = self.data_with_df
        data._db_id = 1234
        data_clone = data.clone()
        # Check equality of the objects.
        self.assertTrue(data.df.equals(data_clone.df))
        # Make sure it's not the original object or df.
        self.assertIsNot(data, data_clone)
        self.assertIsNot(data.df, data_clone.df)
        self.assertIsNone(data_clone._db_id)

    def test_BadData(self) -> None:
        data = {"bad_field": "0_0", "bad_field_2": {"x": 0, "y": "a"}}
        df = pd.DataFrame([data])
        with self.assertRaisesRegex(
            ValueError, "Dataframe must contain required columns"
        ):
            Data(df=df)

    def test_EmptyData(self) -> None:
        data = self.data_without_df
        df = data.df
        self.assertTrue(df.empty)
        self.assertTrue(Data.from_multiple_data([]).df.empty)

        expected_columns = Data.REQUIRED_COLUMNS
        self.assertEqual(set(df.columns), expected_columns)

    def test_from_multiple_with_generator(self) -> None:
        data = Data.from_multiple_data(self.data_with_df for _ in range(2))
        self.assertEqual(len(data.full_df), 2 * len(self.data_with_df.full_df))
        self.assertFalse(data.has_step_column)

    def test_get_df_with_cols_in_expected_order(self) -> None:
        with self.subTest("Wrong order"):
            df = pd.DataFrame(columns=["mean", "trial_index", "hat"], data=[[0] * 3])
            re_ordered = Data._get_df_with_cols_in_expected_order(df=df)
            self.assertEqual(
                re_ordered.columns.to_list(), ["trial_index", "mean", "hat"]
            )
            self.assertIsNot(df, re_ordered)

        with self.subTest("Correct order"):
            df = pd.DataFrame(
                columns=["trial_index", "mean", "hat"], data=[[0 for _ in range(3)]]
            )
            re_ordered = Data._get_df_with_cols_in_expected_order(df=df)
            self.assertEqual(
                re_ordered.columns.to_list(), ["trial_index", "mean", "hat"]
            )
            self.assertIs(df, re_ordered)

    def test_equality(self) -> None:
        self.assertEqual(self.data_with_df, self.data_with_df)
        self.assertEqual(
            self.data_with_df, type(self.data_with_df)(df=self.data_with_df.full_df)
        )

    def test_trial_indices(self) -> None:
        self.assertEqual(
            self.data_with_df.trial_indices,
            set(self.data_with_df.full_df["trial_index"].unique()),
        )

    def test_repr(self) -> None:
        self.assertEqual(
            str(Data(df=self.df)),
            REPR_1000,
        )
        with patch(f"{Data.__module__}.DF_REPR_MAX_LENGTH", 500):
            self.assertEqual(str(Data(df=self.df)), REPR_500)

    def test_from_multiple(self) -> None:
        with self.subTest("Combinining non-empty Data"):
            data = Data.from_multiple_data([Data(df=self.df), Data(df=self.df)])
            self.assertEqual(len(data.df), 2 * len(self.df))

        with self.subTest("Combining empty Data makes empty Data"):
            data = Data.from_multiple_data([Data(), Data()])
            self.assertEqual(data, Data())

    def test_filter(self) -> None:
        data = Data(df=self.df)
        # Test that filter throws when we provide metric names and metric signatures
        with self.assertRaisesRegex(
            UserInputError, "Cannot filter by both metric names and metric signatures."
        ):
            data.filter(metric_names=["a"], metric_signatures=["a_sig"])

        # Test that filter works when we provide metric names and trial indices
        filtered = data.filter(metric_names=["a"], trial_indices=[1])
        self.assertEqual(len(filtered.df), 3)
        self.assertEqual(set(filtered.df["metric_name"]), {"a"})
        self.assertEqual(set(filtered.df["trial_index"]), {1})
        self.assertIsInstance(filtered.df.index, pd.RangeIndex)

        # Test that filter works when we provide metric signatures and trial indices
        filtered = data.filter(metric_signatures=["a_signature"], trial_indices=[1])
        self.assertEqual(len(filtered.df), 3)
        self.assertEqual(set(filtered.df["metric_signature"]), {"a_signature"})
        self.assertEqual(set(filtered.df["trial_index"]), {1})

        # Test that filter works when we provide metric names
        filtered = data.filter(metric_signatures=["b_signature", "a_signature"])
        self.assertEqual(len(filtered.df), 6)
        self.assertEqual(
            set(filtered.df["metric_signature"]), {"a_signature", "b_signature"}
        )

        # Test that filter works when we provide metric signatures
        filtered = data.filter(metric_names=["a"])
        self.assertEqual(len(filtered.df), 3)
        self.assertEqual(set(filtered.df["metric_name"]), {"a"})

        with self.subTest("Metric names and metric signatures both specified"):
            with self.assertRaisesRegex(UserInputError, "Cannot filter by both"):
                _filter_df(
                    df=self.df, metric_names=["a"], metric_signatures=["a_signature"]
                )

        with self.subTest("No filtering"):
            self.assertIs(self.df, _filter_df(df=self.df))

    def test_safecast_df(self) -> None:
        # Create a df with unexpected index ([1])
        df = pd.DataFrame.from_records(
            [
                {
                    "index": 1,
                    "arm_name": "0_0",
                    "trial_index": 0.0,
                    "metric_name": "m",
                    "metric_signature": "m",
                    "mean": 0.0,
                    "sem": None,
                }
            ]
        ).set_index("index")
        self.assertEqual(df.index.get_level_values(0).tolist(), [1])

        safecast_df = Data._safecast_df(df=df)
        self.assertEqual(safecast_df.index.get_level_values(0).to_list(), [0])
        self.assertEqual(df["trial_index"].dtype, int)

    def test_subsample_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one of"):
            _subsample_rate(full_df=self.df)


class TestDataWithStep(TestCase):
    """Tests that are specific to data with a "step" column."""

    def setUp(self) -> None:
        super().setUp()
        self.df = pd.DataFrame(
            [
                {
                    "arm_name": "0_0",
                    MAP_KEY: 0,
                    "mean": 3.0,
                    "sem": 0.3,
                    "trial_index": 0,
                    "metric_name": "a",
                    "metric_signature": "a_sig",
                },
                # repeated arm 0_0
                {
                    "arm_name": "0_0",
                    MAP_KEY: 0,
                    "mean": 2.0,
                    "sem": 0.2,
                    "trial_index": 1,
                    "metric_name": "a",
                    "metric_signature": "a_sig",
                },
                {
                    "arm_name": "0_0",
                    MAP_KEY: 0,
                    "mean": 1.8,
                    "sem": 0.3,
                    "trial_index": 1,
                    "metric_name": "b",
                    "metric_signature": "b_sig",
                },
                {
                    "arm_name": "0_1",
                    MAP_KEY: 0,
                    "mean": 4.0,
                    "sem": 0.6,
                    "trial_index": 1,
                    "metric_name": "a",
                    "metric_signature": "a_sig",
                },
                {
                    "arm_name": "0_1",
                    MAP_KEY: 0,
                    "mean": 3.7,
                    "sem": 0.5,
                    "trial_index": 1,
                    "metric_name": "b",
                    "metric_signature": "b_sig",
                },
                {
                    "arm_name": "0_1",
                    MAP_KEY: 1,
                    "mean": 0.5,
                    "sem": None,
                    "trial_index": 1,
                    "metric_name": "a",
                    "metric_signature": "a_sig",
                },
                {
                    "arm_name": "0_1",
                    MAP_KEY: 1,
                    "mean": 3.0,
                    "sem": None,
                    "trial_index": 1,
                    "metric_name": "b",
                    "metric_signature": "b_sig",
                },
            ]
        )
        self.mmd = Data(df=self.df)

    def test_df(self) -> None:
        df = self.mmd.df
        self.assertEqual(set(df["trial_index"].drop_duplicates()), {0, 1})

        with self.subTest("Empty data"):
            df = Data(
                df=pd.DataFrame(
                    columns=[
                        "trial_index",
                        "arm_name",
                        MAP_KEY,
                        "metric_name",
                        "metric_signature",
                        "mean",
                        "sem",
                    ]
                )
            ).df
            self.assertTrue(df.empty)

    def test_combine(self) -> None:
        with self.subTest("From no MapDatas"):
            data = Data.from_multiple_data([])
            self.assertIsInstance(data, Data)
            self.assertEqual(data.full_df.size, 0)

        with self.subTest("From two MapDatas"):
            mmd_double = Data.from_multiple_data([self.mmd, self.mmd])
            self.assertIsInstance(mmd_double, Data)
            self.assertEqual(mmd_double.full_df.size, 2 * self.mmd.full_df.size)

        with self.subTest("From Datas"):
            data = Data(df=self.mmd.df)
            map_data = Data.from_multiple_data([data])
            self.assertIsInstance(map_data, Data)
            data = Data.from_multiple_data([data])
            self.assertEqual(len(data.full_df), len(map_data.full_df))

    def test_caching(self) -> None:
        with self.subTest("With step column"):
            fresh = Data(df=self.df)
            # Assert df is not cached before first call
            self.assertIsNone(fresh._memo_df)

            self.assertEqual(
                fresh.df.columns.size,
                fresh.full_df.columns.size,
            )

            # Assert df is cached after first call
            self.assertIsNotNone(fresh._memo_df)

            self.assertTrue(
                fresh.df.equals(
                    fresh.full_df.sort_values(MAP_KEY).drop_duplicates(
                        Data.DEDUPLICATE_BY_COLUMNS, keep="last"
                    )
                )
            )

        with self.subTest("No step column"):
            data = Data(df=self.df.drop(columns=["step"]))
            # Assert df is not cached before first call
            self.assertIsNone(data._memo_df)

            self.assertIs(data.df, data.full_df)

            # Nothing cached
            self.assertIsNone(data._memo_df)

    def test_latest(self) -> None:
        seed = 8888

        arm_names = ["0_0", "1_0", "2_0", "3_0"]
        max_epochs = [25, 50, 75, 100]
        metric_names_to_sig = {"a": "a", "b": "b"}
        large_map_df = pd.DataFrame(
            [
                {
                    "arm_name": arm_name,
                    MAP_KEY: epoch + 1,
                    "mean": epoch * 0.1,
                    "sem": 0.1,
                    "trial_index": trial_index,
                    "metric_name": metric_name,
                    "metric_signature": metric_sig,
                }
                for metric_name, metric_sig in metric_names_to_sig.items()
                for trial_index, (arm_name, max_epoch) in enumerate(
                    zip(arm_names, max_epochs)
                )
                for epoch in range(max_epoch)
            ]
        )
        large_map_data = Data(df=large_map_df)

        shuffled_large_map_df = large_map_data.full_df.groupby(
            Data.DEDUPLICATE_BY_COLUMNS
        ).sample(frac=1, random_state=seed)
        shuffled_large_map_data = Data(df=shuffled_large_map_df)

        for rows_per_group in [1, 40]:
            large_map_data_latest = large_map_data.latest(rows_per_group=rows_per_group)

            if rows_per_group == 1:
                self.assertTrue(
                    large_map_data_latest.full_df.groupby("metric_name")[MAP_KEY]
                    .transform(lambda col: set(col) == set(max_epochs))
                    .all()
                )

            # when rows_per_group is larger than the number of rows
            # actually observed in a group
            actual_rows_per_group = large_map_data_latest.full_df.groupby(
                Data.DEDUPLICATE_BY_COLUMNS
            ).size()
            expected_rows_per_group = np.minimum(
                large_map_data_latest.full_df.groupby(Data.DEDUPLICATE_BY_COLUMNS)[
                    MAP_KEY
                ]
                .max()
                .astype(int),
                rows_per_group,
            )
            self.assertTrue(actual_rows_per_group.equals(expected_rows_per_group))

            # behavior should be consistent despite shuffling
            shuffled_large_map_data_latest = shuffled_large_map_data.latest(
                rows_per_group=rows_per_group
            )
            self.assertTrue(
                shuffled_large_map_data_latest.full_df.equals(
                    large_map_data_latest.full_df
                )
            )

        with self.subTest("No step column"):
            data = Data(df=large_map_data.df.drop(columns=["step"]))
            latest = data.latest(rows_per_group=1)
            self.assertIs(latest, data)

            with self.assertRaisesRegex(
                UnsupportedError, "Cannot have rows_per_group greater than 1"
            ):
                data.latest(rows_per_group=2)

    def test_subsample(self) -> None:
        arm_names = ["0_0", "1_0", "2_0", "3_0"]
        max_epochs = [25, 50, 75, 100]
        metric_names_to_sig = {"a": "a", "b": "b"}
        large_map_df = pd.DataFrame(
            [
                {
                    "arm_name": arm_name,
                    MAP_KEY: epoch + 1,
                    "mean": epoch * 0.1,
                    "sem": 0.1,
                    "trial_index": trial_index,
                    "metric_name": metric_name,
                    "metric_signature": metric_sig,
                }
                for metric_name, metric_sig in metric_names_to_sig.items()
                for trial_index, (arm_name, max_epoch) in enumerate(
                    zip(arm_names, max_epochs)
                )
                for epoch in range(max_epoch)
            ]
        )
        large_map_data = Data(df=large_map_df)
        large_map_df_sparse_metric = pd.DataFrame(
            [
                {
                    "arm_name": arm_name,
                    MAP_KEY: epoch + 1,
                    "mean": epoch * 0.1,
                    "sem": 0.1,
                    "trial_index": trial_index,
                    "metric_name": metric_name,
                    "metric_signature": metric_sig,
                }
                for metric_name, metric_sig in metric_names_to_sig.items()
                for trial_index, (arm_name, max_epoch) in enumerate(
                    zip(arm_names, max_epochs)
                )
                for epoch in range(max_epoch if metric_name == "a" else max_epoch // 5)
            ]
        )
        large_map_data_sparse_metric = Data(df=large_map_df_sparse_metric)

        # test keep_every
        subsample = large_map_data.subsample(keep_every=10)
        self.assertEqual(len(subsample.full_df), 52)
        subsample = large_map_data.subsample(keep_every=25)
        self.assertEqual(len(subsample.full_df), 20)
        subsample = large_map_data.subsample(limit_rows_per_group=7)
        self.assertEqual(len(subsample.full_df), 36)

        # test limit_rows_per_group
        subsample = large_map_data.subsample(limit_rows_per_group=1)
        self.assertEqual(len(subsample.full_df), 8)
        subsample = large_map_data.subsample(limit_rows_per_group=7)
        self.assertEqual(len(subsample.full_df), 36)
        subsample = large_map_data.subsample(limit_rows_per_group=10)
        self.assertEqual(len(subsample.full_df), 52)
        subsample = large_map_data.subsample(limit_rows_per_group=1000)
        self.assertEqual(len(subsample.full_df), 500)

        # test limit_rows_per_metric
        subsample = large_map_data.subsample(limit_rows_per_metric=50)
        self.assertEqual(len(subsample.full_df), 100)
        subsample = large_map_data.subsample(limit_rows_per_metric=65)
        self.assertEqual(len(subsample.full_df), 128)
        subsample = large_map_data.subsample(limit_rows_per_metric=1000)
        self.assertEqual(len(subsample.full_df), 500)

        # test include_first_last
        subsample = large_map_data.subsample(
            limit_rows_per_metric=20, include_first_last=True
        )
        self.assertEqual(len(subsample.full_df), 40)
        # check that we 1 and 100 are included
        self.assertEqual(subsample.full_df[MAP_KEY].min(), 1)
        self.assertEqual(subsample.full_df[MAP_KEY].max(), 100)
        subsample = large_map_data.subsample(
            limit_rows_per_metric=20, include_first_last=False
        )
        self.assertEqual(len(subsample.full_df), 40)
        self.assertEqual(subsample.full_df[MAP_KEY].min(), 1)
        self.assertEqual(subsample.full_df[MAP_KEY].max(), 92)

        # test limit_rows_per_metric when some metrics are sparsely
        # reported (we shouldn't subsample those)
        subsample = large_map_data_sparse_metric.subsample(
            limit_rows_per_metric=100, include_first_last=False
        )
        full_df = large_map_data_sparse_metric.full_df
        subsample_map_df = subsample.full_df
        self.assertEqual(
            len(subsample_map_df[subsample_map_df["metric_name"] == "a"]), 85
        )
        self.assertEqual(
            len(subsample_map_df[subsample_map_df["metric_name"] == "b"]),
            len(full_df[full_df["metric_name"] == "b"]),
        )

        with self.subTest("Data without step column"):
            data = Data(df=large_map_data.df.drop(columns=["step"]))
            subsample = data.subsample(keep_every=10, limit_rows_per_group=3)
            self.assertIs(subsample, data)

    def test_dtype_conversion(self) -> None:
        df = self.df
        df[MAP_KEY] = df[MAP_KEY].astype(int)
        data = Data(df=df)
        self.assertEqual(data.full_df[MAP_KEY].dtype, float)

    def test_trial_indices(self) -> None:
        # Test that `trial_indices` is the same before and after setting `df`
        self.assertIsNone(self.mmd._memo_df)
        trial_indices = self.mmd.trial_indices
        self.mmd.df
        self.assertIsNotNone(self.mmd._memo_df)
        self.assertEqual(trial_indices, self.mmd.trial_indices)


class TestCombineDFs(TestCase):
    def test_combine_dfs_favoring_recent(self) -> None:
        original_mean = 2.0
        first_metric_first_data = {
            "arm_name": "0_0",
            "mean": original_mean,
            "sem": 0.2,
            "trial_index": 1,
            "metric_name": "a",
            "start_time": "2018-01-01",
            "end_time": "2018-01-02",
            "metric_signature": "a_signature",
        }
        df1 = pd.DataFrame(
            [
                # metric a
                first_metric_first_data,
                # metric b
                {
                    **first_metric_first_data,
                    **{
                        "metric_name": "b",
                        "metric_signature": "b_signature",
                    },
                },
            ]
        )
        # metric a again with a different mean
        new_mean = 3.7
        df2 = pd.DataFrame([{**first_metric_first_data, **{"mean": new_mean}}])
        result = combine_dfs_favoring_recent(last_df=df1, new_df=df2)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["metric_name"]), {"a", "b"})
        metric_a_mean = result.loc[result["metric_name"] == "a", "mean"].iloc[0]
        self.assertEqual(metric_a_mean, new_mean)

        with self.subTest("New data replaces old data"):
            result = combine_dfs_favoring_recent(last_df=df2, new_df=df1)
            self.assertEqual(len(result), 2)
            self.assertEqual(set(result["metric_name"]), {"a", "b"})
            metric_a_mean = result.loc[result["metric_name"] == "a", "mean"].iloc[0]
            self.assertEqual(metric_a_mean, original_mean)

        df3 = df1.assign(step=1.0)
        with self.subTest("With one df having 'step'"):
            # Both should be kept
            result = combine_dfs_favoring_recent(last_df=df1, new_df=df3)
            self.assertEqual(len(result), 2 * len(df3))

        with self.subTest("Both having 'step'"):
            result = combine_dfs_favoring_recent(last_df=df3, new_df=df3.copy())


class RelativizeDataTest(TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "mean": 2,
                    "sem": 0,
                    "metric_name": "foobar",
                    "metric_signature": "foobar",
                    "arm_name": "status_quo",
                },
                {
                    "trial_index": 0,
                    "mean": 5,
                    "sem": 0,
                    "metric_name": "foobaz",
                    "metric_signature": "foobaz",
                    "arm_name": "status_quo",
                },
                {
                    "trial_index": 0,
                    "mean": 1,
                    "sem": 0,
                    "metric_name": "foobar",
                    "metric_signature": "foobar",
                    "arm_name": "0_0",
                },
                {
                    "trial_index": 0,
                    "mean": 10,
                    "sem": 0,
                    "metric_name": "foobaz",
                    "metric_signature": "foobaz",
                    "arm_name": "0_0",
                },
            ]
        )

        self.expected_relativized_df = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "mean": -0.5,
                    "sem": 0,
                    "metric_name": "foobar",
                    "metric_signature": "foobar",
                    "arm_name": "0_0",
                },
                {
                    "trial_index": 0,
                    "mean": 1,
                    "sem": 0,
                    "metric_name": "foobaz",
                    "metric_signature": "foobaz",
                    "arm_name": "0_0",
                },
            ]
        )
        self.expected_relativized_df_with_sq = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "mean": 0,
                    "sem": 0,
                    "metric_name": "foobar",
                    "metric_signature": "foobar",
                    "arm_name": "status_quo",
                },
                {
                    "trial_index": 0,
                    "mean": -0.5,
                    "sem": 0,
                    "metric_name": "foobar",
                    "metric_signature": "foobar",
                    "arm_name": "0_0",
                },
                {
                    "trial_index": 0,
                    "mean": 0,
                    "sem": 0,
                    "metric_name": "foobaz",
                    "metric_signature": "foobaz",
                    "arm_name": "status_quo",
                },
                {
                    "trial_index": 0,
                    "mean": 1,
                    "sem": 0,
                    "metric_name": "foobaz",
                    "metric_signature": "foobaz",
                    "arm_name": "0_0",
                },
            ]
        )

    def test_relativize_data(self) -> None:
        data = Data(df=self.df)
        expected_relativized_data = Data(df=self.expected_relativized_df)

        expected_relativized_data_with_sq = Data(
            df=self.expected_relativized_df_with_sq
        )

        actual_relativized_data = data.relativize()
        self.assertEqual(expected_relativized_data, actual_relativized_data)

        actual_relativized_data_with_sq = data.relativize(include_sq=True)
        self.assertEqual(
            expected_relativized_data_with_sq, actual_relativized_data_with_sq
        )

        with self.subTest("step column not supported"):
            data = Data(df=self.df.assign(step=0))
            with self.assertRaisesRegex(
                NotImplementedError, "Relativization is not supported"
            ):
                data.relativize()

    def test_relativize_data_no_sem(self) -> None:
        df = self.df.copy()
        df["sem"] = np.nan
        data = Data(df=df)

        expected_relativized_df = self.expected_relativized_df.copy()
        expected_relativized_df["sem"] = np.nan
        expected_relativized_data = Data(df=expected_relativized_df)

        expected_relativized_df_with_sq = self.expected_relativized_df_with_sq.copy()
        expected_relativized_df_with_sq.loc[
            expected_relativized_df_with_sq["arm_name"] != "status_quo", "sem"
        ] = np.nan
        expected_relativized_data_with_sq = Data(df=expected_relativized_df_with_sq)

        actual_relativized_data = data.relativize()
        self.assertEqual(expected_relativized_data, actual_relativized_data)

        actual_relativized_data_with_sq = data.relativize(include_sq=True)
        self.assertEqual(
            expected_relativized_data_with_sq, actual_relativized_data_with_sq
        )

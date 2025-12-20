# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
import pandas as pd
from ax.core.data import Data
from ax.core.map_data import MAP_KEY, MapData
from ax.core.tests.test_data import TestDataBase
from ax.utils.common.testutils import TestCase


class TestMapData(TestDataBase):
    cls: type[MapData] = MapData


class MapDataTest(TestCase):
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

        self.mmd = MapData(df=self.df)

    def test_df(self) -> None:
        df = self.mmd.df
        self.assertEqual(set(df["trial_index"].drop_duplicates()), {0, 1})

    def test_init(self) -> None:
        # Initialize empty
        empty = MapData()
        self.assertTrue(empty.map_df.empty)

        # Check that the required columns include the map keys.
        self.assertEqual(
            empty.REQUIRED_COLUMNS.union([MAP_KEY]), empty.required_columns()
        )
        self.assertEqual(set(empty.map_df.columns), empty.required_columns())

    def test_combine(self) -> None:
        with self.subTest("From no MapDatas"):
            data = MapData.from_multiple_data([])
            self.assertIsInstance(data, MapData)
            self.assertEqual(data.map_df.size, 0)

        with self.subTest("From two MapDatas"):
            mmd_double = MapData.from_multiple_data([self.mmd, self.mmd])
            self.assertIsInstance(mmd_double, MapData)
            self.assertEqual(mmd_double.map_df.size, 2 * self.mmd.map_df.size)

        with self.subTest("From Datas"):
            data = Data(df=self.mmd.df)
            map_data = MapData.from_multiple_data([data])
            self.assertIsInstance(map_data, MapData)
            data = Data.from_multiple_data([data])
            self.assertEqual(len(data.full_df), len(map_data.full_df))

    def test_upcast(self) -> None:
        fresh = MapData(df=self.df)
        # Assert df is not cached before first call
        self.assertIsNone(fresh._memo_df)

        self.assertEqual(
            fresh.df.columns.size,
            fresh.map_df.columns.size,
        )

        # Assert df is cached after first call
        self.assertIsNotNone(fresh._memo_df)

        self.assertTrue(
            fresh.df.equals(
                fresh.map_df.sort_values(MAP_KEY).drop_duplicates(
                    MapData.DEDUPLICATE_BY_COLUMNS, keep="last"
                )
            )
        )

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
        large_map_data = MapData(df=large_map_df)

        shuffled_large_map_df = large_map_data.map_df.groupby(
            MapData.DEDUPLICATE_BY_COLUMNS
        ).sample(frac=1, random_state=seed)
        shuffled_large_map_data = MapData(df=shuffled_large_map_df)

        for rows_per_group in [1, 40]:
            large_map_data_latest = large_map_data.latest(rows_per_group=rows_per_group)

            if rows_per_group == 1:
                self.assertTrue(
                    large_map_data_latest.map_df.groupby("metric_name")[MAP_KEY]
                    .transform(lambda col: set(col) == set(max_epochs))
                    .all()
                )

            # when rows_per_group is larger than the number of rows
            # actually observed in a group
            actual_rows_per_group = large_map_data_latest.map_df.groupby(
                MapData.DEDUPLICATE_BY_COLUMNS
            ).size()
            expected_rows_per_group = np.minimum(
                large_map_data_latest.map_df.groupby(MapData.DEDUPLICATE_BY_COLUMNS)[
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
                shuffled_large_map_data_latest.map_df.equals(
                    large_map_data_latest.map_df
                )
            )

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
        large_map_data = MapData(df=large_map_df)
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
        large_map_data_sparse_metric = MapData(df=large_map_df_sparse_metric)

        # test keep_every
        subsample = large_map_data.subsample(keep_every=10)
        self.assertEqual(len(subsample.map_df), 52)
        subsample = large_map_data.subsample(keep_every=25)
        self.assertEqual(len(subsample.map_df), 20)
        subsample = large_map_data.subsample(limit_rows_per_group=7)
        self.assertEqual(len(subsample.map_df), 36)

        # test limit_rows_per_group
        subsample = large_map_data.subsample(limit_rows_per_group=1)
        self.assertEqual(len(subsample.map_df), 8)
        subsample = large_map_data.subsample(limit_rows_per_group=7)
        self.assertEqual(len(subsample.map_df), 36)
        subsample = large_map_data.subsample(limit_rows_per_group=10)
        self.assertEqual(len(subsample.map_df), 52)
        subsample = large_map_data.subsample(limit_rows_per_group=1000)
        self.assertEqual(len(subsample.map_df), 500)

        # test limit_rows_per_metric
        subsample = large_map_data.subsample(limit_rows_per_metric=50)
        self.assertEqual(len(subsample.map_df), 100)
        subsample = large_map_data.subsample(limit_rows_per_metric=65)
        self.assertEqual(len(subsample.map_df), 128)
        subsample = large_map_data.subsample(limit_rows_per_metric=1000)
        self.assertEqual(len(subsample.map_df), 500)

        # test include_first_last
        subsample = large_map_data.subsample(
            limit_rows_per_metric=20, include_first_last=True
        )
        self.assertEqual(len(subsample.map_df), 40)
        # check that we 1 and 100 are included
        self.assertEqual(subsample.map_df[MAP_KEY].min(), 1)
        self.assertEqual(subsample.map_df[MAP_KEY].max(), 100)
        subsample = large_map_data.subsample(
            limit_rows_per_metric=20, include_first_last=False
        )
        self.assertEqual(len(subsample.map_df), 40)
        self.assertEqual(subsample.map_df[MAP_KEY].min(), 1)
        self.assertEqual(subsample.map_df[MAP_KEY].max(), 92)

        # test limit_rows_per_metric when some metrics are sparsely
        # reported (we shouldn't subsample those)
        subsample = large_map_data_sparse_metric.subsample(
            limit_rows_per_metric=100, include_first_last=False
        )
        map_df = large_map_data_sparse_metric.map_df
        subsample_map_df = subsample.map_df
        self.assertEqual(
            len(subsample_map_df[subsample_map_df["metric_name"] == "a"]), 85
        )
        self.assertEqual(
            len(subsample_map_df[subsample_map_df["metric_name"] == "b"]),
            len(map_df[map_df["metric_name"] == "b"]),
        )

    def test_dtype_conversion(self) -> None:
        df = self.df
        df[MAP_KEY] = df[MAP_KEY].astype(int)
        data = MapData(df=df)
        self.assertEqual(data.map_df[MAP_KEY].dtype, float)

    def test_true_df_deprecation(self) -> None:
        with self.assertWarnsRegex(DeprecationWarning, "true_df"):
            true_df = self.mmd.true_df
        self.assertIs(true_df, self.mmd.map_df)

    def test_trial_indices(self) -> None:
        # Test that `trial_indices` is the same before and after setting `df`
        self.assertIsNone(self.mmd._memo_df)
        trial_indices = self.mmd.trial_indices
        self.mmd.df
        self.assertIsNotNone(self.mmd._memo_df)
        self.assertEqual(trial_indices, self.mmd.trial_indices)

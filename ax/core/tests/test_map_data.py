# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from math import isnan

import numpy as np
import pandas as pd
from ax.core.data import Data
from ax.core.map_data import _tail, MapData, MapKeyInfo
from ax.core.tests.test_data import TestDataBase
from ax.exceptions.core import UnsupportedError
from ax.utils.common.testutils import TestCase


class TestMapData(TestDataBase):
    cls: type[MapData] = MapData


class TestMapKeyInfo(TestCase):
    def test_init(self) -> None:
        with self.subTest("No default passed"):
            map_key_info = MapKeyInfo(key="epoch")
            self.assertTrue(isnan(map_key_info.default_value))

        with self.subTest("Default passed"):
            with self.assertWarnsRegex(Warning, "default_value will be treated as NaN"):
                map_key_info = MapKeyInfo(key="epoch", default_value=5.0)
            self.assertTrue(isnan(map_key_info.default_value))
            self.assertEqual(map_key_info.key, "epoch")


class MapDataTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.df = pd.DataFrame(
            [
                {
                    "arm_name": "0_0",
                    "epoch": 0,
                    "mean": 3.0,
                    "sem": 0.3,
                    "trial_index": 0,
                    "metric_name": "a",
                },
                # repeated arm 0_0
                {
                    "arm_name": "0_0",
                    "epoch": 0,
                    "mean": 2.0,
                    "sem": 0.2,
                    "trial_index": 1,
                    "metric_name": "a",
                },
                {
                    "arm_name": "0_0",
                    "epoch": 0,
                    "mean": 1.8,
                    "sem": 0.3,
                    "trial_index": 1,
                    "metric_name": "b",
                },
                {
                    "arm_name": "0_1",
                    "epoch": 0,
                    "mean": 4.0,
                    "sem": 0.6,
                    "trial_index": 1,
                    "metric_name": "a",
                },
                {
                    "arm_name": "0_1",
                    "epoch": 0,
                    "mean": 3.7,
                    "sem": 0.5,
                    "trial_index": 1,
                    "metric_name": "b",
                },
                {
                    "arm_name": "0_1",
                    "epoch": 1,
                    "mean": 0.5,
                    "sem": None,
                    "trial_index": 1,
                    "metric_name": "a",
                },
                {
                    "arm_name": "0_1",
                    "epoch": 1,
                    "mean": 3.0,
                    "sem": None,
                    "trial_index": 1,
                    "metric_name": "b",
                },
            ]
        )

        self.map_key = "epoch"
        self.map_key_infos = [MapKeyInfo(key=self.map_key)]

        self.mmd = MapData.from_df(df=self.df, map_key=self.map_key)

    def test_df(self) -> None:
        df = self.mmd.df
        self.assertEqual(set(df["trial_index"].drop_duplicates()), {0, 1})

    def test_map_key_info(self) -> None:
        self.assertEqual(self.map_key, self.mmd.map_key)
        self.assertEqual(self.map_key_infos, self.mmd.map_key_infos)

    def test_init(self) -> None:
        # Initialize empty with map key infos.
        empty = MapData(map_key_infos=self.map_key_infos)
        self.assertTrue(empty.map_df.empty)

        # Initialize empty with map key.
        empty = MapData.from_df(map_key=self.map_key)
        self.assertTrue(empty.map_df.empty)
        self.assertEqual(empty.map_key, self.map_key)

        # Check that the required columns include the map keys.
        self.assertEqual(
            empty.REQUIRED_COLUMNS.union(["epoch"]), empty.required_columns()
        )
        self.assertEqual(set(empty.map_df.columns), empty.required_columns())

        with self.assertRaisesRegex(ValueError, "map_key_infos may be `None` iff"):
            MapData(df=self.df, map_key_infos=None)

    def test_from_evaluations(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedError, "MapData.from_evaluations is not supported"
        ):
            MapData.from_evaluations(evaluations={}, trial_index=0)

    def test_properties(self) -> None:
        self.assertEqual(self.mmd.map_key_infos, self.map_key_infos)
        self.assertEqual(self.mmd.map_key, self.map_key)
        self.assertEqual(self.mmd.map_key, "epoch")

    def test_combine(self) -> None:
        with self.subTest("From no MapDatas"):
            data = MapData.from_multiple_map_data([])
            self.assertEqual(data.map_df.size, 0)

        with self.subTest("From two MapDatas with same map_key_info"):
            mmd_double = MapData.from_multiple_map_data([self.mmd, self.mmd])
            self.assertEqual(mmd_double.map_df.size, 2 * self.mmd.map_df.size)
            self.assertEqual(mmd_double.map_key_infos, self.mmd.map_key_infos)
            self.assertEqual(mmd_double.map_key, self.mmd.map_key)

        with self.subTest("From two MapDatas with different map_key_info keys"):
            different_map_df = pd.DataFrame(
                [
                    {
                        "arm_name": "0_3",
                        "timestamp": 11,
                        "mean": 2.0,
                        "sem": 0.2,
                        "trial_index": 1,
                        "metric_name": "a",
                    },
                    {
                        "arm_name": "0_3",
                        "timestamp": 18,
                        "mean": 1.8,
                        "sem": 0.3,
                        "trial_index": 1,
                        "metric_name": "b",
                    },
                ]
            )
            different_mmd = MapData.from_df(df=different_map_df, map_key="timestamp")

            with self.assertRaisesRegex(
                ValueError, "Received MapData with different map keys"
            ):
                MapData.from_multiple_map_data([self.mmd, different_mmd])

        with self.subTest("Only one has a map key"):
            data_df = pd.DataFrame(
                [
                    {
                        "arm_name": "0_4",
                        "mean": 2.0,
                        "sem": 0.2,
                        "trial_index": 1,
                        "metric_name": "a",
                    },
                    {
                        "arm_name": "0_4",
                        "mean": 1.8,
                        "sem": 0.3,
                        "trial_index": 1,
                        "metric_name": "b",
                    },
                ]
            )
            data = Data(df=data_df)

            downcast_combined = MapData.from_multiple_data([self.mmd, data])
            self.assertEqual(
                len(downcast_combined.map_df), len(self.mmd.map_df) + len(data.df)
            )
            self.assertEqual(
                downcast_combined.map_df.columns.size, self.mmd.map_df.columns.size
            )
            self.assertEqual(downcast_combined.map_key, self.map_key)
            self.assertEqual(downcast_combined.map_key_infos, self.map_key_infos)

        # Check that the Data's rows' epoch cell has the correct default value
        self.assertTrue(
            (
                downcast_combined.map_df[downcast_combined.map_df["arm_name"] == "0_4"][
                    "epoch"
                ].isnull()
            ).all()
        )

    def test_from_map_evaluations(self) -> None:
        for sem in (0.5, None):
            eval1 = (3.7, sem) if sem is not None else 3.7
            eval2 = (3.8, sem) if sem is not None else 3.8
            map_data = MapData.from_map_evaluations(
                evaluations={"0_1": [(1.0, {"b": eval1}), (1.0, {"b": eval2})]},
                trial_index=0,
            )
            self.assertEqual(map_data.map_df["sem"].isnull().all(), sem is None)
            self.assertEqual(len(map_data.map_df), 2)

    def test_upcast(self) -> None:
        fresh = MapData.from_df(df=self.df, map_key=self.map_key)
        self.assertIsNone(fresh._memo_df)  # Assert df is not cached before first call

        self.assertEqual(
            fresh.df.columns.size,
            fresh.map_df.columns.size,
        )

        self.assertIsNotNone(fresh._memo_df)  # Assert df is cached after first call

        self.assertTrue(
            fresh.df.equals(
                fresh.map_df.sort_values(fresh.map_key).drop_duplicates(
                    MapData.DEDUPLICATE_BY_COLUMNS, keep="last"
                )
            )
        )

    def test_latest(self) -> None:
        seed = 8888

        arm_names = ["0_0", "1_0", "2_0", "3_0"]
        max_epochs = [25, 50, 75, 100]
        metric_names = ["a", "b"]
        large_map_df = pd.DataFrame(
            [
                {
                    "arm_name": arm_name,
                    "epoch": epoch + 1,
                    "mean": epoch * 0.1,
                    "sem": 0.1,
                    "trial_index": trial_index,
                    "metric_name": metric_name,
                }
                for metric_name in metric_names
                for trial_index, (arm_name, max_epoch) in enumerate(
                    zip(arm_names, max_epochs)
                )
                for epoch in range(max_epoch)
            ]
        )
        large_map_data = MapData.from_df(df=large_map_df, map_key=self.map_key)

        shuffled_large_map_df = large_map_data.map_df.groupby(
            MapData.DEDUPLICATE_BY_COLUMNS
        ).sample(frac=1, random_state=seed)
        shuffled_large_map_data = MapData.from_df(
            df=shuffled_large_map_df, map_key=self.map_key
        )

        for rows_per_group in [1, 40]:
            large_map_data_latest = large_map_data.latest(rows_per_group=rows_per_group)

            if rows_per_group == 1:
                self.assertTrue(
                    large_map_data_latest.map_df.groupby("metric_name")
                    .epoch.transform(lambda col: set(col) == set(max_epochs))
                    .all()
                )

            # when rows_per_group is larger than the number of rows
            # actually observed in a group
            actual_rows_per_group = large_map_data_latest.map_df.groupby(
                MapData.DEDUPLICATE_BY_COLUMNS
            ).size()
            expected_rows_per_group = np.minimum(
                large_map_data_latest.map_df.groupby(MapData.DEDUPLICATE_BY_COLUMNS)
                .epoch.max()
                .astype(int),
                rows_per_group,
            )
            self.assertTrue(actual_rows_per_group.equals(expected_rows_per_group))

            # behavior should be consistent even if map_keys are not in ascending order
            shuffled_large_map_data_latest = shuffled_large_map_data.latest(
                rows_per_group=rows_per_group
            )
            self.assertTrue(
                shuffled_large_map_data_latest.map_df.equals(
                    large_map_data_latest.map_df
                )
            )

    def test_tail(self) -> None:
        """`_tail` is tested more thoroughly but implicitly in `test_latest`."""
        with self.assertRaisesRegex(ValueError, "`map_key` can only be None when"):
            _tail(map_df=self.mmd.map_df, map_key=None)

    def test_subsample(self) -> None:
        arm_names = ["0_0", "1_0", "2_0", "3_0"]
        max_epochs = [25, 50, 75, 100]
        metric_names = ["a", "b"]
        large_map_df = pd.DataFrame(
            [
                {
                    "arm_name": arm_name,
                    "epoch": epoch + 1,
                    "mean": epoch * 0.1,
                    "sem": 0.1,
                    "trial_index": trial_index,
                    "metric_name": metric_name,
                }
                for metric_name in metric_names
                for trial_index, (arm_name, max_epoch) in enumerate(
                    zip(arm_names, max_epochs)
                )
                for epoch in range(max_epoch)
            ]
        )
        large_map_data = MapData.from_df(df=large_map_df, map_key=self.map_key)
        large_map_df_sparse_metric = pd.DataFrame(
            [
                {
                    "arm_name": arm_name,
                    "epoch": epoch + 1,
                    "mean": epoch * 0.1,
                    "sem": 0.1,
                    "trial_index": trial_index,
                    "metric_name": metric_name,
                }
                for metric_name in metric_names
                for trial_index, (arm_name, max_epoch) in enumerate(
                    zip(arm_names, max_epochs)
                )
                for epoch in range(max_epoch if metric_name == "a" else max_epoch // 5)
            ]
        )
        large_map_data_sparse_metric = MapData.from_df(
            df=large_map_df_sparse_metric, map_key=self.map_key
        )

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
        self.assertEqual(subsample.map_df["epoch"].min(), 1)
        self.assertEqual(subsample.map_df["epoch"].max(), 100)
        subsample = large_map_data.subsample(
            limit_rows_per_metric=20, include_first_last=False
        )
        self.assertEqual(len(subsample.map_df), 40)
        self.assertEqual(subsample.map_df["epoch"].min(), 1)
        self.assertEqual(subsample.map_df["epoch"].max(), 92)

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

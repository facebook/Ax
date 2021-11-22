# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pandas as pd
from ax.core.data import Data
from ax.core.map_data import MapData, MapKeyInfo
from ax.utils.common.testutils import TestCase


class MapDataTest(TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            [
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

        self.map_key_infos = [
            MapKeyInfo(
                key="epoch",
                default_value=0,
            )
        ]

        self.mmd = MapData(df=self.df, map_key_infos=self.map_key_infos)

    def test_map_key_info(self):
        self.assertEqual(self.map_key_infos, self.mmd.map_key_infos)

        self.assertEqual(self.mmd.map_key_infos[0].key, "epoch")
        self.assertEqual(self.mmd.map_key_infos[0].default_value, 0)
        self.assertEqual(self.mmd.map_key_infos[0].value_type, int)

    def test_init(self):
        empty = MapData()
        self.assertTrue(empty.map_df.empty)

        with self.assertRaisesRegex(ValueError, "map_key_infos may be `None` iff"):
            MapData(df=self.df, map_key_infos=None)

    def test_properties(self):
        self.assertEqual(self.mmd.map_key_infos, self.map_key_infos)
        self.assertEqual(self.mmd.map_keys, ["epoch"])
        self.assertEqual(self.mmd.map_key_to_type, {"epoch": int})

    def test_combine(self):
        mmd_double = MapData.from_multiple_map_data([self.mmd, self.mmd])
        self.assertEqual(mmd_double.map_df.size, 2 * self.mmd.map_df.size)
        self.assertEqual(mmd_double.map_key_infos, self.mmd.map_key_infos)

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
        different_map_key_infos = [MapKeyInfo(key="timestamp", default_value=0.0)]
        different_mmd = MapData(
            df=different_map_df, map_key_infos=different_map_key_infos
        )

        combined = MapData.from_multiple_map_data([self.mmd, different_mmd])
        self.assertEqual(
            len(combined.map_df), len(self.mmd.map_df) + len(different_mmd.map_df)
        )
        self.assertEqual(combined.map_df.columns.size, self.mmd.map_df.columns.size + 1)
        self.assertEqual(
            combined.map_key_infos, self.map_key_infos + different_map_key_infos
        )

        combined_subset = MapData.from_multiple_map_data(
            [self.mmd, different_mmd], ["a"]
        )
        self.assertTrue((combined_subset.map_df["metric_name"] == "a").all())

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
        self.assertEqual(downcast_combined.map_key_infos, self.map_key_infos)

        # Check that the Data's rows' epoch cell has the correct default value
        self.assertTrue(
            (
                downcast_combined.map_df[downcast_combined.map_df["arm_name"] == "0_4"][
                    "epoch"
                ]
                == self.mmd.map_key_infos[0].default_value
            ).all()
        )

    def test_from_map_evaluations(self):
        map_data = MapData.from_map_evaluations(
            evaluations={
                "0_1": [
                    ({"f1": 1.0, "f2": 0.5}, {"b": (3.7, 0.5)}),
                    ({"f1": 1.0, "f2": 0.75}, {"b": (3.8, 0.5)}),
                ]
            },
            trial_index=0,
        )

        self.assertEqual(len(map_data.map_df), 2)
        self.assertEqual(set(map_data.map_keys), {"f1", "f2"})

        with self.assertRaisesRegex(
            ValueError, "Inconsistent map_key sets in evaluations"
        ):
            MapData.from_map_evaluations(
                evaluations={
                    "0_1": [
                        ({"f1": 1.0, "f2": 0.5}, {"b": (3.7, 0.5)}),
                    ]
                },
                map_key_infos=[MapKeyInfo(key="f1", default_value=0.0)],
                trial_index=0,
            )

    def test_upcast(self):
        fresh = MapData(df=self.df, map_key_infos=self.map_key_infos)
        self.assertIsNone(fresh._memo_df)  # Assert df is not cached before first call

        self.assertEqual(
            fresh.df.columns.size,
            fresh.map_df.columns.size - len(self.mmd.map_key_infos),
        )

        self.assertIsNotNone(fresh._memo_df)  # Assert df is cached after first call

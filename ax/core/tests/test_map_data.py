#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from ax.core.data import Data
from ax.core.map_data import MapData
from ax.utils.common.testutils import TestCase


class MapDataTest(TestCase):
    def setUp(self):
        self.df_hash = "bd72ac38f8ef98671118bda181013c2a"
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
        self.map_keys = ["epoch"]

    def testMapData(self):
        self.assertEqual(MapData(), MapData())
        map_data = MapData(df=self.df, map_keys=self.map_keys)
        self.assertEqual(map_data, map_data)
        self.assertEqual(map_data.df_hash, self.df_hash)

        df = map_data.df
        self.assertEqual(
            float(df[df["arm_name"] == "0_0"][df["metric_name"] == "a"]["mean"]), 2.0
        )
        self.assertEqual(
            float(
                df[df["arm_name"] == "0_1"][df["metric_name"] == "b"][df["epoch"] == 0][
                    "sem"
                ]
            ),
            0.5,
        )

    def testBadMapData(self):
        df = pd.DataFrame([{"bad_field": "0_0", "bad_field_2": {"x": 0, "y": "a"}}])
        with self.assertRaisesRegex(
            ValueError, "map_keys may only be `None` when `df` is also None "
        ):
            MapData(df=df)
        with self.assertRaisesRegex(
            ValueError, "Dataframe must contain required columns"
        ):
            MapData(map_keys=["bad_field"], df=df)

    def testFromMultipleData(self):
        data = [
            MapData(
                df=pd.DataFrame(
                    [
                        {
                            "arm_name": "0_1",
                            "mean": 3.7,
                            "sem": 0.5,
                            "metric_name": "b",
                            "epoch": 0,
                        },
                    ]
                ),
                map_keys=["epoch"],
            ),
            MapData(
                df=pd.DataFrame(
                    [
                        {
                            "arm_name": "0_1",
                            "mean": 3.7,
                            "sem": 0.5,
                            "metric_name": "b",
                            "epoch": 0,
                        },
                        {
                            "arm_name": "0_1",
                            "mean": 3.7,
                            "sem": 0.5,
                            "metric_name": "b",
                            "epoch": 1,
                        },
                    ]
                ),
                map_keys=["epoch"],
            ),
        ]

        merged_data = MapData.from_multiple_data(data)
        self.assertIsInstance(merged_data, MapData)
        self.assertEqual(3, merged_data.df.shape[0])

    def testFromMultipleDataValidation(self):
        # Non-MapData raises an error
        with self.assertRaisesRegex(ValueError, "Non-MapData in inputs."):
            data_elt_A = Data(
                df=pd.DataFrame(
                    [
                        {
                            "arm_name": "0_1",
                            "mean": 3.7,
                            "sem": 0.5,
                            "metric_name": "b",
                        }
                    ]
                ),
            )
            data_elt_B = Data(
                df=pd.DataFrame(
                    [
                        {
                            "arm_name": "0_1",
                            "mean": 3.7,
                            "sem": 0.5,
                            "metric_name": "b",
                        }
                    ]
                ),
            )
            MapData.from_multiple_data([data_elt_A, data_elt_B])
        # Inconsistent keys raise an error
        with self.assertRaisesRegex(
            ValueError, "Inconsistent map_keys found in data iterable."
        ):
            data_elt_A = MapData(
                df=pd.DataFrame(
                    [
                        {
                            "arm_name": "0_1",
                            "mean": 3.7,
                            "sem": 0.5,
                            "metric_name": "b",
                            "epoch": 0,
                        }
                    ]
                ),
                map_keys=["epoch"],
            )
            data_elt_B = MapData(
                df=pd.DataFrame(
                    [
                        {
                            "arm_name": "0_1",
                            "mean": 3.7,
                            "sem": 0.5,
                            "metric_name": "b",
                            "iteration": 1,
                        }
                    ]
                ),
                map_keys=["iteration"],
            )
            MapData.from_multiple_data([data_elt_A, data_elt_B])

    def testUpdate(self):
        base_data = MapData(
            df=pd.DataFrame(
                [
                    {
                        "arm_name": "0_1",
                        "mean": 3.7,
                        "sem": 0.5,
                        "metric_name": "b",
                        "epoch": 0,
                    }
                ]
            ),
            map_keys=["epoch"],
        )
        new_data = MapData(
            df=pd.DataFrame(
                [
                    {
                        "arm_name": "0_1",
                        "mean": 3.7,
                        "sem": 0.5,
                        "metric_name": "b",
                        "epoch": 0,
                    },
                    {
                        "arm_name": "0_1",
                        "mean": 3.7,
                        "sem": 0.5,
                        "metric_name": "b",
                        "epoch": 1,
                    },
                ]
            ),
            map_keys=["epoch"],
        )
        base_data.update(new_data=new_data)
        self.assertEqual(3, base_data.df.shape[0])

    def testUpdateValidation(self):
        base_data = MapData(
            df=pd.DataFrame(
                [
                    {
                        "arm_name": "0_1",
                        "mean": 3.7,
                        "sem": 0.5,
                        "metric_name": "b",
                        "epoch": 0,
                    }
                ]
            ),
            map_keys=["epoch"],
        )
        new_data_wrong_map_keys = MapData(
            df=pd.DataFrame(
                [
                    {
                        "arm_name": "0_1",
                        "mean": 3.7,
                        "sem": 0.5,
                        "metric_name": "b",
                        "iteration": 0,
                    }
                ]
            ),
            map_keys=["iteration"],
        )
        with self.assertRaisesRegex(
            ValueError, "Inconsistent map_keys found in new data."
        ):
            base_data.update(new_data=new_data_wrong_map_keys)

    def testFromMapEvaluations(self):
        map_data = MapData.from_map_evaluations(
            evaluations={
                "0_1": [
                    ({"f1": 1.0, "f2": 0.5}, {"b": (3.7, 0.5)}),
                    ({"f1": 1.0, "f2": 0.75}, {"b": (3.8, 0.5)}),
                ]
            },
            trial_index=0,
        )
        self.assertEqual(len(map_data.df), 2)
        self.assertEqual(map_data.map_keys, ["f1", "f2"])

        with self.assertRaises(ValueError):
            MapData.from_map_evaluations(
                evaluations={
                    "0_1": [
                        ({"f1": 1.0, "f2": 0.5}, {"b": (3.7, 0.5)}),
                        ({"epoch": 1.0, "mc_samples": 0.75}, {"b": (3.8, 0.5)}),
                    ]
                },
                trial_index=0,
            )

    def testCopyStructureWithDF(self):
        map_data = MapData(df=self.df, map_keys=self.map_keys)
        small_df = pd.DataFrame(
            [
                {
                    "arm_name": "0_1",
                    "mean": 3.7,
                    "sem": 0.5,
                    "metric_name": "b",
                    "epoch": 0,
                },
                {
                    "arm_name": "0_1",
                    "mean": 3.7,
                    "sem": 0.5,
                    "metric_name": "b",
                    "epoch": 1,
                },
            ]
        )
        new_map_data = map_data.copy_structure_with_df(df=small_df)
        self.assertEqual(new_map_data.map_keys, ["epoch"])

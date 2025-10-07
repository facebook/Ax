#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
from ax.core.formatting_utils import DataType, raw_evaluations_to_data
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase


class TestRawEvaluationsToData(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.metric_name_to_signature = {"a": "a_signature", "b": "b_signature"}

    def test_single_arm_single_metric(self) -> None:
        with self.subTest("No metric name provided"):
            signature = "objective_a_longname"
            for eval in [3.7, (3.7, None), (3.7, 0.5)]:
                data = raw_evaluations_to_data(
                    raw_data={"0_0": eval},
                    metric_name_to_signature={"a": signature},
                    trial_index=0,
                    data_type=DataType.DATA,
                )
                df = data.df
                self.assertEqual(df["mean"].iloc[0], 3.7)
                self.assertEqual(df["arm_name"].iloc[0], "0_0")
                self.assertEqual(df["trial_index"].iloc[0], 0)
                self.assertEqual(df["metric_signature"].iloc[0], signature)

        with self.subTest(
            "No metric name in eval and multiple in " "metric_name_to_signature"
        ):
            with self.assertRaisesRegex(
                UserInputError,
                "Metric name must be provided in `raw_data` if there are "
                "multiple metrics",
            ):
                raw_evaluations_to_data(
                    raw_data={"0_0": 3.7},
                    metric_name_to_signature=self.metric_name_to_signature,
                    trial_index=0,
                    data_type=DataType.DATA,
                )

        with self.subTest("Invalid DataType"):
            with self.assertRaisesRegex(
                UserInputError, "not compatible with `MapData`"
            ):
                raw_evaluations_to_data(
                    raw_data={"0_0": 5.0},
                    metric_name_to_signature=self.metric_name_to_signature,
                    trial_index=0,
                    data_type=DataType.MAP_DATA,
                )
        with self.subTest("missing signature"):
            data = raw_evaluations_to_data(
                raw_data={"0_0": {"b": 5.0}},
                metric_name_to_signature=self.metric_name_to_signature,
                trial_index=0,
                data_type=DataType.DATA,
            )

    def test_single_arm_multiple_metrics(self) -> None:
        data = raw_evaluations_to_data(
            raw_data={"arm_0": {"a": 5.0, "b": (2.0, 0.1)}},
            metric_name_to_signature=self.metric_name_to_signature,
            trial_index=1,
            data_type=DataType.DATA,
        )
        df = data.df
        self.assertEqual(df["metric_name"].tolist(), ["a", "b"])
        self.assertEqual(
            df["metric_signature"].tolist(), ["a_signature", "b_signature"]
        )
        self.assertEqual(df["mean"].tolist(), [5.0, 2.0])
        self.assertTrue(np.isnan(df["sem"].iloc[0]))
        self.assertEqual(df["sem"].iloc[1], 0.1)

    def test_multiple_arms(self) -> None:
        data = raw_evaluations_to_data(
            raw_data={
                "arm_0": {"a": 1.0},
                "arm_1": {"b": 2.0},
            },
            metric_name_to_signature=self.metric_name_to_signature,
            trial_index=2,
            data_type=DataType.DATA,
        )
        df = data.df
        self.assertSetEqual(set(df["arm_name"]), {"arm_0", "arm_1"})
        self.assertSetEqual(set(df["mean"]), {1.0, 2.0})

    def test_map_data(self) -> None:
        raw_data = {"arm_0": [(0, {"a": (1.0, 0.1)}), (1, {"b": (2.0, 0.2)})]}
        data = raw_evaluations_to_data(
            raw_data=raw_data,
            metric_name_to_signature=self.metric_name_to_signature,
            trial_index=3,
            data_type=DataType.MAP_DATA,
        )
        df = data.full_df
        self.assertEqual(len(df), 2)
        self.assertEqual(df["mean"].tolist(), [1.0, 2.0])
        self.assertEqual(df["sem"].tolist(), [0.1, 0.2])

        with self.subTest("Invalid data type"):
            with self.assertRaisesRegex(UserInputError, "not compatible with `Data`"):
                raw_evaluations_to_data(
                    raw_data=raw_data,
                    metric_name_to_signature=self.metric_name_to_signature,
                    trial_index=3,
                    data_type=DataType.DATA,
                )

    def test_numpy_types(self) -> None:
        arm_name = "0_0"
        inputs = {
            "float64": np.float64(1.6),
            "float32": np.float32(1.6),
            "float16": np.float16(1.6),
            "int64": np.int64(1),
        }
        for np_dtype, number in inputs.items():
            formats = {
                "Scalar": number,
                "Non-map": {"m": number},
                "Map": [(0, {"m": number}), (1, {"m": number})],
            }
            for format_name, evaluation in formats.items():
                data_type = DataType.MAP_DATA if format_name == "Map" else DataType.DATA
                with self.subTest(f"{np_dtype}, {format_name}"):
                    data = raw_evaluations_to_data(
                        raw_data={arm_name: evaluation},
                        metric_name_to_signature={"m": "m"},
                        trial_index=0,
                        data_type=data_type,
                    )
                    self.assertEqual(data.df.dtypes["mean"], float)
                    self.assertEqual(data.df["mean"].iloc[0].item(), float(number))

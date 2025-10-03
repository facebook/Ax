#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

import pandas as pd
from ax.core.data import Data
from ax.core.map_data import MAP_KEY, MapData
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from pyre_extensions import assert_is_instance

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


class TestDataBase(TestCase):
    """
    Covers both Data and MapData tests.

    MapData tests are in test_map_data.py.
    """

    cls: type[Data] = Data

    def setUp(self) -> None:
        super().setUp()
        df = get_test_dataframe()
        if self.cls is Data:
            self.df = df
            self.data_with_df = Data(df=self.df)
            self.data_without_df = Data()
        else:
            df_1 = df.copy().assign(**{MAP_KEY: 0})
            df_2 = df.copy().assign(**{MAP_KEY: 1})
            self.df = pd.concat((df_1, df_2))
            self.data_with_df = MapData(df=self.df)
            self.data_without_df = MapData()

    def test_init(self) -> None:
        # For Data, this is Data(). For MapData, this is MapData(map_keys=[]).
        self.assertEqual(self.data_without_df, self.data_without_df)
        self.assertEqual(self.data_with_df, self.data_with_df)

        df = self.data_with_df.df
        self.assertEqual(
            float(df[df["arm_name"] == "0_0"][df["metric_name"] == "a"]["mean"].item()),
            2.0,
        )
        self.assertEqual(
            float(df[df["arm_name"] == "0_1"][df["metric_name"] == "b"]["sem"].item()),
            0.5,
        )

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
        if self.cls is MapData:
            data = assert_is_instance(data, MapData)
            data_clone = assert_is_instance(data_clone, MapData)
            self.assertIsNot(data.map_df, data_clone.map_df)
            self.assertTrue(data.map_df.equals(data_clone.map_df))

    def test_BadData(self) -> None:
        df = pd.DataFrame([{"bad_field": "0_0", "bad_field_2": {"x": 0, "y": "a"}}])
        with self.assertRaisesRegex(
            ValueError, "Dataframe must contain required columns"
        ):
            if self.cls is Data:
                Data(df=df)
            else:
                MapData(df=df)

    def test_EmptyData(self) -> None:
        data = self.data_without_df
        df = data.df
        self.assertTrue(df.empty)
        self.assertTrue(self.cls.from_multiple_data([]).df.empty)

        if isinstance(data, MapData):
            self.assertTrue(data.map_df.empty)
            expected_columns = Data.REQUIRED_COLUMNS.union({MAP_KEY})
        else:
            expected_columns = Data.REQUIRED_COLUMNS
        self.assertEqual(expected_columns, data.required_columns())
        self.assertEqual(set(df.columns), expected_columns)

    def test_from_multiple_with_generator(self) -> None:
        data = self.cls.from_multiple_data(self.data_with_df for _ in range(2))
        self.assertEqual(len(data.full_df), 2 * len(self.data_with_df.full_df))

    def test_extra_columns(self) -> None:
        value = 3
        extra_col_df = self.df.assign(foo=value)
        data = self.cls(df=extra_col_df)
        self.assertIn("foo", data.full_df.columns)
        self.assertIn("foo", data.df.columns)
        self.assertTrue((data.full_df["foo"] == value).all())

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


class DataTest(TestCase):
    """Tests that are specific to Data and not shared with MapData."""

    def setUp(self) -> None:
        super().setUp()
        self.df_hash = "be6ca1edb2d83e08c460665476d32caa"
        self.df = get_test_dataframe()
        self.metric_name_to_signature = {"a": "a_signature", "b": "b_signature"}

    def test_repr(self) -> None:
        self.assertEqual(
            str(Data(df=self.df)),
            REPR_1000,
        )
        with patch(f"{Data.__module__}.DF_REPR_MAX_LENGTH", 500):
            self.assertEqual(str(Data(df=self.df)), REPR_500)

    def test_OtherClassInequality(self) -> None:
        class CustomData(Data):
            pass

        data = CustomData(df=self.df)
        self.assertNotEqual(data, Data(self.df))

    def test_FromEvaluations(self) -> None:
        for sem in (0.5, None):
            eval1 = (3.7, sem) if sem is not None else 3.7
            data = Data.from_evaluations(
                evaluations={"0_1": {"b": eval1}},
                metric_name_to_signature=self.metric_name_to_signature,
                trial_index=0,
            )
            self.assertEqual(data.df["sem"].isnull()[0], sem is None)
            self.assertEqual(len(data.df), 1)
            self.assertNotEqual(data, Data(self.df))

    def test_FromEvaluationsNameAndSignature(self) -> None:
        data = Data.from_evaluations(
            evaluations={"0_1": {"a": (3.7, 0.5)}},
            metric_name_to_signature=self.metric_name_to_signature,
            trial_index=0,
        )

        self.assertEqual(data.df["metric_name"][0], "a")
        self.assertEqual(data.df["metric_signature"][0], "a_signature")

    def test_FromEvaluationsMissingMetricSigMappingEntry(self) -> None:
        eval1 = (3.7, 0.5)
        with self.assertRaisesRegex(
            UserInputError, "Metric b not found in metric_name_to_signature"
        ):
            Data.from_evaluations(
                evaluations={"0_1": {"b": eval1}},
                metric_name_to_signature={"a": "a"},
                trial_index=0,
            )

    def test_FromEvaluationsExtraMetricSigMappingEntry(self) -> None:
        eval1 = (3.7, 0.5)
        extra_metric_name_to_signature = self.metric_name_to_signature
        extra_metric_name_to_signature["c"] = "c_signature"
        data = Data.from_evaluations(
            evaluations={"0_1": {"b": eval1}},
            metric_name_to_signature=extra_metric_name_to_signature,
            trial_index=0,
        )
        self.assertEqual(set(data.df["metric_signature"]), {"b_signature"})

    def test_from_multiple(self) -> None:
        with self.subTest("Combinining non-empty Data"):
            data = Data.from_multiple_data([Data(df=self.df), Data(df=self.df)])
            self.assertEqual(len(data.df), 2 * len(self.df))

        with self.subTest("Combining empty Data makes empty Data"):
            data = Data.from_multiple_data([Data(), Data()])
            self.assertEqual(data, Data())

        with self.subTest("Can't combine different types"):

            class CustomData(Data):
                pass

            with self.assertRaisesRegex(
                TypeError, "All data objects must be instances of"
            ):
                CustomData.from_multiple_data([Data(), CustomData()])

    def test_FromMultipleDataMismatchedTypes(self) -> None:
        # create two custom data types
        class CustomDataA(Data):
            pass

        class CustomDataB(Data):
            pass

        # Test using `Data.from_multiple_data` to combine non-Data types
        with self.assertRaisesRegex(TypeError, "All data objects must be instances of"):
            Data.from_multiple_data([CustomDataA(), CustomDataB()])

        # Test data of multiple non-empty types raises a value error
        data_elt_A = CustomDataA(df=self.df)
        data_elt_B = CustomDataB(df=self.df)
        with self.assertRaisesRegex(TypeError, "All data objects must be instances of"):
            Data.from_multiple_data([data_elt_A, data_elt_B])

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

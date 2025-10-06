#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
from ax.core.formatting_utils import (
    DataType,
    raw_data_to_evaluation,
    raw_evaluations_to_data,
)
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase


class TestRawDataToEvaluation(TestCase):
    def test_raw_data_is_not_dict_of_dicts(self) -> None:
        with self.assertRaisesRegex(
            UserInputError,
            "Raw data is expected to be just for one arm.",
        ):
            raw_data_to_evaluation(
                # pyre-fixme[6]: For 1st param expected `Union[Dict[str, Union[Tuple[...
                raw_data={"arm_0": {"objective_a": 6}},
                metric_names=["objective_a"],
            )

    def test_it_converts_to_floats_in_dict_and_leaves_tuples(self) -> None:
        result = raw_data_to_evaluation(
            # pyre-fixme[6]: For 1st param expected `Union[Dict[str, Union[Tuple[Unio...
            raw_data={
                "objective_a": 6,
                "objective_b": 1.0,
                "objective_c": ("some", "tuple"),
            },
            metric_names=["objective_a", "objective_b"],
        )
        # pyre-fixme[16]: Item `float` of `Union[Dict[str, typing.Union[typing.Tuple[...
        self.assertEqual(result["objective_a"], (6.0, None))
        # pyre-fixme[16]: Item `float` of `Union[Dict[str, typing.Union[typing.Tuple[...
        self.assertEqual(result["objective_b"], (1.0, None))
        # pyre-fixme[16]: Item `float` of `Union[Dict[str, typing.Union[typing.Tuple[...
        self.assertEqual(result["objective_c"], ("some", "tuple"))

    def test_dict_entries_must_be_int_float_or_tuple(self) -> None:
        with self.assertRaisesRegex(UserInputError, "Raw data for an arm is expected "):
            raw_data_to_evaluation(
                # pyre-fixme[6]: For 1st param expected `Union[Dict[str, Union[Tuple[...
                raw_data={"objective_a": [6.0, None]},
                metric_names=["objective_a"],
            )

    def test_it_requires_a_dict_for_multi_objectives(self) -> None:
        with self.assertRaisesRegex(
            UserInputError,
            "experiments with multiple metrics attached.",
        ):
            raw_data_to_evaluation(
                raw_data=(6.0, None),
                metric_names=["objective_a", "objective_b"],
            )

    def test_it_accepts_a_list_for_map_evaluations(self) -> None:
        raw_data = [(0.0, {"objective_a": (0, 1)}), (1.0, {"objective_a": (1.4, None)})]
        result = raw_data_to_evaluation(raw_data=raw_data, metric_names=["objective_a"])
        self.assertEqual(raw_data, result)

    def test_it_turns_a_tuple_into_a_dict(self) -> None:
        raw_data = (1.4, None)
        result = raw_data_to_evaluation(
            raw_data=raw_data,
            metric_names=["objective_a"],
        )
        # pyre-fixme[16]: Item `float` of `Union[Dict[str, typing.Union[typing.Tuple[...
        self.assertEqual(result["objective_a"], raw_data)

    def test_it_turns_an_int_into_a_dict_of_tuple(self) -> None:
        result = raw_data_to_evaluation(
            raw_data=1,
            metric_names=["objective_a"],
        )
        # pyre-fixme[16]: Item `float` of `Union[Dict[str, typing.Union[typing.Tuple[...
        self.assertEqual(result["objective_a"], (1.0, None))

    def test_it_turns_a_float_into_a_dict_of_tuple(self) -> None:
        result = raw_data_to_evaluation(
            raw_data=1.6,
            metric_names=["objective_a"],
        )
        # pyre-fixme[16]: Item `float` of `Union[Dict[str, typing.Union[typing.Tuple[...
        self.assertEqual(result["objective_a"], (1.6, None))

    def test_it_raises_for_unexpected_types(self) -> None:
        with self.assertRaisesRegex(
            UserInputError,
            "Raw data does not conform to the expected structure.",
        ):
            raw_data_to_evaluation(
                # pyre-fixme[6]: For 1st param expected `Union[Dict[str, Union[Tuple[...
                raw_data="1.6",
                metric_names=["objective_a"],
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

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.formatting_utils import raw_data_to_evaluation
from ax.utils.common.testutils import TestCase


class TestRawDataToEvaluation(TestCase):
    def test_raw_data_is_not_dict_of_dicts(self) -> None:
        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
            raw_data_to_evaluation(
                # pyre-fixme[6]: For 1st param expected `Union[Dict[str, Union[Tuple[...
                raw_data={"objective_a": [6.0, None]},
                metric_names=["objective_a"],
            )

    def test_it_requires_a_dict_for_multi_objectives(self) -> None:
        with self.assertRaises(ValueError):
            raw_data_to_evaluation(
                raw_data=(6.0, None),
                metric_names=["objective_a", "objective_b"],
            )

    def test_it_accepts_a_list_for_single_objectives(self) -> None:
        raw_data = [({"arm__0": {}}, {"objective_a": (1.4, None)})]
        result = raw_data_to_evaluation(
            # pyre-fixme[6]: For 1st param expected `Union[Dict[str, Union[Tuple[Unio...
            raw_data=raw_data,
            metric_names=["objective_a"],
        )
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
        with self.assertRaises(ValueError):
            raw_data_to_evaluation(
                # pyre-fixme[6]: For 1st param expected `Union[Dict[str, Union[Tuple[...
                raw_data="1.6",
                metric_names=["objective_a"],
            )

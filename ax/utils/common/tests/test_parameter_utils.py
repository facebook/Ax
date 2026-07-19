#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import TestCase

from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.types import TParamValue
from ax.exceptions.core import UserInputError
from ax.utils.common.parameter_utils import can_map_to_binary, is_unordered_choice


def get_unordered_choice(
    parameter_type: ParameterType, values: list[TParamValue]
) -> ChoiceParameter:
    return ChoiceParameter(
        "p", parameter_type=parameter_type, values=values, is_ordered=False
    )


def get_ordered_choice(
    parameter_type: ParameterType, values: list[TParamValue]
) -> ChoiceParameter:
    return ChoiceParameter(
        "p", parameter_type=parameter_type, values=values, is_ordered=True
    )


class TestParameterUtils(TestCase):
    def test_can_map_to_binary(self) -> None:
        for p in [
            RangeParameter("p", parameter_type=ParameterType.INT, lower=0, upper=1),
            RangeParameter("p", parameter_type=ParameterType.INT, lower=3, upper=4),
            get_unordered_choice(parameter_type=ParameterType.INT, values=[0, 1]),
            get_unordered_choice(
                parameter_type=ParameterType.STRING, values=["a", "b"]
            ),
        ]:
            self.assertTrue(can_map_to_binary(p))

        for p in [
            RangeParameter("p", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
            get_unordered_choice(parameter_type=ParameterType.INT, values=[0, 1, 2]),
            get_unordered_choice(
                parameter_type=ParameterType.STRING, values=["a", "b", "c"]
            ),
        ]:
            self.assertFalse(can_map_to_binary(p))

    def test_is_unordered_choice_parameter(self) -> None:
        for p in [
            get_unordered_choice(parameter_type=ParameterType.INT, values=[0, 1, 2]),
            get_unordered_choice(
                parameter_type=ParameterType.INT, values=[0, 1, 2, 4, 5]
            ),
            get_unordered_choice(
                parameter_type=ParameterType.STRING, values=["a", "b", "c", "d"]
            ),
        ]:
            self.assertTrue(is_unordered_choice(p, min_choices=3, max_choices=5))

        for p in [
            get_unordered_choice(parameter_type=ParameterType.INT, values=[0, 1]),
            get_ordered_choice(parameter_type=ParameterType.INT, values=[0, 1, 2, 4]),
            RangeParameter("p", parameter_type=ParameterType.INT, lower=0, upper=3),
            get_ordered_choice(
                parameter_type=ParameterType.STRING, values=["0", "1", "2"]
            ),
        ]:
            self.assertFalse(is_unordered_choice(p, min_choices=3, max_choices=5))

        # Check exceptions
        p = get_unordered_choice(parameter_type=ParameterType.INT, values=[0, 1, 2])
        with self.assertRaisesRegex(
            UserInputError, "`min_choices` must be a non-negative integer."
        ):
            is_unordered_choice(p, min_choices=-3)
        with self.assertRaisesRegex(
            UserInputError, "`max_choices` must be a non-negative integer."
        ):
            is_unordered_choice(p, max_choices=-1)
        with self.assertRaisesRegex(
            UserInputError, "`min_choices` cannot be larger than than `max_choices`."
        ):
            is_unordered_choice(p, min_choices=3, max_choices=2)

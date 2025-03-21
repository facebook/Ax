#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest

from ax.modelbridge.transforms.choice_encode import (
    ChoiceEncode,
    ChoiceToNumericChoice,
    OrderedChoiceEncode,
    OrderedChoiceToIntegerRange,
)

from ax.modelbridge.transforms.int_range_to_choice import IntRangeToChoice
from ax.modelbridge.transforms.map_key_to_float import MapKeyToFloat

from ax.modelbridge.transforms.task_encode import TaskChoiceToIntTaskChoice, TaskEncode

from ax.storage.json_store.decoders import transform_type_from_json
from ax.storage.json_store.encoders import transform_type_to_dict


class TestTransformEncode(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.deprecatedTestCases = [
            (OrderedChoiceEncode, OrderedChoiceToIntegerRange),
            (ChoiceEncode, ChoiceToNumericChoice),
            (TaskEncode, TaskChoiceToIntTaskChoice),
        ]

    def test_encode_and_decode_transform(self) -> None:
        with self.subTest("IntRangeToChoice"):
            transform_dict = transform_type_to_dict(IntRangeToChoice)
            self.assertEqual(transform_dict["__type"], "Type[Transform]")
            self.assertIn("IntRangeToChoice", transform_dict["transform_type"])

            decoded_transform_type = transform_type_from_json(transform_dict)
            self.assertEqual(decoded_transform_type, IntRangeToChoice)

        with self.subTest("MapKeyToFloat"):
            transform_dict = transform_type_to_dict(MapKeyToFloat)
            self.assertEqual(transform_dict["__type"], "Type[Transform]")
            self.assertIn("MapKeyToFloat", transform_dict["transform_type"])

            decoded_transform_type = transform_type_from_json(transform_dict)
            self.assertEqual(decoded_transform_type, MapKeyToFloat)

    def test_encode_and_decode_deprecated_transforms(self) -> None:
        for deprecated_type, current_type in self.deprecatedTestCases:
            transform_dict = transform_type_to_dict(deprecated_type)
            self.assertEqual(transform_dict["__type"], "Type[Transform]")
            self.assertIn(deprecated_type.__name__, transform_dict["transform_type"])

            decoded_transform_type = transform_type_from_json(transform_dict)
            # Deprecated type is decoded into the current type.
            self.assertNotEqual(decoded_transform_type, deprecated_type)
            self.assertEqual(decoded_transform_type, current_type)

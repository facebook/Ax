# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


import unittest

from ax.modelbridge.transforms.choice_encode import (
    ChoiceEncode,
    ChoiceToNumericChoice,
    OrderedChoiceEncode,
    OrderedChoiceToIntegerRange,
)

from ax.modelbridge.transforms.int_range_to_choice import IntRangeToChoice

from ax.storage.json_store.decoders import transform_type_from_json
from ax.storage.json_store.encoders import transform_type_to_dict


class TestTransformEncode(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_encode_and_decode_transform(self) -> None:
        registry_index = 2

        transform_dict = transform_type_to_dict(IntRangeToChoice)
        self.assertEqual(transform_dict["__type"], "Type[Transform]")
        self.assertEqual(transform_dict["index_in_registry"], registry_index)
        self.assertIn("IntRangeToChoice", transform_dict["transform_type"])

        decoded_transform_type = transform_type_from_json(transform_dict)
        self.assertEqual(decoded_transform_type, IntRangeToChoice)

    def test_encode_and_decode_deprecated_choice_transform(self) -> None:
        deprecated_type = ChoiceEncode
        current_type = ChoiceToNumericChoice
        registry_index = 19

        transform_dict = transform_type_to_dict(deprecated_type)
        self.assertEqual(transform_dict["__type"], "Type[Transform]")
        self.assertEqual(transform_dict["index_in_registry"], registry_index)
        self.assertIn("ChoiceEncode", transform_dict["transform_type"])

        decoded_transform_type = transform_type_from_json(transform_dict)
        self.assertEqual(decoded_transform_type, current_type)

    def test_encode_and_decode_deprecated_ordered_choice_transform(self) -> None:
        deprecated_type = OrderedChoiceEncode
        current_type = OrderedChoiceToIntegerRange
        registry_index = 7

        transform_dict = transform_type_to_dict(deprecated_type)
        self.assertEqual(transform_dict["__type"], "Type[Transform]")
        self.assertEqual(transform_dict["index_in_registry"], registry_index)
        self.assertIn("OrderedChoiceEncode", transform_dict["transform_type"])

        decoded_transform_type = transform_type_from_json(transform_dict)
        self.assertEqual(decoded_transform_type, current_type)

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.models.base import Model
from ax.utils.common.testutils import TestCase


class BaseModelTest(TestCase):
    def test_base_model(self) -> None:
        model = Model()
        raw_state = {"foo": "bar", "two": 3.0}
        self.assertEqual(model.serialize_state(raw_state), raw_state)
        self.assertEqual(model.deserialize_state(raw_state), raw_state)
        self.assertEqual(model._get_state(), {})
        with self.assertRaisesRegex(
            NotImplementedError, "Feature importance not available"
        ):
            model.feature_importances()

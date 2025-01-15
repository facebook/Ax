#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import (
    assert_is_instance_dict,
    assert_is_instance_list,
    assert_is_instance_optional,
)
from ax.utils.common.typeutils_nonnative import numpy_type_to_python_type
from pyre_extensions import assert_is_instance


class TestTypeUtils(TestCase):
    def test_assert_is_instance(self) -> None:
        self.assertEqual(assert_is_instance(2.0, float), 2.0)
        with self.assertRaises(TypeError):
            assert_is_instance(2, float)

    def test_assert_is_instance_list(self) -> None:
        self.assertEqual(assert_is_instance_list([1.0, 2.0], float), [1.0, 2.0])
        with self.assertRaises(TypeError):
            assert_is_instance_list([1.0, 2], float)

    def test_assert_is_instance_optional(self) -> None:
        self.assertEqual(assert_is_instance_optional(None, float), None)
        with self.assertRaises(TypeError):
            assert_is_instance_optional(2, float)

    def test_assert_is_instance_dict(self) -> None:
        self.assertEqual(assert_is_instance_dict({"some": 1}, str, int), {"some": 1})
        with self.assertRaises(TypeError):
            assert_is_instance_dict({"some": 1.0}, str, int)
        with self.assertRaises(TypeError):
            assert_is_instance_dict({1: 1}, str, int)

    def test_numpy_type_to_python_type(self) -> None:
        self.assertEqual(type(numpy_type_to_python_type(np.int64(2))), int)
        self.assertEqual(type(numpy_type_to_python_type(np.float64(2))), float)

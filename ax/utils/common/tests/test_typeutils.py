#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import (
    checked_cast,
    checked_cast_dict,
    checked_cast_list,
    checked_cast_optional,
    not_none,
    numpy_type_to_python_type,
)


class TestTypeUtils(TestCase):
    def test_not_none(self):
        self.assertEqual(not_none("not_none"), "not_none")
        with self.assertRaises(ValueError):
            not_none(None)

    def test_checked_cast(self):
        self.assertEqual(checked_cast(float, 2.0), 2.0)
        with self.assertRaises(ValueError):
            checked_cast(float, 2)

    def test_checked_cast_list(self):
        self.assertEqual(checked_cast_list(float, [1.0, 2.0]), [1.0, 2.0])
        with self.assertRaises(ValueError):
            checked_cast_list(float, [1.0, 2])

    def test_checked_cast_optional(self):
        self.assertEqual(checked_cast_optional(float, None), None)
        with self.assertRaises(ValueError):
            checked_cast_optional(float, 2)

    def test_checked_cast_dict(self):
        self.assertEqual(checked_cast_dict(str, int, {"some": 1}), {"some": 1})
        with self.assertRaises(ValueError):
            checked_cast_dict(str, int, {"some": 1.0})
        with self.assertRaises(ValueError):
            checked_cast_dict(str, int, {1: 1})

    def test_numpy_type_to_python_type(self):
        self.assertEqual(type(numpy_type_to_python_type(np.int64(2))), int)
        self.assertEqual(type(numpy_type_to_python_type(np.float64(2))), float)

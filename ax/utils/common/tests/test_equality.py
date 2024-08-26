#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from datetime import datetime

import numpy as np
import pandas as pd
from ax.utils.common.equality import (
    dataframe_equals,
    datetime_equals,
    equality_typechecker,
    is_ax_equal,
    object_attribute_dicts_find_unequal_fields,
    same_elements,
)
from ax.utils.common.testutils import TestCase


class EqualityTest(TestCase):
    def test_EqualityTypechecker(self) -> None:
        @equality_typechecker
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def eq(x, y):
            return x == y

        self.assertFalse(eq(5, 5.0))
        self.assertTrue(eq(5, 5))

    def test_ListsEquals(self) -> None:
        self.assertFalse(same_elements([0], [0, 1]))
        self.assertFalse(same_elements([1, 0], [0, 2]))
        self.assertFalse(same_elements([1, 1], [1, 2]))
        self.assertFalse(same_elements([1, 2], [1, 1]))
        self.assertFalse(same_elements([1, 1, 2], [1, 2, 2]))
        self.assertTrue(same_elements([1, 0], [0, 1]))

    def test_ListsEquals_floats(self) -> None:
        self.assertTrue(same_elements([0.0], [0.000000000000001]))
        self.assertTrue(same_elements([float("nan")], [float("nan")]))

    def test_DatetimeEquals(self) -> None:
        now = datetime.now()
        self.assertTrue(datetime_equals(None, None))
        self.assertFalse(datetime_equals(None, now))
        self.assertTrue(datetime_equals(now, now))

    def test_DataframeEquals(self) -> None:
        pd1 = pd.DataFrame.from_records([{"x": 100, "y": 200}])
        pd2 = pd.DataFrame.from_records([{"y": 200, "x": 100}])
        pd3 = pd.DataFrame.from_records([{"x": 100, "y": 300}])

        self.assertTrue(dataframe_equals(pd.DataFrame(), pd.DataFrame()))
        self.assertTrue(dataframe_equals(pd1, pd2))
        self.assertFalse(dataframe_equals(pd1, pd3))

    def test_numpy_equals(self) -> None:
        # Simple check.
        np_0 = {"cov": np.array([[0.1, 0.0], [0.0, 0.1]])}
        np_1 = {"cov": np.array([[0.1, 0.0], [0.0, 0.1]])}
        self.assertEqual(
            object_attribute_dicts_find_unequal_fields(np_0, np_1), ({}, {})
        )
        # Unequal.
        np_1 = {"cov": np.array([[0.1, 0.0], [0.1, 0.1]])}
        self.assertEqual(
            object_attribute_dicts_find_unequal_fields(np_0, np_1),
            ({}, {"cov": (np_0["cov"], np_1["cov"])}),
        )
        # With NaNs.
        np_1 = {"cov": np.array([[0.1, float("nan")], [float("nan"), 0.1]])}
        self.assertEqual(
            object_attribute_dicts_find_unequal_fields(np_0, np_1),
            ({}, {"cov": (np_0["cov"], np_1["cov"])}),
        )
        np_0 = {"cov": np.array([[0.1, float("nan")], [float("nan"), 0.1]])}
        self.assertEqual(
            object_attribute_dicts_find_unequal_fields(np_0, np_1), ({}, {})
        )

    def test_is_ax_equal_with_different_types(self) -> None:
        self.assertTrue(is_ax_equal(1, 1.0))
        self.assertTrue(is_ax_equal(1, np.ones(1)))
        self.assertFalse(is_ax_equal(1, np.ones(5)))

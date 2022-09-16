#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ax.plot.helper import arm_name_to_sort_key, extend_range
from ax.utils.common.testutils import TestCase


class HelperTest(TestCase):
    def test_extend_range(self) -> None:
        with self.assertRaises(ValueError):
            extend_range(lower=1, upper=-1)
        self.assertEqual(extend_range(lower=-1, upper=1), (-1.2, 1.2))
        self.assertEqual(extend_range(lower=-1, upper=0, percent=30), (-1.3, 0.3))
        self.assertEqual(extend_range(lower=0, upper=1, percent=50), (-0.5, 1.5))

    def test_arm_name_to_sort_key(self) -> None:
        arm_names = ["0_0", "1_10", "1_2", "10_0", "control"]
        sorted_names = sorted(arm_names, key=arm_name_to_sort_key, reverse=True)
        expected = ["control", "0_0", "1_2", "1_10", "10_0"]
        self.assertEqual(sorted_names, expected)

        arm_names = ["0_0", "0", "1_10", "3_2_x", "3_x", "1_2", "control"]
        sorted_names = sorted(arm_names, key=arm_name_to_sort_key, reverse=True)
        expected = ["control", "3_x", "3_2_x", "0", "0_0", "1_2", "1_10"]
        self.assertEqual(sorted_names, expected)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ax.plot.helper import extend_range
from ax.utils.common.testutils import TestCase


class HelperTest(TestCase):
    def test_extend_range(self):
        with self.assertRaises(ValueError):
            extend_range(lower=1, upper=-1)
        self.assertEqual(extend_range(lower=-1, upper=1), (-1.2, 1.2))
        self.assertEqual(extend_range(lower=-1, upper=0, percent=30), (-1.3, 0.3))
        self.assertEqual(extend_range(lower=0, upper=1, percent=50), (-0.5, 1.5))

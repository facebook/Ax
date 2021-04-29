#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.modelbridge.transforms.utils import ClosestLookupDict
from ax.utils.common.testutils import TestCase


class TransformUtilsTest(TestCase):
    def test_closest_lookup_dict(self):
        # test empty lookup
        d = ClosestLookupDict()
        with self.assertRaises(RuntimeError):
            d[0]
        # basic test
        keys = (1.0, 2, 4)
        vals = ("a", "b", "c")
        d = ClosestLookupDict(zip(keys, vals))
        for k, v in zip(keys, vals):
            self.assertEqual(d[k], v)
        self.assertEqual(d[2.5], "b")
        self.assertEqual(d[0], "a")
        self.assertEqual(d[6], "c")
        with self.assertRaises(ValueError):
            d["str_key"] = 3

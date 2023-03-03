#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from ax.utils.common.testutils import TestCase
from ax.utils.testing.utils import generic_equals


class TestUtils(TestCase):
    def test_generic_equals(self) -> None:
        # Basics.
        self.assertTrue(generic_equals(5, 5))
        self.assertFalse(generic_equals(5, 1))
        self.assertTrue(generic_equals("abc", "abc"))
        self.assertFalse(generic_equals("abc", "abcd"))
        self.assertFalse(generic_equals("abc", 5))
        # With tensors.
        self.assertTrue(generic_equals(torch.ones(2), torch.ones(2)))
        self.assertFalse(generic_equals(torch.ones(2), torch.zeros(2)))
        self.assertFalse(generic_equals(torch.ones(2), [0, 0]))
        # Dictionaries.
        self.assertTrue(generic_equals({"a": torch.ones(2)}, {"a": torch.ones(2)}))
        self.assertFalse(generic_equals({"a": torch.ones(2)}, {"a": torch.zeros(2)}))
        self.assertFalse(
            generic_equals({"a": torch.ones(2)}, {"a": torch.ones(2), "b": 5})
        )
        self.assertFalse(generic_equals({"a": torch.ones(2)}, [torch.ones(2)]))
        self.assertTrue(
            generic_equals({"a": torch.ones(2), "b": 2}, {"b": 2, "a": torch.ones(2)})
        )
        # Tuple / list.
        self.assertTrue(generic_equals([3, 2], [3, 2]))
        self.assertTrue(generic_equals([3, (2, 3)], [3, (2, 3)]))
        self.assertFalse(generic_equals([3, (2, 3)], [3, (2, 4)]))
        self.assertFalse(generic_equals([0, 1], range(2)))
        # Other.
        self.assertTrue(generic_equals(range(2), range(2)))
        self.assertTrue(generic_equals(np.ones(2), np.ones(2)))
        self.assertFalse(generic_equals(np.ones(2), np.zeros(2)))
        self.assertTrue(generic_equals({1, 2}, {1, 2}))
        self.assertFalse(generic_equals({1, 2}, {1, 2, 3}))

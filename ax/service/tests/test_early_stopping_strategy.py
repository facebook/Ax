#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.service.early_stopping_strategy import BaseEarlyStoppingStrategy
from ax.utils.common.testutils import TestCase


class TestEarlyStoppingStrategy(TestCase):
    def test_early_stopping_strategy(self):
        # can't instantiate abstract class
        with self.assertRaises(TypeError):
            BaseEarlyStoppingStrategy()

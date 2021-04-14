#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.service.early_stopping_strategy import BaseEarlyStoppingStrategy
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class TestEarlyStoppingStrategy(TestCase):
    def test_early_stopping_strategy(self):
        exp = get_branin_experiment()
        early_stopping_strategy = BaseEarlyStoppingStrategy()
        should_stop = early_stopping_strategy.should_stop_trial_early(
            trial_index=0, experiment=exp
        )
        self.assertFalse(
            should_stop, f"Expected should_stop to be False, got {should_stop}"
        )

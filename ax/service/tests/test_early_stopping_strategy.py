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
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices={0}, experiment=exp
        )
        self.assertDictEqual(should_stop, {0: None})

        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices={0, 1}, experiment=exp
        )
        self.assertDictEqual(should_stop, {0: None, 1: None})

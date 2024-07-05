#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.exceptions.core import AxWarning
from ax.telemetry.generation_strategy import GenerationStrategyCreatedRecord
from ax.utils.common.testutils import TestCase
from ax.utils.testing.modeling_stubs import (
    get_generation_strategy,
    sobol_gpei_generation_node_gs,
)


class TestGenerationStrategy(TestCase):
    def test_generation_strategy_created_record_from_generation_strategy(self) -> None:
        gs = get_generation_strategy()
        record = GenerationStrategyCreatedRecord.from_generation_strategy(
            generation_strategy=gs
        )
        expected = GenerationStrategyCreatedRecord(
            generation_strategy_name="Sobol+BO_MIXED",
            num_requested_initialization_trials=6,
            num_requested_bayesopt_trials=-1,
            num_requested_other_trials=0,
            max_parallelism=3,
        )
        self.assertEqual(record, expected)

    def test_generation_strategy_created_record_node_based(self) -> None:
        gs = sobol_gpei_generation_node_gs()
        with self.assertWarnsRegex(
            AxWarning,
            "`GenerationStrategyCreatedRecord` does not fully support node-based*",
        ):
            record = GenerationStrategyCreatedRecord.from_generation_strategy(
                generation_strategy=gs
            )
        expected = GenerationStrategyCreatedRecord(
            generation_strategy_name="Sobol+MBM_Nodes",
            num_requested_initialization_trials=None,
            num_requested_bayesopt_trials=None,
            num_requested_other_trials=None,
            max_parallelism=None,
        )
        self.assertEqual(record, expected)

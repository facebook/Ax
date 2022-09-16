# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.scheduler import SchedulerOptions
from ax.utils.common.testutils import TestCase


class TestBenchmarkMethod(TestCase):
    def test_benchmark_method(self) -> None:
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=10,
                )
            ],
            name="SOBOL",
        )
        options = SchedulerOptions(total_trials=10)
        method = BenchmarkMethod(
            name="Sobol10", generation_strategy=gs, scheduler_options=options
        )

        self.assertEqual(method.generation_strategy, gs)
        self.assertEqual(method.scheduler_options, options)

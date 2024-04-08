# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

from ax.benchmark.benchmark_method import (
    BenchmarkMethod,
    get_benchmark_scheduler_options,
)
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.scheduler import SchedulerOptions


def get_sobol_benchmark_method(
    distribute_replications: bool,
    scheduler_options: Optional[SchedulerOptions] = None,
) -> BenchmarkMethod:
    generation_strategy = GenerationStrategy(
        name="Sobol",
        steps=[GenerationStep(model=Models.SOBOL, num_trials=-1)],
    )

    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=scheduler_options or get_benchmark_scheduler_options(),
        distribute_replications=distribute_replications,
    )

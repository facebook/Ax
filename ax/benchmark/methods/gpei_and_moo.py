# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.scheduler import SchedulerOptions


def get_gpei_default() -> BenchmarkMethod:
    generation_strategy = GenerationStrategy(
        name="SOBOL+GPEI::default",
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=5),
            GenerationStep(
                model=Models.GPEI,
                num_trials=-1,
                max_parallelism=1,
            ),
        ],
    )

    scheduler_options = SchedulerOptions(
        init_seconds_between_polls=0, max_pending_trials=1
    )

    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=scheduler_options,
    )


def get_moo_default() -> BenchmarkMethod:
    generation_strategy = GenerationStrategy(
        name="SOBOL+MOO::default",
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=5),
            GenerationStep(
                model=Models.MOO,
                num_trials=-1,
                max_parallelism=1,
            ),
        ],
    )

    scheduler_options = SchedulerOptions(
        init_seconds_between_polls=0, max_pending_trials=1
    )

    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=scheduler_options,
    )

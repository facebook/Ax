# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from ax.benchmark.benchmark_method import (
    BenchmarkMethod,
    get_sequential_optimization_scheduler_options,
)
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.scheduler import SchedulerOptions


def get_saasbo_default(
    scheduler_options: Optional[SchedulerOptions] = None,
    distribute_replications: bool = True,
) -> BenchmarkMethod:
    generation_strategy = GenerationStrategy(
        name="SOBOL+FULLYBAYESIAN::default",
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=5, min_trials_observed=5),
            GenerationStep(
                model=Models.SAASBO,
                num_trials=-1,
                max_parallelism=1,
            ),
        ],
    )

    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=scheduler_options
        or get_sequential_optimization_scheduler_options(),
        distribute_replications=distribute_replications,
    )


def get_saasbo_moo_default(
    scheduler_options: Optional[SchedulerOptions] = None,
    distribute_replications: bool = True,
) -> BenchmarkMethod:
    generation_strategy = GenerationStrategy(
        name="SOBOL+FULLYBAYESIANMOO::default",
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=5, min_trials_observed=5),
            GenerationStep(
                model=Models.SAASBO,
                num_trials=-1,
                max_parallelism=1,
            ),
        ],
    )

    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=scheduler_options
        or get_sequential_optimization_scheduler_options(),
        distribute_replications=distribute_replications,
    )

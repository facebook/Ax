# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.service.scheduler import SchedulerOptions


def get_saasbo_default() -> BenchmarkMethod:
    generation_strategy = GenerationStrategy(
        name="SOBOL+FULLYBAYESIAN::default",
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=5, min_trials_observed=3),
            GenerationStep(
                model=Models.FULLYBAYESIAN,
                num_trials=-1,
            ),
        ],
    )

    scheduler_options = SchedulerOptions(total_trials=30)

    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=scheduler_options,
    )


def get_saasbo_moo_default() -> BenchmarkMethod:
    generation_strategy = GenerationStrategy(
        name="SOBOL+FULLYBAYESIANMOO::default",
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=5, min_trials_observed=3),
            GenerationStep(
                model=Models.FULLYBAYESIANMOO,
                num_trials=-1,
            ),
        ],
    )

    scheduler_options = SchedulerOptions(total_trials=30)

    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=scheduler_options,
    )

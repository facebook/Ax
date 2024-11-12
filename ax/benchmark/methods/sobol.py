# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models


def get_sobol_generation_strategy() -> GenerationStrategy:
    return GenerationStrategy(
        name="Sobol",
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=-1),
        ],
    )


def get_sobol_benchmark_method(
    distribute_replications: bool,
    batch_size: int = 1,
) -> BenchmarkMethod:
    return BenchmarkMethod(
        generation_strategy=get_sobol_generation_strategy(),
        batch_size=batch_size,
        distribute_replications=distribute_replications,
    )

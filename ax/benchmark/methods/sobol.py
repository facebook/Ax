# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.adapter.registry import Generators
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.generation_strategy.generation_strategy import (
    GenerationStep,
    GenerationStrategy,
)


def get_sobol_generation_strategy() -> GenerationStrategy:
    return GenerationStrategy(
        name="Sobol",
        steps=[
            GenerationStep(generator=Generators.SOBOL, num_trials=-1),
        ],
    )


def get_sobol_benchmark_method(
    batch_size: int = 1,
) -> BenchmarkMethod:
    return BenchmarkMethod(
        generation_strategy=get_sobol_generation_strategy(),
        batch_size=batch_size,
    )

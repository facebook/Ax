# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.service.scheduler import SchedulerOptions


def get_choose_generation_strategy_method(
    problem: BenchmarkProblem, num_trials: int = 30
) -> BenchmarkMethod:
    generation_strategy = choose_generation_strategy(
        search_space=problem.search_space,
        optimization_config=problem.optimization_config,
        num_trials=num_trials,
    )

    return BenchmarkMethod(
        name=f"ChooseGenerationStrategy::{problem.name}",
        generation_strategy=generation_strategy,
        scheduler_options=SchedulerOptions(total_trials=num_trials),
    )

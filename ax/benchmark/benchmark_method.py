# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service.utils.scheduler_options import SchedulerOptions
from ax.utils.common.base import Base


# Used as a default for the `replications_per_machine` argument in
# `BenchmarkMethod` and its subclasses. For more information, see the docstring
# for `BenchmarkMethod`.
DEFAULT_REPLICATIONS_PER_MACHINE = 10


@dataclass(frozen=True)
class BenchmarkMethod(Base):
    """
    Benchmark method, represented in terms of Ax generation strategy (which
    tells us which models to use when) and scheduler options (which tell us
    extra execution information like maximum parallelism, early stopping
    configuration, etc.).

    Note: If BenchmarkMethod.scheduler_options.total_trials is lower than
    BenchmarkProblem.num_trials, only the number of trials specified in the
    former will be run.

    Args:
        name: Name of the method.
        generation_strategy: A `GenerationStrategy`, used to specify the
            optimization method.
        scheduler_options: A `SchedulerOptions` object, specifying how the
            benchmarks will be run (e.g. sequentially or not).
        replications_per_machine: How many replications to run (sequentially) on
            a single machine. A benchmark replication can be run with
            `ax.benchmark.benchmark.benchmark_replication`. Slower methods will
            need a smaller value.
    """

    name: str
    generation_strategy: GenerationStrategy
    scheduler_options: SchedulerOptions
    replications_per_machine: int = DEFAULT_REPLICATIONS_PER_MACHINE


def get_sequential_optimization_scheduler_options(
    timeout_hours: int = 4,
) -> SchedulerOptions:
    """The typical SchedulerOptions used in benchmarking.

    Args:
        timeout_hours: The maximum amount of time (in hours) to run each
            benchmark replication. Defaults to 4 hours.
    """
    return SchedulerOptions(
        # Enforce sequential trials by default
        max_pending_trials=1,
        # Do not throttle, as is often necessary when polling real endpoints
        init_seconds_between_polls=0,
        min_seconds_before_poll=0,
        timeout_hours=timeout_hours,
    )

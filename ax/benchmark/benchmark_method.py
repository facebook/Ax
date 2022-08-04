# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service.utils.scheduler_options import SchedulerOptions
from ax.utils.common.base import Base


@dataclass(frozen=True)
class BenchmarkMethod(Base):
    """Benchmark method, represented in terms of Ax generation strategy (which tells us
    which models to use when) and scheduler options (which tell us extra execution
    information like maximum parallelism, early stopping configuration, etc.). Note:
    if BenchmarkMethod.scheduler_optionss.total_trials is lower than
    BenchmarkProblem.num_trials only the number of trials specified in the former will
    be run.
    """

    name: str
    generation_strategy: GenerationStrategy
    scheduler_options: SchedulerOptions


def get_sequential_optimization_scheduler_options() -> SchedulerOptions:
    """The typical SchedulerOptions used in benchmarking."""
    return SchedulerOptions(
        # Enforce sequential trials by default
        max_pending_trials=1,
        # Do not throttle, as is often necessary when polling real endpoints
        init_seconds_between_polls=0,
        min_seconds_before_poll=0,
        # Time the experiment out after 4 hours
        timeout_hours=4,
    )

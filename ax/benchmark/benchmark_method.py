# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from dataclasses import dataclass

from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service.utils.scheduler_options import SchedulerOptions, TrialType
from ax.utils.common.base import Base
from ax.utils.common.logger import get_logger


logger: logging.Logger = get_logger("BenchmarkMethod")


@dataclass(frozen=True)
class BenchmarkMethod(Base):
    """Benchmark method, represented in terms of Ax generation strategy (which tells us
    which models to use when) and scheduler options (which tell us extra execution
    information like maximum parallelism, early stopping configuration, etc.).

    Note: If `BenchmarkMethod.scheduler_options.total_trials` is less than
    `BenchmarkProblem.num_trials` then only the number of trials specified in the
    former will be run.

    Args:
        name: String description.
        generation_strategy: The `GenerationStrategy` to use.
        scheduler_options: `SchedulerOptions` that specify options such as
            `max_pending_trials`, `timeout_hours`, and `batch_size`. Can be
            generated with sensible defaults for benchmarking with
            `get_benchmark_scheduler_options`.
        distribute_replications: Indicates whether the replications should be
            run in a distributed manner. Ax itself does not use this attribute.
    """

    name: str
    generation_strategy: GenerationStrategy
    scheduler_options: SchedulerOptions
    distribute_replications: bool = False


def get_benchmark_scheduler_options(
    timeout_hours: int = 4,
    batch_size: int = 1,
) -> SchedulerOptions:
    """The typical SchedulerOptions used in benchmarking.

    Currently, regardless of batch size, all pending trials must complete before
    new ones are generated. That is, when batch_size > 1, the design is "batch
    sequential", and when batch_size = 1, the design is "fully sequential."

    Args:
        timeout_hours: The maximum amount of time (in hours) to run each
            benchmark replication. Defaults to 4 hours.
        batch_size: Number of trials to generate at once.
    """

    return SchedulerOptions(
        # No new candidates can be generated while any are pending.
        # If batched, an entire batch must finish before the next can be
        # generated.
        max_pending_trials=1,
        # Do not throttle, as is often necessary when polling real endpoints
        init_seconds_between_polls=0,
        min_seconds_before_poll=0,
        timeout_hours=timeout_hours,
        trial_type=TrialType.TRIAL if batch_size == 1 else TrialType.BATCH_TRIAL,
        batch_size=batch_size,
    )

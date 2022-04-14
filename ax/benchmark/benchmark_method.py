# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from ax.exceptions.core import UserInputError
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service.utils.scheduler_options import SchedulerOptions
from ax.utils.common.base import Base


@dataclass(frozen=True)
class BenchmarkMethod(Base):
    """Benchmark method, represented in terms of Ax generation strategy (which tells us
    which models to use when) and scheduler options (which tell us extra execution
    information like maximum parallelism, early stopping configuration, etc.)
    """

    name: str
    generation_strategy: GenerationStrategy
    scheduler_options: SchedulerOptions

    def __post_init__(self) -> None:
        if self.scheduler_options.total_trials is None:
            raise UserInputError(
                "SchedulerOptions.total_trials may not be None in BenchmarkMethod."
            )

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass

from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy

from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.utils.common.base import Base


@dataclass(kw_only=True)
class BenchmarkMethod(Base):
    """Benchmark method, represented in terms of Ax generation strategy (which tells us
    which models to use when) and Orchestrator options (which tell us extra execution
    information like maximum parallelism, early stopping configuration, etc.).

    Args:
        name: String description.
        generation_strategy: The `GenerationStrategy` to use.
        timeout_hours: Number of hours after which to stop a benchmark
            replication.
        distribute_replications: Indicates whether the replications should be
            run in a distributed manner. Ax itself does not use this attribute.
        batch_size: Number of arms per trial. If greater than 1, trials are
            ``BatchTrial``s; otherwise, they are ``Trial``s. Defaults to 1. This
            and the following arguments are passed to ``OrchestratorOptions``.
        run_trials_in_batches: Passed to ``OrchestratorOptions``.
        max_pending_trials: Passed to ``OrchestratorOptions``.
    """

    name: str = "DEFAULT"
    generation_strategy: GenerationStrategy

    timeout_hours: float = 4.0
    distribute_replications: bool = False

    batch_size: int | None = 1
    run_trials_in_batches: bool = False
    max_pending_trials: int = 1
    early_stopping_strategy: BaseEarlyStoppingStrategy | None = None

    def __post_init__(self) -> None:
        if self.name == "DEFAULT":
            self.name = self.generation_strategy.name

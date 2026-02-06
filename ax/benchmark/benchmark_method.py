# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from dataclasses import dataclass, field

from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.utils.common.base import Base


@dataclass(kw_only=True)
class BenchmarkMethod(Base):
    """Benchmark method, represented in terms of Ax generation strategy (which tells us
    which models to use when) and Orchestrator options (which tell us extra execution
    information like maximum concurrency, early stopping configuration, etc.).

    Args:
        name: String description. Defaults to the name of the generation strategy.
        generation_strategy: The `GenerationStrategy` to use.
        batch_size: Number of arms per trial. Defaults to 1. If greater than 1,
            trials are ``BatchTrial``s; otherwise, they are ``Trial``s.
            Passed to ``OrchestratorOptions``.
        max_concurrent_trials: Passed to ``OrchestratorOptions``.
        early_stopping_strategy: Passed to ``OrchestratorOptions``.
    """

    name: str = "DEFAULT"
    generation_strategy: GenerationStrategy
    # Options for the Orchestrator.
    batch_size: int | None = 1
    max_concurrent_trials: int = 1
    early_stopping_strategy: BaseEarlyStoppingStrategy | None = None
    # Deprecated
    max_pending_trials: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.max_pending_trials is not None:
            warnings.warn(
                "`max_pending_trials` is deprecated and will be removed in Ax 1.4. "
                "Use `max_concurrent_trials` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.max_concurrent_trials != 1:
                raise UserInputError(
                    "Cannot specify both `max_pending_trials` and "
                    "`max_concurrent_trials`."
                )
            object.__setattr__(self, "max_concurrent_trials", self.max_pending_trials)
            object.__setattr__(self, "max_pending_trials", None)
        if self.name == "DEFAULT":
            self.name = self.generation_strategy.name

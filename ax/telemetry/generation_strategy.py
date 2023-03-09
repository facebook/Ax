# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from ax.modelbridge.generation_strategy import GenerationStrategy

from ax.telemetry.common import INITIALIZATION_MODELS, OTHER_MODELS


@dataclass(frozen=True)
class GenerationStrategyCreatedRecord:
    """
    Record of the GenerationStrategy creation event. This can be used for telemetry in
    settings where many GenerationStrategy are being created either manually or
    programatically. In order to facilitate easy serialization only include simple
    types: numbers, strings, bools, and None.
    """

    generation_strategy_name: str

    # -1 indicates unlimited trials requested, 0 indicates no trials requested
    num_requested_initialization_trials: int  # Typically the number of Sobol trials
    num_requested_bayesopt_trials: int
    num_requested_other_trials: int

    # Minimum `max_parallelism` across GenerationSteps, i.e. the bottleneck
    max_parallelism: int

    @classmethod
    def from_generation_strategy(
        cls, generation_strategy: GenerationStrategy
    ) -> GenerationStrategyCreatedRecord:
        # Minimum `max_parallelism` across GenerationSteps, i.e. the bottleneck
        true_max_parallelism = min(
            step.max_parallelism or inf for step in generation_strategy._steps
        )

        return cls(
            generation_strategy_name=generation_strategy.name,
            num_requested_initialization_trials=sum(
                step.num_trials
                for step in generation_strategy._steps
                if step.model in INITIALIZATION_MODELS
            ),
            num_requested_bayesopt_trials=sum(
                step.num_trials
                for step in generation_strategy._steps
                if step.model not in INITIALIZATION_MODELS + OTHER_MODELS
            ),
            num_requested_other_trials=sum(
                step.num_trials
                for step in generation_strategy._steps
                if step.model in OTHER_MODELS
            ),
            max_parallelism=true_max_parallelism
            if isinstance(true_max_parallelism, int)
            else -1,
        )

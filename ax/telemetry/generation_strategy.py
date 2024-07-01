# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import inf
from typing import Optional

from ax.exceptions.core import AxWarning
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
    num_requested_initialization_trials: Optional[
        int  # Typically the number of Sobol trials
    ]
    num_requested_bayesopt_trials: Optional[int]
    num_requested_other_trials: Optional[int]

    # Minimum `max_parallelism` across GenerationSteps, i.e. the bottleneck
    max_parallelism: Optional[int]

    @classmethod
    def from_generation_strategy(
        cls, generation_strategy: GenerationStrategy
    ) -> GenerationStrategyCreatedRecord:
        if generation_strategy.is_node_based:
            warnings.warn(
                "`GenerationStrategyCreatedRecord` does not fully support node-based "
                "generation strategies. This will result in an incomplete record.",
                category=AxWarning,
                stacklevel=4,
            )
            # TODO [T192965545]: Support node-based generation strategies in telemetry
            return cls(
                generation_strategy_name=generation_strategy.name,
                num_requested_initialization_trials=None,
                num_requested_bayesopt_trials=None,
                num_requested_other_trials=None,
                max_parallelism=None,
            )

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
            max_parallelism=(
                true_max_parallelism if isinstance(true_max_parallelism, int) else -1
            ),
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Sequence
from functools import reduce

from ax.core.experiment import Experiment
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_node import GenerationNode


class LogicalEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    def __init__(
        self,
        left: BaseEarlyStoppingStrategy,
        right: BaseEarlyStoppingStrategy,
    ) -> None:
        super().__init__()

        self.left = left
        self.right = right


class AndEarlyStoppingStrategy(LogicalEarlyStoppingStrategy):
    def _is_harmful(
        self,
        trial_indices: set[int],
        experiment: Experiment,
    ) -> bool:
        """AND logic: harmful if either strategy considers it harmful."""
        return self.left._is_harmful(
            trial_indices=trial_indices,
            experiment=experiment,
        ) or self.right._is_harmful(
            trial_indices=trial_indices,
            experiment=experiment,
        )

    def _should_stop_trials_early(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        current_node: GenerationNode | None = None,
    ) -> dict[int, str | None]:
        left = self.left.should_stop_trials_early(
            trial_indices=trial_indices,
            experiment=experiment,
            current_node=current_node,
        )
        right = self.right.should_stop_trials_early(
            trial_indices=trial_indices,
            experiment=experiment,
            current_node=current_node,
        )
        return {
            trial: f"{left[trial]}, {right[trial]}" for trial in left if trial in right
        }


class OrEarlyStoppingStrategy(LogicalEarlyStoppingStrategy):
    @classmethod
    def from_early_stopping_strategies(
        cls,
        strategies: Sequence[BaseEarlyStoppingStrategy],
    ) -> BaseEarlyStoppingStrategy:
        if len(strategies) < 1:
            raise UserInputError("strategies must not be empty")

        return reduce(
            lambda left, right: OrEarlyStoppingStrategy(left=left, right=right),
            strategies[1:],
            strategies[0],
        )

    def _is_harmful(
        self,
        trial_indices: set[int],
        experiment: Experiment,
    ) -> bool:
        """OR logic: harmful if both strategies consider it harmful."""
        return self.left._is_harmful(
            trial_indices=trial_indices,
            experiment=experiment,
        ) and self.right._is_harmful(
            trial_indices=trial_indices,
            experiment=experiment,
        )

    def _should_stop_trials_early(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        current_node: GenerationNode | None = None,
    ) -> dict[int, str | None]:
        return {
            **self.left.should_stop_trials_early(
                trial_indices=trial_indices,
                experiment=experiment,
                current_node=current_node,
            ),
            **self.right.should_stop_trials_early(
                trial_indices=trial_indices,
                experiment=experiment,
                current_node=current_node,
            ),
        }

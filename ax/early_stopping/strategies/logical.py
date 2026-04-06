# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Sequence
from functools import reduce

from ax.core.experiment import Experiment
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy, TArmsToStop
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

    def _should_stop_arms(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        current_node: GenerationNode | None = None,
    ) -> TArmsToStop:
        left = self.left.should_stop_arms(
            trial_indices=trial_indices,
            experiment=experiment,
            current_node=current_node,
        )
        right = self.right.should_stop_arms(
            trial_indices=trial_indices,
            experiment=experiment,
            current_node=current_node,
        )
        # Combine at the arm level: only stop arms that both strategies agree on
        result: TArmsToStop = {}
        for trial in left:
            if trial in right:
                combined_arms: dict[str, str | None] = {}
                for arm_name in left[trial]:
                    if arm_name in right[trial]:
                        combined_arms[arm_name] = (
                            f"{left[trial][arm_name]}, {right[trial][arm_name]}"
                        )
                if combined_arms:
                    result[trial] = combined_arms
        return result


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

    def _should_stop_arms(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        current_node: GenerationNode | None = None,
    ) -> TArmsToStop:
        left = self.left.should_stop_arms(
            trial_indices=trial_indices,
            experiment=experiment,
            current_node=current_node,
        )
        right = self.right.should_stop_arms(
            trial_indices=trial_indices,
            experiment=experiment,
            current_node=current_node,
        )
        # Merge at arm level: stop arms that either strategy wants to stop
        result: TArmsToStop = {}
        all_trials = set(left) | set(right)
        for trial in all_trials:
            left_arms = left.get(trial, {})
            right_arms = right.get(trial, {})
            merged_arms: dict[str, str | None] = {}
            for arm_name in set(left_arms) | set(right_arms):
                reasons = []
                if arm_name in left_arms and left_arms[arm_name] is not None:
                    reasons.append(left_arms[arm_name])
                if arm_name in right_arms and right_arms[arm_name] is not None:
                    reasons.append(right_arms[arm_name])
                merged_arms[arm_name] = ", ".join(reasons) if reasons else None
            if merged_arms:
                result[trial] = merged_arms
        return result

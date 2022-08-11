# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import reduce
from typing import Any, Dict, Optional, Sequence, Set

from ax.core.experiment import Experiment
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.exceptions.core import UserInputError


class LogicalEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    def __init__(
        self,
        left: BaseEarlyStoppingStrategy,
        right: BaseEarlyStoppingStrategy,
        seconds_between_polls: int = 300,
        true_objective_metric_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            seconds_between_polls=seconds_between_polls,
            true_objective_metric_name=true_objective_metric_name,
        )

        self.left = left
        self.right = right


class AndEarlyStoppingStrategy(LogicalEarlyStoppingStrategy):
    def should_stop_trials_early(
        self,
        trial_indices: Set[int],
        experiment: Experiment,
        **kwargs: Dict[str, Any],
    ) -> Dict[int, Optional[str]]:

        left = self.left.should_stop_trials_early(
            trial_indices=trial_indices, experiment=experiment, **kwargs
        )
        right = self.right.should_stop_trials_early(
            trial_indices=trial_indices, experiment=experiment, **kwargs
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

    def should_stop_trials_early(
        self,
        trial_indices: Set[int],
        experiment: Experiment,
        **kwargs: Dict[str, Any],
    ) -> Dict[int, Optional[str]]:
        return {
            **self.left.should_stop_trials_early(
                trial_indices=trial_indices, experiment=experiment, **kwargs
            ),
            **self.right.should_stop_trials_early(
                trial_indices=trial_indices, experiment=experiment, **kwargs
            ),
        }

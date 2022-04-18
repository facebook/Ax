# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Set

from ax.core.experiment import Experiment
from ax.early_stopping.strategies import BaseEarlyStoppingStrategy
from ax.utils.common.typeutils import not_none


def should_stop_trials_early(
    early_stopping_strategy: Optional[BaseEarlyStoppingStrategy],
    trial_indices: Set[int],
    experiment: Experiment,
) -> Dict[int, Optional[str]]:
    """Evaluate whether to early-stop running trials.

    Args:
        early_stopping_strategy: A ``BaseEarlyStoppingStrategy`` that determines
            whether a trial should be stopped given the state of an experiment.
        trial_indices: Indices of trials to consider for early stopping.
        experiment: The experiment containing the trials.

    Returns:
        A dictionary mapping trial indices that should be early stopped to
        (optional) messages with the associated reason.
    """
    if early_stopping_strategy is None:
        return {}

    early_stopping_strategy = not_none(early_stopping_strategy)
    return early_stopping_strategy.should_stop_trials_early(
        trial_indices=trial_indices, experiment=experiment
    )

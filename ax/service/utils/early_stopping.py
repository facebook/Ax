# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.core.experiment import Experiment
from ax.early_stopping.strategies import BaseEarlyStoppingStrategy
from pyre_extensions import none_throws


def should_stop_trials_early(
    early_stopping_strategy: BaseEarlyStoppingStrategy | None,
    trial_indices: set[int],
    experiment: Experiment,
) -> dict[int, str | None]:
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

    early_stopping_strategy = none_throws(early_stopping_strategy)
    return early_stopping_strategy.should_stop_trials_early(
        trial_indices=trial_indices, experiment=experiment
    )


def get_early_stopping_metrics(
    experiment: Experiment, early_stopping_strategy: BaseEarlyStoppingStrategy | None
) -> list[str]:
    """A helper function that returns a list of metric names on which a given
    `early_stopping_strategy` is operating."""
    if early_stopping_strategy is None:
        return []
    if early_stopping_strategy.metric_names is not None:
        return list(early_stopping_strategy.metric_names)
    # TODO: generalize this to multi-objective ess
    default_objective, _ = early_stopping_strategy._default_objective_and_direction(
        experiment=experiment
    )
    return [default_objective]

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import sys
from datetime import datetime
from enum import Enum, unique
from math import ceil
from typing import Any

from ax.core import ObservationFeatures
from ax.core.base_trial import STATUSES_EXPECTING_DATA, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.exceptions.generation_strategy import AxGenerationException

from ax.modelbridge.generation_node import GenerationNode
from ax.utils.common.constants import Keys
from ax.utils.common.typeutils import checked_cast


@unique
class NodeInputConstructors(Enum):
    """An enum which maps to a the name of a callable method for constructing
    ``GenerationNode`` inputs.

    NOTE: The methods defined by this enum should all share identical signatures
    and reside in this file.
    """

    ALL_N = "consume_all_n"
    REPEAT_N = "repeat_arm_n"
    REMAINING_N = "remaining_n"
    TARGET_TRIAL_FIXED_FEATURES = "set_target_trial"

    def __call__(
        self,
        previous_node: GenerationNode | None,
        next_node: GenerationNode,
        gs_gen_call_kwargs: dict[str, Any],
        experiment: Experiment,
    ) -> int:
        """Defines a callable method for the Enum as all values are methods"""
        try:
            method = getattr(sys.modules[__name__], self.value)
        except AttributeError:
            raise ValueError(
                f"{self.value} is not defined as a method in "
                "``generation_node_input_constructors.py``. Please add the method "
                "to the file."
            )
        return method(
            previous_node=previous_node,
            next_node=next_node,
            gs_gen_call_kwargs=gs_gen_call_kwargs,
            experiment=experiment,
        )


@unique
class InputConstructorPurpose(Enum):
    """A simple enum to indicate the purpose of the input constructor.

    Explanation of the different purposes:
        N: Defines the logic to determine the number of arms to generate from the
           next ``GenerationNode`` given the total number of arms expected in
           this trial.
    """

    N = "n"
    FIXED_FEATURES = "fixed_features"


def set_target_trial(
    previous_node: GenerationNode | None,
    next_node: GenerationNode,
    gs_gen_call_kwargs: dict[str, Any],
    experiment: Experiment,
) -> ObservationFeatures | None:
    """Determine the target trial for the next node based on the current state of the
    ``Experiment``.

     Args:
        previous_node: The previous node in the ``GenerationStrategy``. This is the node
            that is being transition away from, and is provided for easy access to
            properties of this node.
        next_node: The next node in the ``GenerationStrategy``. This is the node that
            will leverage the inputs defined by this input constructor.
        gs_gen_call_kwargs: The kwargs passed to the ``GenerationStrategy``'s
            gen call.
        experiment: The experiment associated with this ``GenerationStrategy``.
    Returns:
        An ``ObservationFeatures`` object that defines the target trial for the next
        node.
    """
    target_trial_idx = _get_target_trial_index(
        experiment=experiment, next_node=next_node
    )
    return ObservationFeatures(
        parameters={},
        trial_index=target_trial_idx,
    )


def consume_all_n(
    previous_node: GenerationNode | None,
    next_node: GenerationNode,
    gs_gen_call_kwargs: dict[str, Any],
    experiment: Experiment,
) -> int:
    """Generate total requested number of arms from the next node.

    Example: Initial exploration with Sobol will generate all arms from a
    single sobol node.

    Args:
        previous_node: The previous node in the ``GenerationStrategy``. This is the node
            that is being transition away from, and is provided for easy access to
            properties of this node.
        next_node: The next node in the ``GenerationStrategy``. This is the node that
            will leverage the inputs defined by this input constructor.
        gs_gen_call_kwargs: The kwargs passed to the ``GenerationStrategy``'s
            gen call.
        experiment: The experiment associated with this ``GenerationStrategy``.
    Returns:
        The total number of requested arms from the next node.
    """
    # TODO: @mgarrard handle case where n isn't specified
    if gs_gen_call_kwargs.get("n") is None:
        raise NotImplementedError(
            f"Currently `{consume_all_n.__name__}` only supports cases where n is "
            "specified"
        )
    return gs_gen_call_kwargs.get("n")


def repeat_arm_n(
    previous_node: GenerationNode | None,
    next_node: GenerationNode,
    gs_gen_call_kwargs: dict[str, Any],
    experiment: Experiment,
) -> int:
    """Generate a small percentage of arms requested to be used for repeat arms in
    the next trial.

    Args:
        previous_node: The previous node in the ``GenerationStrategy``. This is the node
            that is being transition away from, and is provided for easy access to
            properties of this node.
        next_node: The next node in the ``GenerationStrategy``. This is the node that
            will leverage the inputs defined by this input constructor.
        gs_gen_call_kwargs: The kwargs passed to the ``GenerationStrategy``'s
            gen call.
        experiment: The experiment associated with this ``GenerationStrategy``.
    Returns:
        The number of requested arms from the next node
    """
    if gs_gen_call_kwargs.get("n") is None:
        raise NotImplementedError(
            f"Currently `{repeat_arm_n.__name__}` only supports cases where n is "
            "specified"
        )
    total_n = gs_gen_call_kwargs.get("n")
    if total_n < 6:
        # if the next trial is small, we don't want to waste allocation on repeat arms
        # users can still manually add repeat arms if they want before allocation
        return 0
    elif total_n <= 10:
        return 1
    return ceil(total_n / 10)


def remaining_n(
    previous_node: GenerationNode | None,
    next_node: GenerationNode,
    gs_gen_call_kwargs: dict[str, Any],
    experiment: Experiment,
) -> int:
    """Generate the remaining number of arms requested for this trial in gs.gen().

    Args:
        previous_node: The previous node in the ``GenerationStrategy``. This is the node
            that is being transition away from, and is provided for easy access to
            properties of this node.
        next_node: The next node in the ``GenerationStrategy``. This is the node that
            will leverage the inputs defined by this input constructor.
        gs_gen_call_kwargs: The kwargs passed to the ``GenerationStrategy``'s
            gen call.
        experiment: The experiment associated with this ``GenerationStrategy``.
    Returns:
        The number of requested arms from the next node
    """
    if gs_gen_call_kwargs.get("n") is None:
        raise NotImplementedError(
            f"Currently `{remaining_n.__name__}` only supports cases where n is "
            "specified"
        )
    # TODO: @mgarrard improve this logic to be more robust
    grs_this_gen = gs_gen_call_kwargs.get("grs_this_gen")
    total_n = gs_gen_call_kwargs.get("n")
    # if all arms have been generated, return 0
    return max(total_n - sum(len(gr.arms) for gr in grs_this_gen), 0)


# Helper methods for input constructors
def _get_target_trial_index(experiment: Experiment, next_node: GenerationNode) -> int:
    """Get the index of the target trial in the ``Experiment``.

    Find the target trial giving priority in the following order:
        1. a running long-run trial
        2. Most recent trial expecting data with running trials be considered the most
            recent

    In the event of any ties, the tie breaking order is:
        a. longest running trial by duration
        b. trial with most arms
        c. arbitraty selection

    Args:
        experiment: The experiment associated with this ``GenerationStrategy``.

    Returns:
        The index of the target trial in the ``Experiment``.
    """
    # TODO: @mgarrard improve logic to include trial_obsolete_threshold that
    # takes into account the age of the trial, and consider more heavily weighting
    # long run trials.
    running_trials = [
        checked_cast(BatchTrial, trial)
        for trial in experiment.trials_by_status[TrialStatus.RUNNING]
    ]
    sorted_running_trials = _sort_trials(trials=running_trials, trials_are_running=True)
    # Priority 1: Any running long-run trial
    target_trial_idx = next(
        (
            trial.index
            for trial in sorted_running_trials
            if trial.trial_type == Keys.LONG_RUN
        ),
        None,
    )
    if target_trial_idx is not None:
        return target_trial_idx

    # Priority 2: longest running currently running trial
    if len(sorted_running_trials) > 0:
        return sorted_running_trials[0].index

    # Priortiy 3: the longest running trial expecting data, discounting running trials
    # as we handled those above
    trials_expecting_data = [
        checked_cast(BatchTrial, trial)
        for trial in experiment.trials_expecting_data
        if trial.status != TrialStatus.RUNNING
    ]
    sorted_trials_expecting_data = _sort_trials(
        trials=trials_expecting_data, trials_are_running=False
    )
    if len(sorted_trials_expecting_data) > 0:
        return sorted_trials_expecting_data[0].index

    raise AxGenerationException(
        f"Attempting to construct for input into {next_node} but no trials match the "
        "expected conditions. Often this could be due to no trials on the experiment "
        f"that are in status {STATUSES_EXPECTING_DATA} on the experiment. The trials "
        f"on this experiment are: {experiment.trials}."
    )
    return 0


def _sort_trials(
    trials: list[BatchTrial],
    trials_are_running: bool,
) -> list[BatchTrial]:
    """Sort a list of trials by (1) duration of trial, (2) number of arms in trial.

    Args:
        trials: The trials to sort.
        trials_are_running: Whether the trials are running or not, used to determine
            the trial duration for sorting

    Returns:
        The sorted trials.
    """
    default_time_run_started = datetime.now()
    twelve_hours_in_secs = 12 * 60 * 60
    sorted_trials_expecting_data = sorted(
        trials,
        key=lambda t: (
            # First sorting criterion: trial duration, if a trial's duration is within
            # 12 hours of another trial, we consider them to be a tie
            int(
                (
                    # if the trial is running, we set end time to now for sorting ease
                    (
                        _time_trial_completed_safe(trial=t).timestamp()
                        if not trials_are_running
                        else default_time_run_started.timestamp()
                    )
                    - _time_trial_started_safe(
                        trial=t, default_time_run_started=default_time_run_started
                    ).timestamp()
                )
                // twelve_hours_in_secs
            ),
            # In event of a tie, we want the trial with the most arms
            +len(t.arms_by_name),
        ),
        reverse=True,
    )
    return sorted_trials_expecting_data


def _time_trial_started_safe(
    trial: BatchTrial, default_time_run_started: datetime
) -> datetime:
    """Not all RUNNING trials have ``time_run_started`` defined.
    This function accepts, but penalizes those trials by using a
    default ``time_run_started``, which moves them to the end of
    the sort because they would be running a very short time.

    Args:
        trial: The trial to check.
        default_time_run_started: The time to use if `time_run_started` is not defined.
            Do not use ``default_time_run_started=datetime.now()`` as it will be
            slightly different for each trial.  Instead set ``val = datetime.now()``
            and then pass ``val`` as the ``default_time_run_started`` argument.
    """
    return (
        trial.time_run_started
        if trial.time_run_started is not None
        else default_time_run_started
    )


def _time_trial_completed_safe(trial: BatchTrial) -> datetime:
    """Not all COMPLETED trials have `time_completed` defined.
    This functions accepts, but penalizes those trials by
    choosing epoch 0 as the completed time,
    which moves them to the end of the sort because
    they would be running a very short time."""
    return (
        trial.time_completed
        if trial.time_completed is not None
        else datetime.fromtimestamp(0)
    )

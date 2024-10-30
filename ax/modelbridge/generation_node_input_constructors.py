# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from enum import Enum, unique
from math import ceil, floor
from typing import Any

from ax.core import ObservationFeatures
from ax.core.base_trial import STATUSES_EXPECTING_DATA
from ax.core.experiment import Experiment
from ax.core.utils import get_target_trial_index
from ax.exceptions.generation_strategy import AxGenerationException

from ax.modelbridge.generation_node import GenerationNode
from ax.utils.common.constants import Keys
from ax.utils.common.func_enum import FuncEnum


@unique
class InputConstructorPurpose(Enum):
    """A simple enum to indicate the purpose of the input constructor.
    Values in this enum will match argument names for ``GenerationNode.gen``.

    Explanation of the different purposes:
        N: Defines the logic to determine the number of arms to generate from the
           next ``GenerationNode`` given the total number of arms expected in
           this trial.
    """

    N = "n"
    FIXED_FEATURES = "fixed_features"
    STATUS_QUO_FEATURES = "status_quo_features"


class NodeInputConstructors(FuncEnum):
    """An enum which maps to a the name of a callable method for constructing
    ``GenerationNode`` inputs. Recommendation: ends of the names of members
    of this enum should match the corresponding ``InputConstructorPurpose`` name.

    NOTE: All functions defined by this enum should share identical arguments in
    their signatures, and the return type should be the same across all functions
    that are used for the same ``InputConstructorPurpose``.
    """

    ALL_N = "consume_all_n"
    REPEAT_N = "repeat_arm_n"
    REMAINING_N = "remaining_n"
    TARGET_TRIAL_FIXED_FEATURES = "set_target_trial"
    STATUS_QUO_FEATURES = "get_status_quo"

    # pyre-ignore[3]: Input constructors will be used to make different inputs,
    # so we need to allow `Any` return type here.
    def __call__(
        self,
        previous_node: GenerationNode | None,
        next_node: GenerationNode,
        gs_gen_call_kwargs: dict[str, Any],
        experiment: Experiment,
    ) -> Any:
        """Defines a method, by which the members of this enum can be called,
        e.g. ``NodeInputConstructors.ALL_N(**kwargs)``, which will call the
        ``consume_all_n`` function from this file, since the name of this
        function corresponds to the value of the enum member ``ALL_N``."""
        return super().__call__(
            previous_node=previous_node,
            next_node=next_node,
            gs_gen_call_kwargs=gs_gen_call_kwargs,
            experiment=experiment,
        )


# ------------------------- Purpose: `fixed_features` ------------------------- #


def get_status_quo(
    previous_node: GenerationNode | None,
    next_node: GenerationNode,
    gs_gen_call_kwargs: dict[str, Any],
    experiment: Experiment,
) -> ObservationFeatures | None:
    """Get the status quo features to pass to the fit of the next node, if applicable.

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
        An ``ObservationFeatures`` object that defines the status quo observation
        features for fitting the model in the next node.
    """
    target_trial_idx = get_target_trial_index(experiment=experiment)
    if target_trial_idx is None:
        raise AxGenerationException(
            f"Attempting to construct status quo input into {next_node} but couldn't "
            "identify the target trial. Often this could be due to no trials on the "
            f"experiment that are in status {STATUSES_EXPECTING_DATA} on the "
            f"experiment. The trials on this experiment are: {experiment.trials}."
        )
    if experiment.status_quo is None:
        raise AxGenerationException(
            f"Attempting to construct status quo input into {next_node} but the "
            "experiment has no status quo. Please set a status quo before "
            "generating."
        )
    return ObservationFeatures(
        parameters=experiment.status_quo.parameters,
        trial_index=target_trial_idx,
    )


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
    target_trial_idx = get_target_trial_index(experiment=experiment)
    if target_trial_idx is None:
        raise AxGenerationException(
            f"Attempting to construct for input into {next_node} but no trials match "
            "the expected conditions. Often this could be due to no trials on the "
            f"experiment that are in status {STATUSES_EXPECTING_DATA} on the "
            f"experiment. The trials on this experiment are: {experiment.trials}."
        )
    return ObservationFeatures(
        parameters={},
        trial_index=target_trial_idx,
    )


# ------------------------- Purpose: `n` ------------------------- #


def consume_all_n(
    previous_node: GenerationNode | None,
    next_node: GenerationNode,
    gs_gen_call_kwargs: dict[str, Any],
    experiment: Experiment,
) -> int:
    """Generate total requested number of arms from the next node.

    Example: Initial exploration with Sobol will generate all arms from a
    single sobol node.

    Note: If no `n` is provided to the ``GenerationStrategy`` gen call, we will use
    the default number of arms for the next node, defined as a constant `DEFAULT_N`
    in the ``GenerationStrategy`` file.

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
    n_kwarg = gs_gen_call_kwargs.get("n")
    return (
        n_kwarg
        if n_kwarg is not None
        else _get_default_n(experiment=experiment, next_node=next_node)
    )


def repeat_arm_n(
    previous_node: GenerationNode | None,
    next_node: GenerationNode,
    gs_gen_call_kwargs: dict[str, Any],
    experiment: Experiment,
) -> int:
    """Generate a small percentage of arms requested to be used for repeat arms in
    the next trial.

    Note: If no `n` is provided to the ``GenerationStrategy`` gen call, we will use
    the default number of arms for the next node, defined as a constant `DEFAULT_N`
    in the ``GenerationStrategy`` file.

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
    n_kwarg = gs_gen_call_kwargs.get("n")
    total_n = (
        n_kwarg
        if n_kwarg is not None
        else _get_default_n(experiment=experiment, next_node=next_node)
    )
    if total_n < 6:
        # if the next trial is small, we don't want to waste allocation on repeat arms
        # users can still manually add repeat arms if they want before allocation
        # and we need to designated this node as skipped for proper transition
        next_node._should_skip = True
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

    Note: If no `n` is provided to the ``GenerationStrategy`` gen call, we will use
    the default number of arms for the next node, defined as a constant `DEFAULT_N`
    in the ``GenerationStrategy`` file.

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
    # TODO: @mgarrard improve this logic to be more robust
    grs_this_gen = gs_gen_call_kwargs.get("grs_this_gen", [])
    n_kwarg = gs_gen_call_kwargs.get("n")
    total_n = (
        n_kwarg
        if n_kwarg is not None
        else _get_default_n(experiment=experiment, next_node=next_node)
    )
    # if all arms have been generated, return 0
    return max(total_n - sum(len(gr.arms) for gr in grs_this_gen), 0)


# Helper methods for input constructors
def _get_default_n(experiment: Experiment, next_node: GenerationNode) -> int:
    """Get the default number of arms to generate from the next node.

    Args:
        experiment: The experiment associated with this ``GenerationStrategy``.
        next_node: The next node in the ``GenerationStrategy``. This is the node that
            will leverage the inputs defined by this input constructor.

    Returns:
        The default number of arms to generate from the next node, used if no n is
        provided to the ``GenerationStrategy``'s gen call.
    """
    n_for_this_trial = None
    total_concurrent_arms = experiment._properties.get(
        Keys.EXPERIMENT_TOTAL_CONCURRENT_ARMS.value
    )

    # If total_concurrent_arms is set, we will use that in conjunction with the trial
    # type to determine the number of arms to generate from this node
    if total_concurrent_arms is not None:
        if next_node._trial_type == Keys.SHORT_RUN:
            n_for_this_trial = floor(0.5 * total_concurrent_arms)
        elif next_node._trial_type == Keys.LONG_RUN:
            n_for_this_trial = ceil(0.5 * total_concurrent_arms)
        else:
            n_for_this_trial = total_concurrent_arms

    return (
        n_for_this_trial
        if n_for_this_trial is not None
        # GS default n is 1, but these input constructors are used for nodes that
        # should generate more than 1 arm per trial, default to 10
        else next_node.generation_strategy.DEFAULT_N * 10
    )

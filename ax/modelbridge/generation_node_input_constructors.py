# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import sys
from enum import Enum, unique
from math import ceil
from typing import Any

from ax.core import ObservationFeatures

from ax.modelbridge.generation_node import GenerationNode


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
    Returns:
        An ``ObservationFeatures`` object that defines the target trial for the next
        node.
    """

    # TODO: @mgarrard implement logic in follow-up diff
    return ObservationFeatures(
        parameters={},
        trial_index=0,
    )


def consume_all_n(
    previous_node: GenerationNode | None,
    next_node: GenerationNode,
    gs_gen_call_kwargs: dict[str, Any],
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

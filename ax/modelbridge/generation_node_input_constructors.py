# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from enum import Enum, unique
from typing import Any, Dict, Optional

from ax.modelbridge.generation_node import GenerationNode


def consume_all_n(
    previous_node: Optional[GenerationNode],
    next_node: GenerationNode,
    gs_gen_call_kwargs: Dict[str, Any],
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


@unique
class NodeInputConstructors(Enum):
    """An enum which maps to a callable method for constructing ``GenerationNode``
    inputs.
    """

    ALL_N = consume_all_n

    def __call__(
        self,
        previous_node: Optional[GenerationNode],
        next_node: GenerationNode,
        gs_gen_call_kwargs: Dict[str, Any],
    ) -> int:
        """Defines a callable method for the Enum as all values are methods"""
        return self(
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

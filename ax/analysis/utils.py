# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-safe

from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from pyre_extensions import none_throws


def extract_relevant_adapter(
    experiment: Experiment | None,
    generation_strategy: GenerationStrategy | None,
    adapter: Adapter | None,
) -> Adapter:
    """
    Analysis.compute(...) takes in both a GenerationStrategy and an Adapter as optional
    arguments. To keep consistency we always want use the same logic for extracting the
    relevant adapter across all Analysis classes.

    * If an Adapter is provided, return it.
    * If a GenerationStrategy is provided, return the Adapter from the current
        GenerationNode. Additionally, if it has not been fit, fit it using the
        Experiment.
    * If both are provided prefer the Adapter.
    * If neither are provided, raise an Exception.
    """

    if generation_strategy is None and adapter is None:
        raise UserInputError(
            "Must provide either a GenerationStrategy or an Adapter to compute this "
            "Analysis."
        )

    if adapter is not None:
        return adapter

    generation_strategy = none_throws(generation_strategy)

    if (model := generation_strategy.model) is not None:
        return model

    if experiment is None:
        raise UserInputError(
            "Provided GenerationStrategy has no model, but no Experiment was provided "
            "to source data to fit the model."
        )

    generation_strategy.current_node._fit(experiment=experiment)

    return none_throws(generation_strategy.model)

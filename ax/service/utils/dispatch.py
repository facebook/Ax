#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from ax.core.parameter import ChoiceParameter, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.utils.common.logger import get_logger


logger: logging.Logger = get_logger(__name__)


def choose_generation_strategy(search_space: SearchSpace) -> GenerationStrategy:
    """Select an appropriate generation strategy based on the properties of
    the search space."""
    num_continuous_parameters, num_discrete_choices = 0, 0
    for parameter in search_space.parameters:
        if isinstance(parameter, ChoiceParameter):
            num_discrete_choices += len(parameter.values)
        if isinstance(parameter, RangeParameter):
            num_continuous_parameters += 1
    # If there are more discrete choices than continuous parameters, Sobol
    # will do better than GP+EI.
    if num_continuous_parameters >= num_discrete_choices:
        sobol_arms = max(5, len(search_space.parameters))
        logger.info(
            "Using Bayesian Optimization generation strategy. Iterations after "
            f"{sobol_arms} will take longer to generate due to model-fitting."
        )
        return GenerationStrategy(
            name="Sobol+GPEI",
            steps=[
                GenerationStep(
                    model=Models.SOBOL, num_arms=sobol_arms, min_arms_observed=5
                ),
                GenerationStep(model=Models.GPEI, num_arms=-1),
            ],
        )
    else:
        logger.info(f"Using Sobol generation strategy.")
        return GenerationStrategy(
            name="Sobol", steps=[GenerationStep(model=Models.SOBOL, num_arms=-1)]
        )

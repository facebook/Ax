#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
from math import ceil
from typing import Optional

from ax.core.parameter import ChoiceParameter, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.utils.common.logger import get_logger


logger: logging.Logger = get_logger(__name__)


def choose_generation_strategy(
    search_space: SearchSpace,
    arms_per_trial: int = 1,
    enforce_sequential_optimization: bool = True,
    random_seed: Optional[int] = None,
) -> GenerationStrategy:
    """Select an appropriate generation strategy based on the properties of
    the search space."""
    model_kwargs = {"seed": random_seed} if (random_seed is not None) else None
    num_continuous_parameters, num_discrete_choices = 0, 0
    for parameter in search_space.parameters.values():
        if isinstance(parameter, ChoiceParameter):
            num_discrete_choices += len(parameter.values)
        if isinstance(parameter, RangeParameter):
            num_continuous_parameters += 1
    # If there are more discrete choices than continuous parameters, Sobol
    # will do better than GP+EI.
    if num_continuous_parameters >= num_discrete_choices:
        # Ensure that number of arms per model is divisible by batch size.
        sobol_arms = (
            ceil(max(5, len(search_space.parameters)) / arms_per_trial) * arms_per_trial
        )
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_arms=sobol_arms,
                    min_arms_observed=ceil(sobol_arms / 2),
                    enforce_num_arms=enforce_sequential_optimization,
                    model_kwargs=model_kwargs,
                ),
                GenerationStep(
                    model=Models.GPEI, num_arms=-1, recommended_max_parallelism=3
                ),
            ]
        )
        logger.info(
            f"Using Bayesian Optimization generation strategy: {gs}. Iterations "
            f"after {sobol_arms} will take longer to generate due to model-fitting."
        )
        return gs
    else:
        logger.info(f"Using Sobol generation strategy.")
        return GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL, num_arms=-1, model_kwargs=model_kwargs
                )
            ]
        )

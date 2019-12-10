#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from math import ceil
from typing import Optional, Tuple, Type, cast

from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Cont_X_trans, Models, Y_trans
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.winsorize import Winsorize
from ax.utils.common.logger import get_logger


logger: logging.Logger = get_logger(__name__)


def _make_sobol_step(
    num_arms: int = -1,
    min_arms_observed: Optional[int] = None,
    enforce_num_arms: bool = True,
    recommended_max_parallelism: Optional[int] = None,
    seed: Optional[int] = None,
) -> GenerationStep:
    """Shortcut for creating a Sobol generation step."""
    return GenerationStep(
        model=Models.SOBOL,
        num_arms=num_arms,
        # NOTE: ceil(-1 / 2) = 0, so this is safe to do when num arms is -1.
        min_arms_observed=min_arms_observed or ceil(num_arms / 2),
        enforce_num_arms=enforce_num_arms,
        recommended_max_parallelism=recommended_max_parallelism,
        model_kwargs={"deduplicate": True, "seed": seed},
    )


def _make_botorch_step(
    num_arms: int = -1,
    min_arms_observed: Optional[int] = None,
    enforce_num_arms: bool = True,
    recommended_max_parallelism: Optional[int] = None,
    winsorize: bool = False,
    winsorization_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> GenerationStep:
    """Shortcut for creating a BayesOpt generation step."""
    if (winsorize and winsorization_limits is None) or (
        winsorization_limits is not None and not winsorize
    ):
        raise ValueError(  # pragma: no cover
            "To apply winsorization, specify `winsorize=True` and provide the "
            "winsorization limits."
        )
    model_kwargs = None
    if winsorize:
        assert winsorization_limits is not None
        model_kwargs = {
            "transforms": [cast(Type[Transform], Winsorize)] + Cont_X_trans + Y_trans,
            "transform_configs": {
                "Winsorize": {
                    "winsorization_lower": winsorization_limits[0],
                    "winsorization_upper": winsorization_limits[1],
                }
            },
        }
    return GenerationStep(
        model=Models.GPEI,
        num_arms=num_arms,
        # NOTE: ceil(-1 / 2) = 0, so this is safe to do when num arms is -1.
        min_arms_observed=min_arms_observed or ceil(num_arms / 2),
        enforce_num_arms=enforce_num_arms,
        recommended_max_parallelism=recommended_max_parallelism,
        model_kwargs=model_kwargs,
    )


def _should_use_gp(search_space: SearchSpace, num_trials: Optional[int] = None) -> bool:
    """We should use only Sobol and not GPEI if:
    1. there are less continuous parameters in the search space than the sum of
    options for the choice parameters,
    2. the number of total iterations in the optimization is known in advance and
    there are less distinct points in the search space than the known intended
    number of total iterations.
    """
    num_continuous_parameters, num_discrete_choices, num_possible_points = 0, 0, 1
    all_range_parameters_are_int = True
    for parameter in search_space.parameters.values():
        if isinstance(parameter, ChoiceParameter):
            num_discrete_choices += len(parameter.values)
            num_possible_points *= len(parameter.values)
        if isinstance(parameter, RangeParameter):
            num_continuous_parameters += 1
            if parameter.parameter_type != ParameterType.INT:
                all_range_parameters_are_int = False
            else:
                num_possible_points *= int(parameter.upper - parameter.lower)

    if (  # If number of trials is known and it enough to try all possible points,
        num_trials is not None  # we should use Sobol and not BO.
        and all_range_parameters_are_int
        and num_possible_points <= num_trials
    ):
        return False

    return num_continuous_parameters >= num_discrete_choices


def choose_generation_strategy(
    search_space: SearchSpace,
    arms_per_trial: int = 1,
    enforce_sequential_optimization: bool = True,
    random_seed: Optional[int] = None,
    winsorize_botorch_model: bool = False,
    winsorization_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
    no_bayesian_optimization: bool = False,
    num_trials: Optional[int] = None,
) -> GenerationStrategy:
    """Select an appropriate generation strategy based on the properties of
    the search space.

    Args:
        search_space: SearchSpace, based on the properties of which to select the
            generation strategy.
        arms_per_trial: If a trial is batched, how many arms will be in each batch.
            Defaults to 1, which corresponds to a regular, non-batched, `Trial`.
        enforce_sequential_optimization: Whether to enforce that the generation
            strategy needs to be updated with `min_arms_observed` observations for
            a given generation step before proceeding to the next one.
        random_seed: Fixed random seed for the Sobol generator.
        winsorize_botorch_model: Whether to apply the winsorization transform
            prior to applying other transforms for fitting the BoTorch model.
        winsorization_limits: Bounds for winsorization, if winsorizing, expressed
            as percentile. Usually only the upper winsorization trim is used when
            minimizing, and only the lower when maximizing.
        no_bayesian_optimization: If True, Bayesian optimization generation
            strategy will not be suggested and quasi-random strategy will be used.
        num_trials: Total number of trials in the optimization, if
            known in advance.
    """
    # If there are more discrete choices than continuous parameters, Sobol
    # will do better than GP+EI.
    if not no_bayesian_optimization and _should_use_gp(
        search_space=search_space, num_trials=num_trials
    ):
        # Ensure that number of arms per model is divisible by batch size.
        sobol_arms = max(5, len(search_space.parameters))
        if arms_per_trial != 1:  # pragma: no cover
            # If using batches, ensure that initialization sample is divisible by
            # the batch size.
            sobol_arms = ceil(sobol_arms / arms_per_trial) * arms_per_trial
        gs = GenerationStrategy(
            steps=[
                _make_sobol_step(
                    num_arms=sobol_arms,
                    enforce_num_arms=enforce_sequential_optimization,
                    seed=random_seed,
                ),
                _make_botorch_step(
                    recommended_max_parallelism=3,
                    winsorize=winsorize_botorch_model,
                    winsorization_limits=winsorization_limits,
                ),
            ]
        )
        logger.info(
            f"Using Bayesian Optimization generation strategy: {gs}. Iterations "
            f"after {sobol_arms} will take longer to generate due to model-fitting."
        )
        return gs

    logger.info(f"Using Sobol generation strategy.")
    return GenerationStrategy(steps=[_make_sobol_step(seed=random_seed)])

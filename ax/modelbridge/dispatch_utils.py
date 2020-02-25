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


DEFAULT_BAYESIAN_PARALLELISM = 3


def _make_sobol_step(
    num_trials: int = -1,
    min_trials_observed: Optional[int] = None,
    enforce_num_trials: bool = True,
    max_parallelism: Optional[int] = None,
    seed: Optional[int] = None,
) -> GenerationStep:
    """Shortcut for creating a Sobol generation step."""
    return GenerationStep(
        model=Models.SOBOL,
        num_trials=num_trials,
        # NOTE: ceil(-1 / 2) = 0, so this is safe to do when num trials is -1.
        min_trials_observed=min_trials_observed or ceil(num_trials / 2),
        enforce_num_trials=enforce_num_trials,
        max_parallelism=max_parallelism,
        model_kwargs={"deduplicate": True, "seed": seed},
    )


def _make_botorch_step(
    num_trials: int = -1,
    min_trials_observed: Optional[int] = None,
    enforce_num_trials: bool = True,
    max_parallelism: Optional[int] = None,
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
        num_trials=num_trials,
        # NOTE: ceil(-1 / 2) = 0, so this is safe to do when num trials is -1.
        min_trials_observed=min_trials_observed or ceil(num_trials / 2),
        enforce_num_trials=enforce_num_trials,
        max_parallelism=max_parallelism,
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
    use_batch_trials: bool = False,
    enforce_sequential_optimization: bool = True,
    random_seed: Optional[int] = None,
    winsorize_botorch_model: bool = False,
    winsorization_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
    no_bayesian_optimization: bool = False,
    num_trials: Optional[int] = None,
    num_initialization_trials: Optional[int] = None,
    no_max_parallelism: bool = False,
    max_parallelism_cap: Optional[int] = None,
) -> GenerationStrategy:
    """Select an appropriate generation strategy based on the properties of
    the search space and expected settings of the experiment, such as number of
    arms per trial, optimization algorithm settings, expected number of trials
    in the experiment, etc.

    Args:
        search_space: SearchSpace, based on the properties of which to select the
            generation strategy.
        use_batch_trials: Whether this generation strategy will be used to generate
            batched trials instead of 1-arm trials.
        enforce_sequential_optimization: Whether to enforce that the generation
            strategy needs to be updated with `min_trials_observed` observations for
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
        num_initialization_trials: Specific number of initialization trials, if wanted.
            Typically, initialization trials are generated quasi-randomly.
        no_max_parallelism: If True, no limit on parallelism will be imposed. Be aware
            that parallelism is limited to improve performance of Bayesian optimization,
            so only disable its limiting if there is a good reason to do so.
        max_parallelism_cap: Integer representing a cap on parallelism in this gen.
            strategy; if specified, generation strategy will not generate trials if
            more than `max_parallelism_cap` trials for current generation step are
            running. Note that less than `max_parallelism_cap` may be scheduled in
            parallel if beneficial for Bayesian optimization performance;
            `max_parallelism_cap` is meant to just be a hard limit on parallelism (e.g.,
            to avoid overloading machine(s) that evaluate the experiment trials).
    """
    # If there are more discrete choices than continuous parameters, Sobol
    # will do better than GP+EI.
    if not no_bayesian_optimization and _should_use_gp(
        search_space=search_space, num_trials=num_trials
    ):
        # If number of initialization trials is not specified, estimate it.
        if num_initialization_trials is None:
            if use_batch_trials:  # Batched trials.
                num_initialization_trials = 1
            else:  # 1-arm trials.
                num_initialization_trials = max(5, len(search_space.parameters))
        if no_max_parallelism:
            bo_parallelism = None
        elif max_parallelism_cap is None:
            bo_parallelism = DEFAULT_BAYESIAN_PARALLELISM
        else:
            bo_parallelism = min(max_parallelism_cap, DEFAULT_BAYESIAN_PARALLELISM)
        gs = GenerationStrategy(
            steps=[
                _make_sobol_step(
                    num_trials=num_initialization_trials,
                    enforce_num_trials=enforce_sequential_optimization,
                    seed=random_seed,
                    max_parallelism=max_parallelism_cap,
                ),
                _make_botorch_step(
                    winsorize=winsorize_botorch_model,
                    winsorization_limits=winsorization_limits,
                    max_parallelism=bo_parallelism,
                ),
            ]
        )
        logger.info(
            f"Using Bayesian Optimization generation strategy: {gs}. Iterations after"
            f" {num_initialization_trials} will take longer to generate due to "
            " model-fitting."
        )
        return gs

    logger.info(f"Using Sobol generation strategy.")
    return GenerationStrategy(steps=[_make_sobol_step(seed=random_seed)])

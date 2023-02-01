#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
from math import ceil
from typing import Any, cast, Dict, Optional, Type, Union

import torch
from ax.core.experiment import Experiment
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Cont_X_trans, Models, Y_trans
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.winsorize import Winsorize
from ax.models.types import TConfig
from ax.models.winsorization_config import WinsorizationConfig
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none


logger: logging.Logger = get_logger(__name__)


DEFAULT_BAYESIAN_PARALLELISM = 3
# `BO_MIXED` optimizes all range parameters once for each combination of choice
# parameters, then takes the optimum of those optima. The cost associated with this
# method grows with the number of combinations, and so it is only used when the
# number of enumerated discrete combinations is below some maximum value.
MAX_DISCRETE_ENUMERATIONS_MIXED = 65
MAX_DISCRETE_ENUMERATIONS_NO_CONTINUOUS_OPTIMIZATION = 1e4
SAASBO_INCOMPATIBLE_MESSAGE = (
    "SAASBO is incompatible with {} generation strategy. "
    "Disregarding user input `use_saasbo = True`."
)


def _make_sobol_step(
    num_trials: int = -1,
    min_trials_observed: Optional[int] = None,
    enforce_num_trials: bool = True,
    max_parallelism: Optional[int] = None,
    seed: Optional[int] = None,
    should_deduplicate: bool = False,
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
        should_deduplicate=should_deduplicate,
    )


def _make_botorch_step(
    num_trials: int = -1,
    min_trials_observed: Optional[int] = None,
    enforce_num_trials: bool = True,
    max_parallelism: Optional[int] = None,
    model: Models = Models.GPEI,
    model_kwargs: Optional[Dict[str, Any]] = None,
    winsorization_config: Optional[
        Union[WinsorizationConfig, Dict[str, WinsorizationConfig]]
    ] = None,
    no_winsorization: bool = False,
    should_deduplicate: bool = False,
    verbose: Optional[bool] = None,
    disable_progbar: Optional[bool] = None,
    derelativize_with_raw_status_quo: bool = False,
    use_update: Optional[bool] = None,
) -> GenerationStep:
    """Shortcut for creating a BayesOpt generation step."""

    winsorization_transform_config = _get_winsorization_transform_config(
        winsorization_config=winsorization_config,
        no_winsorization=no_winsorization,
        derelativize_with_raw_status_quo=derelativize_with_raw_status_quo,
    )

    derelativization_transform_config = {
        "use_raw_status_quo": derelativize_with_raw_status_quo
    }

    model_kwargs = model_kwargs or {}
    if not no_winsorization:
        transforms = [cast(Type[Transform], Winsorize)] + Cont_X_trans + Y_trans
        model_kwargs.update({"transforms": transforms})
        if winsorization_transform_config is not None:
            transform_configs = {
                "Winsorize": winsorization_transform_config,
                "Derelativize": derelativization_transform_config,
            }
            model_kwargs.update({"transform_configs": transform_configs})

    if verbose is not None:
        model_kwargs.update({"verbose": verbose})
    if disable_progbar is not None:
        model_kwargs.update({"disable_progbar": disable_progbar})
    return GenerationStep(
        model=model,
        num_trials=num_trials,
        # NOTE: ceil(-1 / 2) = 0, so this is safe to do when num trials is -1.
        min_trials_observed=min_trials_observed or ceil(num_trials / 2),
        enforce_num_trials=enforce_num_trials,
        max_parallelism=max_parallelism,
        use_update=use_update if use_update is not None else is_saasbo(model),
        # `model_kwargs` should default to `None` if empty
        model_kwargs=model_kwargs if len(model_kwargs) > 0 else None,
        should_deduplicate=should_deduplicate,
    )


def _suggest_gp_model(
    search_space: SearchSpace,
    num_trials: Optional[int] = None,
    optimization_config: Optional[OptimizationConfig] = None,
    use_saasbo: bool = False,
) -> Union[None, Models]:
    """Suggest a model based on the search space. None means we use Sobol.

    1. We use Sobol if the number of total iterations in the optimization is
    known in advance and there are fewer distinct points in the search space
    than the known intended number of total iterations.
    2. We use ``BO_MIXED`` if there are fewer ordered parameters in the search space
    than the sum of options for the *unordered* choice parameters, and the number
    of discrete enumerations to be performed by the optimizer is less than
    ``MAX_DISCRETE_ENUMERATIONS_MIXED``, or if there are only choice parameters and
    the number of choice combinations to enumerate is less than
    ``MAX_DISCRETE_ENUMERATIONS_CHOICE_ONLY``. ``BO_MIXED`` is not currently enabled
    for multi-objective optimization. Note that we do not count 2-level choice
    parameters as unordered, since these do not affect the modeling choice.
    3. We use ``MOO`` if ``optimization_config`` has multiple objectives and
    ``use_saasbo is False``.
    4. We use ``FULLYBAYESIANMOO`` if ``optimization_config`` has multiple objectives
    and ``use_saasbo is True``.
    5. If none of the above and ``use_saasbo is False``, we use ``GPEI``.
    6. If none of the above and ``use_saasbo is True``, we use ``FULLYBAYESIAN``.
    """
    # Count tunable parameter types.
    num_ordered_parameters = num_unordered_choices = 0
    num_enumerated_combinations = num_possible_points = 1
    all_range_parameters_are_discrete = True
    all_parameters_are_enumerated = True
    for parameter in search_space.tunable_parameters.values():
        should_enumerate_param = False
        num_param_discrete_values = None
        if isinstance(parameter, ChoiceParameter):
            should_enumerate_param = True
            num_param_discrete_values = len(parameter.values)
            num_possible_points *= num_param_discrete_values
            if parameter.is_ordered is False and num_param_discrete_values > 2:
                num_unordered_choices += num_param_discrete_values
            else:
                num_ordered_parameters += 1
        elif isinstance(parameter, RangeParameter):
            num_ordered_parameters += 1
            if parameter.parameter_type == ParameterType.FLOAT:
                all_range_parameters_are_discrete = False
            else:
                num_param_discrete_values = int(parameter.upper - parameter.lower) + 1
                num_possible_points *= num_param_discrete_values

        if should_enumerate_param:
            num_enumerated_combinations *= not_none(num_param_discrete_values)
        else:
            all_parameters_are_enumerated = False

    # Use Sobol if number of trials is known and sufficient to try all possible points.
    if (
        num_trials is not None
        and all_range_parameters_are_discrete
        and num_possible_points <= num_trials
    ):
        logger.info("Using Sobol since we can enumerate the search space.")
        if use_saasbo:
            logger.warning(SAASBO_INCOMPATIBLE_MESSAGE.format("Sobol"))
        return None

    is_moo_problem = optimization_config and optimization_config.is_moo_problem
    if num_ordered_parameters > num_unordered_choices:
        logger.info(
            "Using Bayesian optimization since there are more ordered parameters than "
            "there are categories for the unordered categorical parameters."
        )
        if is_moo_problem:
            return Models.FULLYBAYESIANMOO if use_saasbo else Models.MOO
        return Models.FULLYBAYESIAN if use_saasbo else Models.GPEI

    # Use mixed Bayesian optimization when appropriate. This logic is currently tied to
    # the fact that acquisition function optimization for mixed BayesOpt currently
    # enumerates all combinations of choice parameters.
    if num_enumerated_combinations <= MAX_DISCRETE_ENUMERATIONS_MIXED or (
        all_parameters_are_enumerated
        and num_enumerated_combinations
        < MAX_DISCRETE_ENUMERATIONS_NO_CONTINUOUS_OPTIMIZATION
    ):
        logger.info(
            "Using Bayesian optimization with a categorical kernel for improved "
            "performance with a large number of unordered categorical parameters."
        )
        if use_saasbo:
            logger.warning(SAASBO_INCOMPATIBLE_MESSAGE.format("`BO_MIXED`"))
        return Models.BO_MIXED

    logger.info(
        f"Using Sobol since there are more than {MAX_DISCRETE_ENUMERATIONS_MIXED} "
        "combinations of enumerated parameters. For improved performance, make sure "
        "that all ordered `ChoiceParameter`s are encoded as such (`is_ordered=True`), "
        "and use `RangeParameter`s in place of ordered `ChoiceParameter`s where "
        "possible. Also, consider removing some or all unordered `ChoiceParameter`s."
    )
    if use_saasbo:
        logger.warning(SAASBO_INCOMPATIBLE_MESSAGE.format("Sobol"))
    return None


def calculate_num_initialization_trials(
    num_tunable_parameters: int,
    num_trials: Optional[int],
    use_batch_trials: bool,
) -> int:
    """
    Applies rules from high to low priority
     - 1 for batch trials.
     - At least 5
     - At most 1/5th of num_trials.
     - Twice the number of tunable parameters
    """
    if use_batch_trials:  # Batched trials.
        return 1

    ret = 2 * num_tunable_parameters
    if num_trials is not None:
        ret = min(ret, not_none(num_trials) // 5)
    return max(ret, 5)


def choose_generation_strategy(
    search_space: SearchSpace,
    *,
    use_batch_trials: bool = False,
    enforce_sequential_optimization: bool = True,
    random_seed: Optional[int] = None,
    torch_device: Optional[torch.device] = None,
    no_winsorization: bool = False,
    winsorization_config: Optional[
        Union[WinsorizationConfig, Dict[str, WinsorizationConfig]]
    ] = None,
    derelativize_with_raw_status_quo: bool = False,
    no_bayesian_optimization: bool = False,
    num_trials: Optional[int] = None,
    num_initialization_trials: Optional[int] = None,
    num_completed_initialization_trials: int = 0,
    max_initialization_trials: Optional[int] = None,
    max_parallelism_cap: Optional[int] = None,
    max_parallelism_override: Optional[int] = None,
    optimization_config: Optional[OptimizationConfig] = None,
    should_deduplicate: bool = False,
    use_saasbo: bool = False,
    verbose: Optional[bool] = None,
    disable_progbar: Optional[bool] = None,
    experiment: Optional[Experiment] = None,
    use_update: Optional[bool] = None,
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
        enforce_sequential_optimization: Whether to enforce that 1) the generation
            strategy needs to be updated with ``min_trials_observed`` observations for
            a given generation step before proceeding to the next one and 2) maximum
            number of trials running at once (max_parallelism) if enforced for the
            BayesOpt step. NOTE: ``max_parallelism_override`` and
            ``max_parallelism_cap`` settings will still take their effect on max
            parallelism even if ``enforce_sequential_optimization=False``, so if those
            settings are specified, max parallelism will be enforced.
        random_seed: Fixed random seed for the Sobol generator.
        torch_device: The device to use for generation steps implemented in PyTorch
            (e.g. via BoTorch). Some generation steps (in particular EHVI-based ones
            for multi-objective optimization) can be sped up by running candidate
            generation on the GPU. If not specified, uses the default torch device
            (usually the CPU).
        no_winsorization: Whether to apply the winsorization transform
            prior to applying other transforms for fitting the BoTorch model.
        winsorization_config: Explicit winsorization settings, if winsorizing. Usually
            only `upper_quantile_margin` is set when minimizing, and only
            `lower_quantile_margin` when maximizing.
        derelativize_with_raw_status_quo: Whether to derelativize using the raw status
            quo values in any transforms. This argument is primarily to allow automatic
            Winsorization when relative constraints are present. Note: automatic
            Winsorization will fail if this is set to `False` (or unset) and there
            are relative constraints present.
        no_bayesian_optimization: If True, Bayesian optimization generation
            strategy will not be suggested and quasi-random strategy will be used.
        num_trials: Total number of trials in the optimization, if
            known in advance.
        num_initialization_trials: Specific number of initialization trials, if wanted.
            Typically, initialization trials are generated quasi-randomly.
        max_initialization_trials: If ``num_initialization_trials`` unspecified, it
            will be determined automatically. This arg provides a cap on that
            automatically determined number.
        num_completed_initialization_trials: The final calculated number of
            initialization trials is reduced by this number. This is useful when
            warm-starting an experiment, to specify what number of completed trials
            can be used to satisfy the initialization_trial requirement.
        max_parallelism_cap: Integer cap on parallelism in this generation strategy.
            If specified, ``max_parallelism`` setting in each generation step will be
            set to the minimum of the default setting for that step and the value of
            this cap. ``max_parallelism_cap`` is meant to just be a hard limit on
            parallelism (e.g. to avoid overloading machine(s) that evaluate the
            experiment trials). Specify only if not specifying
            ``max_parallelism_override``.
        max_parallelism_override: Integer, with which to override the default max
            parallelism setting for all steps in the generation strategy returned from
            this function. Each generation step has a ``max_parallelism`` value, which
            restricts how many trials can run simultaneously during a given generation
            step. By default, the parallelism setting is chosen as appropriate for the
            model in a given generation step. If ``max_parallelism_override`` is -1,
            no max parallelism will be enforced for any step of the generation
            strategy. Be aware that parallelism is limited to improve performance of
            Bayesian optimization, so only disable its limiting if necessary.
        optimization_config: used to infer whether to use MOO and will be passed in to
            ``Winsorize`` via its ``transform_config`` in order to determine default
            winsorization behavior when necessary.
        should_deduplicate: Whether to deduplicate the parameters of proposed arms
            against those of previous arms via rejection sampling. If this is True,
            the generation strategy will discard generator runs produced from the
            generation step that has `should_deduplicate=True` if they contain arms
            already present on the experiment and replace them with new generator runs.
            If no generator run with entirely unique arms could be produced in 5
            attempts, a `GenerationStrategyRepeatedPoints` error will be raised, as we
            assume that the optimization converged when the model can no longer suggest
            unique arms.
        use_saasbo: Whether to use SAAS prior for any GPEI generation steps.
        verbose: Whether GP model should produce verbose logs. If not ``None``, its
            value gets added to ``model_kwargs`` during ``generation_strategy``
            construction. Defaults to ``True`` for SAASBO, else ``None``. Verbose
            outputs are currently only available for SAASBO, so if ``verbose is not
            None`` for a different model type, it will be overridden to ``None`` with
            a warning.
        disable_progbar: Whether GP model should produce a progress bar. If not
            ``None``, its value gets added to ``model_kwargs`` during
            ``generation_strategy`` construction. Defaults to ``True`` for SAASBO, else
            ``None``. Progress bars are currently only available for SAASBO, so if
            ``disable_probar is not None`` for a different model type, it will be
            overridden to ``None`` with a warning.
        experiment: If specified, ``_experiment`` attribute of the generation strategy
            will be set to this experiment (useful for associating a generation
            strategy with a given experiment before it's first used to ``gen`` with
            that experiment). Can also provide `optimization_config` if it is not
            provided as an arg to this function.
        use_update: Whether to use ``ModelBridge.update`` to update the model with
            new data rather than fitting it from scratch. This is much more efficient,
            particularly when running trials in parallel. Note that this is not
            compatible with metrics that are available while running.
            It will default to True if using SAASBO and the given experiment does not
            have any metrics that are available while running.
    """
    if experiment is not None:
        if optimization_config is None:
            optimization_config = experiment.optimization_config
        metrics_available_while_running = any(
            m.is_available_while_running() for m in experiment.metrics.values()
        )
        if metrics_available_while_running:
            if use_update is True:
                raise UnsupportedError(
                    "Got `use_update=True` but the experiment has metrics that are "
                    "available while running. Set `use_update=False`."
                )
            else:
                use_update = False

    suggested_model = _suggest_gp_model(
        search_space=search_space,
        num_trials=num_trials,
        optimization_config=optimization_config,
        use_saasbo=use_saasbo,
    )
    # Determine max parallelism for the generation steps.
    if max_parallelism_override == -1:
        # `max_parallelism_override` of -1 means no max parallelism enforcement in
        # the generation strategy, which means `max_parallelism=None` in gen. steps.
        sobol_parallelism = bo_parallelism = None
    elif max_parallelism_override is not None:
        sobol_parallelism = bo_parallelism = max_parallelism_override
    elif max_parallelism_cap is not None:  # Max parallelism override is None by now
        sobol_parallelism = max_parallelism_cap
        bo_parallelism = min(max_parallelism_cap, DEFAULT_BAYESIAN_PARALLELISM)
    elif not enforce_sequential_optimization:
        # If no max parallelism settings specified and not enforcing sequential
        # optimization, do not limit parallelism.
        sobol_parallelism = bo_parallelism = None
    else:  # No additional max parallelism settings, use defaults
        sobol_parallelism = None  # No restriction on Sobol phase
        bo_parallelism = DEFAULT_BAYESIAN_PARALLELISM

    if not no_bayesian_optimization and suggested_model is not None:
        if not enforce_sequential_optimization and (  # pragma: no cover
            max_parallelism_override or max_parallelism_cap
        ):
            logger.info(
                "If `enforce_sequential_optimization` is False, max parallelism is "
                "not enforced and other max parallelism settings will be ignored."
            )
        if max_parallelism_override and max_parallelism_cap:
            raise ValueError(
                "If `max_parallelism_override` specified, cannot also apply "
                "`max_parallelism_cap`."
            )

        # If number of initialization trials is not specified, estimate it.
        logger.info(
            "Calculating the number of remaining initialization trials based on "
            f"num_initialization_trials={num_initialization_trials} "
            f"max_initialization_trials={max_initialization_trials} "
            f"num_tunable_parameters={len(search_space.tunable_parameters)} "
            f"num_trials={num_trials} "
            f"use_batch_trials={use_batch_trials}"
        )
        if num_initialization_trials is None:
            num_initialization_trials = calculate_num_initialization_trials(
                num_tunable_parameters=len(search_space.tunable_parameters),
                num_trials=num_trials,
                use_batch_trials=use_batch_trials,
            )
            if max_initialization_trials is not None:
                num_initialization_trials = min(
                    num_initialization_trials, max_initialization_trials
                )
            logger.info(
                f"calculated num_initialization_trials={num_initialization_trials}"
            )
        num_remaining_initialization_trials = max(
            0, num_initialization_trials - max(0, num_completed_initialization_trials)
        )
        logger.info(
            "num_completed_initialization_trials="
            f"{num_completed_initialization_trials} "
            f"num_remaining_initialization_trials={num_remaining_initialization_trials}"
        )
        steps = []
        # `verbose` and `disable_progbar` defaults and overrides
        model_is_saasbo = is_saasbo(suggested_model)
        if verbose is None and model_is_saasbo:
            verbose = True
        elif verbose is not None and not model_is_saasbo:
            logger.warning(
                f"Overriding `verbose = {verbose}` to `None` for non-SAASBO GP step."
            )
            verbose = None
        if disable_progbar is not None and not model_is_saasbo:
            logger.warning(
                f"Overriding `disable_progbar = {disable_progbar}` to `None` for "
                "non-SAASBO GP step."
            )
            disable_progbar = None

        # Create `generation_strategy`, adding first Sobol step
        # if `num_remaining_initialization_trials` is > 0.
        if num_remaining_initialization_trials > 0:
            steps.append(
                _make_sobol_step(
                    num_trials=num_remaining_initialization_trials,
                    enforce_num_trials=enforce_sequential_optimization,
                    seed=random_seed,
                    max_parallelism=sobol_parallelism,
                    should_deduplicate=should_deduplicate,
                )
            )
        steps.append(
            _make_botorch_step(
                model=suggested_model,
                winsorization_config=winsorization_config,
                derelativize_with_raw_status_quo=derelativize_with_raw_status_quo,
                no_winsorization=no_winsorization,
                max_parallelism=bo_parallelism,
                model_kwargs={"torch_device": torch_device},
                should_deduplicate=should_deduplicate,
                verbose=verbose,
                disable_progbar=disable_progbar,
                use_update=use_update,
            ),
        )
        gs = GenerationStrategy(steps=steps)
        logger.info(
            f"Using Bayesian Optimization generation strategy: {gs}. Iterations after"
            f" {num_remaining_initialization_trials} will take longer to generate due"
            " to model-fitting."
        )
    else:  # `no_bayesian_optimization` is True or we could not suggest BO model
        if verbose is not None:
            logger.warning(
                f"Ignoring `verbose = {verbose}` for `generation_strategy` "
                "without a GP step."
            )

        gs = GenerationStrategy(
            steps=[
                _make_sobol_step(
                    seed=random_seed,
                    should_deduplicate=should_deduplicate,
                    max_parallelism=sobol_parallelism,
                )
            ]
        )
        logger.info("Using Sobol generation strategy.")
    if experiment:
        gs.experiment = experiment
    return gs


def _get_winsorization_transform_config(
    winsorization_config: Optional[
        Union[WinsorizationConfig, Dict[str, WinsorizationConfig]]
    ],
    derelativize_with_raw_status_quo: bool,
    no_winsorization: bool,
) -> Optional[TConfig]:
    if no_winsorization:
        if winsorization_config is not None:
            warnings.warn(
                "`no_winsorization = True` but `winsorization_config` has been set. "
                "Not winsorizing."
            )
        return None
    if winsorization_config:
        return {"winsorization_config": winsorization_config}
    return {"derelativize_with_raw_status_quo": derelativize_with_raw_status_quo}


def is_saasbo(model: Models) -> bool:
    return model.name in ["FULLYBAYESIANMOO", "FULLYBAYESIAN"]

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import warnings
from math import ceil
from typing import Any, cast

import torch
from ax.adapter.base import DataLoaderConfig
from ax.adapter.registry import GeneratorRegistryBase, Generators
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.winsorize import Winsorize
from ax.core.experiment import Experiment
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.generation_strategy.generation_strategy import (
    GenerationStep,
    GenerationStrategy,
)
from ax.generators.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from ax.generators.types import TConfig
from ax.generators.winsorization_config import WinsorizationConfig
from ax.utils.common.logger import get_logger
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from pyre_extensions import none_throws


logger: logging.Logger = get_logger(__name__)


DEFAULT_BAYESIAN_PARALLELISM = 3
# `BO_MIXED` optimizes all range parameters once for each combination of choice
# parameters, then takes the optimum of those optima. The cost associated with this
# method grows with the number of combinations, and so it is only used when the
# number of enumerated discrete combinations is below some maximum value.
MAX_DISCRETE_ENUMERATIONS_MIXED = 65
MAX_DISCRETE_ENUMERATIONS_NO_CONTINUOUS_OPTIMIZATION = 1e4
MAX_ONE_HOT_ENCODINGS_CONTINUOUS_OPTIMIZATION = 33
SAASBO_INCOMPATIBLE_MESSAGE = (
    "SAASBO is incompatible with {} generation strategy. "
    "Disregarding user input `use_saasbo = True`."
)


def _make_sobol_step(
    num_trials: int = -1,
    min_trials_observed: int | None = None,
    enforce_num_trials: bool = True,
    max_parallelism: int | None = None,
    seed: int | None = None,
    should_deduplicate: bool = False,
) -> GenerationStep:
    """Shortcut for creating a Sobol generation step."""
    return GenerationStep(
        generator=Generators.SOBOL,
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
    min_trials_observed: int | None = None,
    enforce_num_trials: bool = True,
    max_parallelism: int | None = None,
    generator: GeneratorRegistryBase = Generators.BOTORCH_MODULAR,
    model_kwargs: dict[str, Any] | None = None,
    winsorization_config: None
    | (WinsorizationConfig | dict[str, WinsorizationConfig]) = None,
    no_winsorization: bool = False,
    should_deduplicate: bool = False,
    disable_progbar: bool | None = None,
    jit_compile: bool | None = None,
    derelativize_with_raw_status_quo: bool = False,
    fit_out_of_design: bool = False,
    use_saasbo: bool = False,
    use_input_warping: bool = False,
) -> GenerationStep:
    """Shortcut for creating a BayesOpt generation step."""
    model_kwargs = model_kwargs or {}

    winsorization_transform_config = _get_winsorization_transform_config(
        winsorization_config=winsorization_config,
        no_winsorization=no_winsorization,
        derelativize_with_raw_status_quo=derelativize_with_raw_status_quo,
    )

    derelativization_transform_config = {
        "use_raw_status_quo": derelativize_with_raw_status_quo
    }
    model_kwargs["transform_configs"] = model_kwargs.get("transform_configs", {})
    model_kwargs["transform_configs"]["Derelativize"] = (
        derelativization_transform_config
    )
    model_kwargs["data_loader_config"] = DataLoaderConfig(
        fit_out_of_design=fit_out_of_design
    )

    if not no_winsorization:
        _, default_bridge_kwargs = generator.view_defaults()
        default_transforms = default_bridge_kwargs["transforms"]
        transforms = model_kwargs.get("transforms", default_transforms)
        model_kwargs["transforms"] = [cast(type[Transform], Winsorize)] + transforms
        if winsorization_transform_config is not None:
            model_kwargs["transform_configs"]["Winsorize"] = (
                winsorization_transform_config
            )

    if use_saasbo and (generator is Generators.BOTORCH_MODULAR):
        model_kwargs["surrogate_spec"] = SurrogateSpec(
            model_configs=[
                ModelConfig(
                    botorch_model_class=SaasFullyBayesianSingleTaskGP,
                    model_options={"use_input_warping": use_input_warping},
                    mll_options={
                        "disable_progbar": disable_progbar,
                        "jit_compile": jit_compile,
                    },
                    name=f"{'Warped ' if use_input_warping else ''}SAAS",
                )
            ]
        )
    elif disable_progbar is not None or jit_compile is not None:
        logger.info(
            "`disable_progbar`, and `jit_compile` are only supported with"
            " fully Bayesian models. These are being ignored."
        )
    return GenerationStep(
        generator=generator,
        num_trials=num_trials,
        # NOTE: ceil(-1 / 2) = 0, so this is safe to do when num trials is -1.
        min_trials_observed=min_trials_observed or ceil(num_trials / 2),
        enforce_num_trials=enforce_num_trials,
        max_parallelism=max_parallelism,
        model_kwargs=model_kwargs,
        should_deduplicate=should_deduplicate,
    )


def _suggest_gp_model(
    search_space: SearchSpace,
    num_trials: int | None = None,
    optimization_config: OptimizationConfig | None = None,
    use_saasbo: bool = False,
) -> None | GeneratorRegistryBase:
    """Suggest a model based on the search space. None means we use Sobol.

    1. We use Sobol if the number of total iterations in the optimization is
    known in advance and there are fewer distinct points in the search space
    than the known intended number of total iterations.
    2. We use ``BO_MIXED`` if there are fewer ordered parameters in the search space
    than the sum of options for the *unordered* choice parameters, and the number
    of discrete enumerations to be performed by the optimizer is less than
    ``MAX_DISCRETE_ENUMERATIONS_MIXED``, or if there are only choice parameters and
    the number of choice combinations to enumerate is less than
    ``MAX_DISCRETE_ENUMERATIONS_CHOICE_ONLY``. Note that we do not count 2-level choice
    parameters as unordered, since these do not affect the modeling choice.
    3. If there are more ordered parameters in the search space than the sum of options
    for the *unordered* choice parameters, or if there is at least one ordered
    parameter and the number of parameters needed to encode all unordered parameters
    is less than ``MAX_ONE_HOT_ENCODINGS_CONTINUOUS_OPTIMIZATION``, we use BO with
    continuous relaxation.
    * For BO, we use Modular BoTorch Model with ``SingleTaskGP`` if ``use_saasbo is
    False`` and with ``SaasFullyBayesianSingleTaskGP`` (aka SAASBO) otherwise.
    """
    # Count tunable parameter types.
    num_ordered_parameters = 0
    num_unordered_choices = 0
    num_enumerated_combinations = 1
    num_possible_points = 1
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
                num_param_discrete_values = parameter.cardinality()
                # pyre-fixme[58]: `*` is not supported for operand types `int` and
                #  `Union[float, int]`.
                num_possible_points *= num_param_discrete_values

        if should_enumerate_param:
            num_enumerated_combinations *= none_throws(num_param_discrete_values)
        else:
            all_parameters_are_enumerated = False

    # Use Sobol if number of trials is known and sufficient to try all possible points.
    if (
        num_trials is not None
        and all_range_parameters_are_discrete
        and num_possible_points <= num_trials
    ):
        logger.debug("Using Sobol since we can enumerate the search space.")
        if use_saasbo:
            logger.warning(SAASBO_INCOMPATIBLE_MESSAGE.format("Sobol"))
        return None

    # Use mixed Bayesian optimization when appropriate. This logic is currently tied to
    # the fact that acquisition function optimization for mixed BayesOpt currently
    # enumerates all combinations of choice parameters.
    # We use continuous relaxation if there are more ordered parameters than there
    # are choices for unordered parameters.
    if (num_ordered_parameters < num_unordered_choices) and (
        num_enumerated_combinations <= MAX_DISCRETE_ENUMERATIONS_MIXED
        or (
            all_parameters_are_enumerated
            and num_enumerated_combinations
            < MAX_DISCRETE_ENUMERATIONS_NO_CONTINUOUS_OPTIMIZATION
        )
    ):
        logger.debug(
            "Using Bayesian optimization with a categorical kernel for improved "
            "performance with a large number of unordered categorical parameters."
        )
        if use_saasbo:
            logger.warning(SAASBO_INCOMPATIBLE_MESSAGE.format("`BO_MIXED`"))
        return Generators.BO_MIXED

    if num_ordered_parameters >= num_unordered_choices or (
        num_unordered_choices < MAX_ONE_HOT_ENCODINGS_CONTINUOUS_OPTIMIZATION
        and num_ordered_parameters > 0
    ):
        # These use one-hot encoding for unordered choice parameters, resulting in a
        # total of num_unordered_choices OHE parameters.
        # So, we do not want to use them when there are too many unordered choices.
        method = Generators.BOTORCH_MODULAR
        reason = (
            (
                "there are more ordered parameters than there are categories for the "
                "unordered categorical parameters."
                if num_ordered_parameters >= num_unordered_choices
                else "there is at least one ordered parameter and there are fewer than "
                f"{MAX_ONE_HOT_ENCODINGS_CONTINUOUS_OPTIMIZATION} choices for "
                "unordered parameters."
            )
            if num_unordered_choices > 0
            else "there is at least one ordered parameter"
            " and there are no unordered categorical parameters."
        )
        logger.info(f"Using {method} since {reason}")
        return method

    logger.warning(
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
    num_trials: int | None,
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
        ret = min(ret, none_throws(num_trials) // 5)
    return max(ret, 5)


def choose_generation_strategy_legacy(
    search_space: SearchSpace,
    *,
    use_batch_trials: bool = False,
    enforce_sequential_optimization: bool = True,
    random_seed: int | None = None,
    torch_device: torch.device | None = None,
    no_winsorization: bool = False,
    winsorization_config: None
    | (WinsorizationConfig | dict[str, WinsorizationConfig]) = None,
    derelativize_with_raw_status_quo: bool = False,
    force_random_search: bool = False,
    num_trials: int | None = None,
    num_initialization_trials: int | None = None,
    num_completed_initialization_trials: int = 0,
    max_initialization_trials: int | None = None,
    min_sobol_trials_observed: int | None = None,
    max_parallelism_cap: int | None = None,
    max_parallelism_override: int | None = None,
    optimization_config: OptimizationConfig | None = None,
    should_deduplicate: bool = False,
    use_saasbo: bool = False,
    disable_progbar: bool | None = None,
    jit_compile: bool | None = None,
    experiment: Experiment | None = None,
    suggested_model_override: GeneratorRegistryBase | None = None,
    fit_out_of_design: bool = False,
    use_input_warping: bool = False,
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
        force_random_search: If True, quasi-random generation strategy will be used
            rather than Bayesian optimization.
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
        min_sobol_trials_observed: Minimum number of Sobol trials that must be
            observed before proceeding to the next generation step. Defaults to
            `ceil(num_initialization_trials / 2)`.
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
        disable_progbar: Whether GP model should produce a progress bar. If not
            ``None``, its value gets added to ``model_kwargs`` during
            ``generation_strategy`` construction. Defaults to ``True`` for SAASBO, else
            ``None``. Progress bars are currently only available for SAASBO, so if
            ``disable_probar is not None`` for a different model type, it will be
            overridden to ``None`` with a warning.
        jit_compile: Whether to use jit compilation in Pyro when SAASBO is used.
        experiment: If specified, ``_experiment`` attribute of the generation strategy
            will be set to this experiment (useful for associating a generation
            strategy with a given experiment before it's first used to ``gen`` with
            that experiment). Can also provide `optimization_config` if it is not
            provided as an arg to this function.
        suggested_model_override: If specified, this model will be used for the GP
            step and automatic selection will be skipped.
        fit_out_of_design: Whether to include out-of-design points in the model.
        use_input_warping: Whether to use input warping in the model. This is only
            supported in conjunction with use_saasbo=True.
    """
    if experiment is not None and optimization_config is None:
        optimization_config = experiment.optimization_config

    suggested_model = suggested_model_override or _suggest_gp_model(
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

    if not force_random_search and suggested_model is not None:
        if not enforce_sequential_optimization and (
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
        logger.debug(
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
            logger.debug(
                f"calculated num_initialization_trials={num_initialization_trials}"
            )
        num_remaining_initialization_trials = max(
            0, num_initialization_trials - max(0, num_completed_initialization_trials)
        )
        logger.debug(
            "num_completed_initialization_trials="
            f"{num_completed_initialization_trials} "
            f"num_remaining_initialization_trials={num_remaining_initialization_trials}"
        )
        steps = []
        # `disable_progbar` and jit_compile defaults and overrides
        model_is_saasbo = use_saasbo and (suggested_model is Generators.BOTORCH_MODULAR)
        if disable_progbar is not None and not model_is_saasbo:
            logger.warning(
                f"Overriding `disable_progbar = {disable_progbar}` to `None` for "
                "non-SAASBO GP step."
            )
            disable_progbar = None
        if jit_compile is not None and not model_is_saasbo:
            logger.warning(
                f"Overriding `jit_compile = {jit_compile}` to `None` for "
                "non-SAASBO GP step."
            )
            jit_compile = None

        model_kwargs: dict[str, Any] = {
            "torch_device": torch_device,
            "data_loader_config": DataLoaderConfig(
                fit_out_of_design=fit_out_of_design,
            ),
        }

        # Create `generation_strategy`, adding first Sobol step
        # if `num_remaining_initialization_trials` is > 0.
        if num_remaining_initialization_trials > 0:
            steps.append(
                _make_sobol_step(
                    num_trials=num_remaining_initialization_trials,
                    min_trials_observed=min_sobol_trials_observed,
                    enforce_num_trials=enforce_sequential_optimization,
                    seed=random_seed,
                    max_parallelism=sobol_parallelism,
                    should_deduplicate=should_deduplicate,
                )
            )
        steps.append(
            _make_botorch_step(
                generator=suggested_model,
                winsorization_config=winsorization_config,
                derelativize_with_raw_status_quo=derelativize_with_raw_status_quo,
                no_winsorization=no_winsorization,
                max_parallelism=bo_parallelism,
                model_kwargs=model_kwargs,
                should_deduplicate=should_deduplicate,
                disable_progbar=disable_progbar,
                jit_compile=jit_compile,
                use_saasbo=use_saasbo,
                use_input_warping=use_input_warping,
            ),
        )
        # set name for GS
        bo_step = steps[-1]
        surrogate_spec = bo_step.model_kwargs.get("surrogate_spec")
        name = None
        if (
            bo_step.generator is Generators.BOTORCH_MODULAR
            and surrogate_spec is not None
            and (model_config := surrogate_spec.model_configs[0]).botorch_model_class
            == SaasFullyBayesianSingleTaskGP
        ):
            name = f"Sobol+{model_config.name}"
        gs = GenerationStrategy(steps=steps, name=name)
        logger.info(
            f"Using Bayesian Optimization generation strategy: {gs}. Iterations after"
            f" {num_remaining_initialization_trials} will take longer to generate due"
            " to model-fitting."
        )
    else:  # `force_random_search` is True or we could not suggest BO model
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
    winsorization_config: None | (WinsorizationConfig | dict[str, WinsorizationConfig]),
    derelativize_with_raw_status_quo: bool,
    no_winsorization: bool,
) -> TConfig | None:
    if no_winsorization:
        if winsorization_config is not None:
            warnings.warn(
                "`no_winsorization = True` but `winsorization_config` has been set. "
                "Not winsorizing.",
                stacklevel=2,
            )
        return None
    if winsorization_config:
        return {"winsorization_config": winsorization_config}
    return {"derelativize_with_raw_status_quo": derelativize_with_raw_status_quo}

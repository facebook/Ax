#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import Any

import torch
from ax.adapter.registry import Generators
from ax.api.utils.structs import GenerationStrategyDispatchStruct
from ax.core.experiment_status import ExperimentStatus
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.dispatch_utils import get_derelativize_config
from ax.generation_strategy.generation_strategy import (
    GenerationNode,
    GenerationStrategy,
)
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MaxTrialsAwaitingData, MinTrials
from ax.generators.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from pyre_extensions import none_throws


def _get_sobol_node(
    initialization_budget: int | None,
    min_observed_initialization_trials: int | None,
    initialize_with_center: bool,
    use_existing_trials_for_initialization: bool,
    allow_exceeding_initialization_budget: bool,
    initialization_random_seed: int | None,
) -> GenerationNode:
    """Constructs a Sobol node based on inputs from
    ``struct``. The Sobol generator utilizes
    `initialization_random_seed` if specified.

    This node always transitions to "MBM", using the following transition criteria:
    - MinTrials enforcing the initialization budget.
        - If the initialization budget is not specified, it defaults to 5.
        - The TC will not block generation if `allow_exceeding_initialization_budget`
            is set to True.
        - The TC excludes FAILED and ABANDONED trials from the count, so that
            more trials can be generated to meet the
            `min_observed_initialization_trials` requirement.
        - `use_existing_trials_for_initialization` controls whether trials previously
            attached to the experiment are counted as part of the initialization budget.
    - MinTrials enforcing the minimum number of observed initialization trials.
        - If `min_observed_initialization_trials` is not specified, it defaults
            to `max(1, initialization_budget // 2)`.
        - The TC currently only counts trials in status COMPLETED (with data attached)
            as observed trials.
    """
    # Set the default options.
    initialization_budget = initialization_budget
    if initialization_budget is None:
        initialization_budget = 5
    if min_observed_initialization_trials is None:
        min_observed_initialization_trials = max(1, initialization_budget // 2)
    if initialize_with_center and not use_existing_trials_for_initialization:
        # Account for center point in initialization, since the TC will not count it.
        initialization_budget -= 1
    # Construct the transition criteria.
    transition_criteria = [
        MinTrials(  # This represents the initialization budget.
            threshold=initialization_budget,
            transition_to="MBM",
            use_all_trials_in_exp=use_existing_trials_for_initialization,
            not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
        ),
        MinTrials(  # This represents minimum observed trials requirement.
            threshold=min_observed_initialization_trials,
            transition_to="MBM",
            use_all_trials_in_exp=True,
            only_in_statuses=[TrialStatus.COMPLETED],
            count_only_trials_with_data=True,
        ),
    ]
    # If we want to enforce the initialization budget, add a pausing
    # criterion that prevents exceeding the budget.
    pausing_criteria = None
    if not allow_exceeding_initialization_budget:
        pausing_criteria = [
            MaxTrialsAwaitingData(
                threshold=initialization_budget,
                not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
                use_all_trials_in_exp=use_existing_trials_for_initialization,
            )
        ]
    return GenerationNode(
        name="Sobol",
        generator_specs=[
            GeneratorSpec(
                generator_enum=Generators.SOBOL,
                generator_kwargs={"seed": initialization_random_seed},
            )
        ],
        transition_criteria=transition_criteria,
        pausing_criteria=pausing_criteria,
        should_deduplicate=True,
        suggested_experiment_status=ExperimentStatus.INITIALIZATION,
    )


def _get_mbm_node(
    method: str,
    torch_device: str | None,
    simplify_parameter_changes: bool,
    model_config: ModelConfig | None = None,
    botorch_acqf_class: type[AcquisitionFunction] | None = None,
) -> tuple[GenerationNode, str]:
    """Constructs an MBM node based on the method specified in
    ``struct``.

    Args:
        method: The method to use for the MBM node. This can be one of
            - "quality": Uses Warped SAAS model.
            - "fast": Uses MBM defaults.
            - "custom": Uses the provided ``model_config``.
        torch_device: The torch device to use for the MBM node.
        simplify_parameter_changes: Whether to use BONSAI [Daulton2026bonsai]_ to
            simplify parameter changes in the MBM node.
        model_config: Optional model config to use for the MBM node.
            This is only supported when ``method`` is "custom".
        botorch_acqf_class: An optional BoTorch ``AcquisitionFunction`` class
            to use for the MBM node.
    """
    # Construct the surrogate spec.
    if method == "custom":
        model_config = none_throws(model_config)
        model_configs = [model_config]
        mbm_name = (
            model_config.name if model_config.name is not None else "custom_config"
        )
    elif method == "quality":
        model_configs = [
            ModelConfig(
                botorch_model_class=SaasFullyBayesianSingleTaskGP,
                model_options={"use_input_warping": True},
                mll_options={
                    "disable_progbar": True,
                },
                name="WarpedSAAS",
            )
        ]
        mbm_name = method
    elif method == "fast":
        model_configs = [ModelConfig(name="MBM defaults")]
        mbm_name = method
    else:
        raise UnsupportedError(f"Unsupported generation method: {method}.")

    # Append acquisition function class name to the node name if provided.
    if botorch_acqf_class is not None:
        mbm_name = f"{mbm_name}+{botorch_acqf_class.__name__}"

    device = None if torch_device is None else torch.device(torch_device)

    # Construct generator kwargs.
    generator_kwargs: dict[str, Any] = {
        "surrogate_spec": SurrogateSpec(model_configs=model_configs),
        "torch_device": device,
        "transform_configs": get_derelativize_config(
            derelativize_with_raw_status_quo=True
        ),
        "acquisition_options": {
            "prune_irrelevant_parameters": simplify_parameter_changes
        },
    }
    if botorch_acqf_class is not None:
        generator_kwargs["botorch_acqf_class"] = botorch_acqf_class

    return GenerationNode(
        name="MBM",
        generator_specs=[
            GeneratorSpec(
                generator_enum=Generators.BOTORCH_MODULAR,
                generator_kwargs=generator_kwargs,
            )
        ],
        should_deduplicate=True,
        suggested_experiment_status=ExperimentStatus.OPTIMIZATION,
    ), mbm_name


def choose_generation_strategy(
    struct: GenerationStrategyDispatchStruct,
    model_config: ModelConfig | None = None,
    botorch_acqf_class: type[AcquisitionFunction] | None = None,
) -> GenerationStrategy:
    """
    Choose a generation strategy based on the properties of the experiment and the
    inputs provided in ``struct``.

    NOTE: The behavior of this function is subject to change. It will be updated to
    produce best general purpose generation strategies based on benchmarking results.

    Args:
        struct: A ``GenerationStrategyDispatchStruct``
            object that informs
            the choice of generation strategy.
        model_config: An optional ``ModelConfig`` to use for the Bayesian optimization
            phase. This must be provided when ``struct.method`` is ``"custom"``, and
            must not be provided otherwise.
        botorch_acqf_class: An optional BoTorch ``AcquisitionFunction`` class to use
            for the Bayesian optimization phase. When provided, it will be passed as a
            model kwarg to the MBM node and its name will be appended to the node name.

    Returns:
        A generation strategy.
    """
    # Validate model_config usage.
    if struct.method == "custom":
        if model_config is None:
            raise UserInputError("model_config must be provided when method='custom'.")
    elif model_config is not None:
        raise UserInputError(
            "model_config should only be provided when method='custom'. "
            f"Got method='{struct.method}'."
        )

    # Handle the random search case.
    if struct.method == "random_search":
        nodes = [
            GenerationNode(
                name="Sobol",
                generator_specs=[
                    GeneratorSpec(
                        generator_enum=Generators.SOBOL,
                        generator_kwargs={"seed": struct.initialization_random_seed},
                    )
                ],
                suggested_experiment_status=ExperimentStatus.INITIALIZATION,
            )
        ]
        gs_name = "QuasiRandomSearch"
    else:
        mbm_node, mbm_name = _get_mbm_node(
            method=struct.method,
            torch_device=struct.torch_device,
            simplify_parameter_changes=struct.simplify_parameter_changes,
            model_config=model_config,
            botorch_acqf_class=botorch_acqf_class,
        )
        if (
            struct.initialization_budget is None
            or struct.initialization_budget > struct.initialize_with_center
        ):
            nodes = [
                _get_sobol_node(
                    initialization_budget=struct.initialization_budget,
                    min_observed_initialization_trials=struct.min_observed_initialization_trials,  # noqa: E501
                    initialize_with_center=struct.initialize_with_center,
                    use_existing_trials_for_initialization=struct.use_existing_trials_for_initialization,  # noqa: E501
                    allow_exceeding_initialization_budget=struct.allow_exceeding_initialization_budget,  # noqa: E501
                    initialization_random_seed=struct.initialization_random_seed,
                ),
                mbm_node,
            ]
            gs_name = f"Sobol+MBM:{mbm_name}"
        else:
            nodes = [mbm_node]
            gs_name = f"MBM:{mbm_name}"
    if struct.initialize_with_center and (
        struct.initialization_budget is None or struct.initialization_budget > 0
    ):
        center_node = CenterGenerationNode(next_node_name=nodes[0].name)
        nodes.insert(0, center_node)
        gs_name = f"Center+{gs_name}"
    return GenerationStrategy(name=gs_name, nodes=nodes)

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from ax.core.base_trial import TrialStatus
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.generation_strategy import GenerationNode, GenerationStrategy
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.registry import Models
from ax.modelbridge.transition_criterion import MinTrials
from ax.models.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from ax.preview.api.configs import GenerationMethod, GenerationStrategyConfig
from botorch.models.transforms.input import Normalize, Warp
from gpytorch.kernels.linear_kernel import LinearKernel


def _get_sobol_node(
    gs_config: GenerationStrategyConfig,
) -> GenerationNode:
    """Constructs a Sobol node based on inputs from ``gs_config``.
    The Sobol generator utilizes `initialization_random_seed` if specified.

    This node always transitions to "MBM", using the following transition criteria:
    - MinTrials enforcing the initialization budget.
        - If the initialization budget is not specified, it defaults to 5.
        - The TC will not block generation if `allow_exceeding_initialization_budget`
            is set to True.
        - The TC is currently not restricted to any trial statuses and will
            count all trials.
        - `use_existing_trials_for_initialization` controls whether trials previously
            attached to the experiment are counted as part of the initialization budget.
    - MinTrials enforcing the minimum number of observed initialization trials.
        - If `min_observed_initialization_trials` is not specified, it defaults
            to `max(1, initialization_budget // 2)`.
        - The TC currently only counts trials in status COMPLETED (with data attached)
            as observed trials.
        - `use_existing_trials_for_initialization` controls whether trials previously
            attached to the experiment are counted as part of the required number of
            observed initialization trials.
    """
    # Set the default options.
    initialization_budget = gs_config.initialization_budget
    if initialization_budget is None:
        initialization_budget = 5
    min_observed_initialization_trials = gs_config.min_observed_initialization_trials
    if min_observed_initialization_trials is None:
        min_observed_initialization_trials = max(1, initialization_budget // 2)
    # Construct the transition criteria.
    transition_criteria = [
        MinTrials(  # This represents the initialization budget.
            threshold=initialization_budget,
            transition_to="MBM",
            block_gen_if_met=(not gs_config.allow_exceeding_initialization_budget),
            block_transition_if_unmet=True,
            use_all_trials_in_exp=gs_config.use_existing_trials_for_initialization,
        ),
        MinTrials(  # This represents minimum observed trials requirement.
            threshold=min_observed_initialization_trials,
            transition_to="MBM",
            block_gen_if_met=False,
            block_transition_if_unmet=True,
            use_all_trials_in_exp=gs_config.use_existing_trials_for_initialization,
            only_in_statuses=[TrialStatus.COMPLETED],
            count_only_trials_with_data=True,
        ),
    ]
    return GenerationNode(
        node_name="Sobol",
        model_specs=[
            ModelSpec(
                model_enum=Models.SOBOL,
                model_kwargs={"seed": gs_config.initialization_random_seed},
            )
        ],
        transition_criteria=transition_criteria,
        should_deduplicate=True,
    )


def _get_mbm_node(
    gs_config: GenerationStrategyConfig,
) -> GenerationNode:
    """Constructs an MBM node based on the method specified in ``gs_config``.

    The ``SurrogateSpec`` takes the following form for the given method:
    - BALANCED: Two model configs: one with MBM defaults, the other with
        linear kernel with input warping.
    - FAST: An empty model config that utilizes MBM defaults.
    """
    # Construct the surrogate spec.
    if gs_config.method == GenerationMethod.FAST:
        model_configs = [ModelConfig(name="MBM defaults")]
    elif gs_config.method == GenerationMethod.BALANCED:
        model_configs = [
            ModelConfig(name="MBM defaults"),
            ModelConfig(
                covar_module_class=LinearKernel,
                input_transform_classes=[Warp, Normalize],
                input_transform_options={"Normalize": {"center": 0.0}},
                name="LinearKernel with Warp",
            ),
        ]
    else:
        raise UnsupportedError(f"Unsupported generation method: {gs_config.method}.")
    torch_device = (
        None if gs_config.torch_device is None else torch.device(gs_config.torch_device)
    )
    return GenerationNode(
        node_name="MBM",
        model_specs=[
            ModelSpec(
                model_enum=Models.BOTORCH_MODULAR,
                model_kwargs={
                    "surrogate_spec": SurrogateSpec(model_configs=model_configs),
                    "torch_device": torch_device,
                },
            )
        ],
        should_deduplicate=True,
    )


def choose_generation_strategy(
    gs_config: GenerationStrategyConfig,
) -> GenerationStrategy:
    """Choose a generation strategy based on the properties of the experiment
    and the inputs provided in ``gs_config``.

    NOTE: The behavior of this function is subject to change. It will be updated to
    produce best general purpose generation strategies based on benchmarking results.

    Args:
        gs_config: A ``GenerationStrategyConfig`` object that informs
            the choice of generation strategy.

    Returns:
        A generation strategy.
    """
    # Handle the random search case.
    if gs_config.method == GenerationMethod.RANDOM_SEARCH:
        return GenerationStrategy(
            name="QuasiRandomSearch",
            nodes=[
                GenerationNode(
                    node_name="Sobol",
                    model_specs=[
                        ModelSpec(
                            model_enum=Models.SOBOL,
                            model_kwargs={"seed": gs_config.initialization_random_seed},
                        )
                    ],
                )
            ],
        )
    # Construct the nodes.
    sobol_node = _get_sobol_node(gs_config)
    # Construct the MBM node.
    mbm_node = _get_mbm_node(gs_config)
    method_str = gs_config.method.value
    return GenerationStrategy(
        name=f"Sobol+MBM:{method_str}",
        nodes=[sobol_node, mbm_node],
    )

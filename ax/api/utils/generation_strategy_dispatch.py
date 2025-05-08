#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from ax.api.utils.structs import GenerationStrategyDispatchStruct
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UnsupportedError
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.generation_strategy import (
    GenerationNode,
    GenerationStrategy,
)
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.modelbridge.registry import Generators
from ax.models.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from botorch.models.transforms.input import Normalize, Warp
from gpytorch.kernels.linear_kernel import LinearKernel


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
        - The TC is currently not restricted to any trial statuses and will
            count all trials.
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
    min_observed_initialization_trials = min_observed_initialization_trials
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
            block_gen_if_met=(not allow_exceeding_initialization_budget),
            block_transition_if_unmet=True,
            use_all_trials_in_exp=use_existing_trials_for_initialization,
        ),
        MinTrials(  # This represents minimum observed trials requirement.
            threshold=min_observed_initialization_trials,
            transition_to="MBM",
            block_gen_if_met=False,
            block_transition_if_unmet=True,
            use_all_trials_in_exp=True,
            only_in_statuses=[TrialStatus.COMPLETED],
            count_only_trials_with_data=True,
        ),
    ]
    return GenerationNode(
        node_name="Sobol",
        model_specs=[
            GeneratorSpec(
                model_enum=Generators.SOBOL,
                model_kwargs={"seed": initialization_random_seed},
            )
        ],
        transition_criteria=transition_criteria,
        should_deduplicate=True,
    )


def _get_mbm_node(
    method: str,
    torch_device: str | None,
) -> GenerationNode:
    """Constructs an MBM node based on the method specified in
    ``struct``.

    The ``SurrogateSpec`` takes the following form for the given method:
    - BALANCED: Two model configs: one with MBM defaults, the other with
        linear kernel with input warping.
    - FAST: An empty model config that utilizes MBM defaults.
    """
    # Construct the surrogate spec.
    if method == "fast":
        model_configs = [ModelConfig(name="MBM defaults")]
    elif method == "balanced":
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
        raise UnsupportedError(f"Unsupported generation method: {method}.")

    device = None if torch_device is None else torch.device(torch_device)

    return GenerationNode(
        node_name="MBM",
        model_specs=[
            GeneratorSpec(
                model_enum=Generators.BOTORCH_MODULAR,
                model_kwargs={
                    "surrogate_spec": SurrogateSpec(model_configs=model_configs),
                    "torch_device": device,
                },
            )
        ],
        should_deduplicate=True,
    )


def choose_generation_strategy(
    struct: GenerationStrategyDispatchStruct,
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

    Returns:
        A generation strategy.
    """
    # Handle the random search case.
    if struct.method == "random_search":
        nodes = [
            GenerationNode(
                node_name="Sobol",
                model_specs=[
                    GeneratorSpec(
                        model_enum=Generators.SOBOL,
                        model_kwargs={"seed": struct.initialization_random_seed},
                    )
                ],
            )
        ]
        gs_name = "QuasiRandomSearch"
    else:
        nodes = [
            _get_sobol_node(
                initialization_budget=struct.initialization_budget,
                min_observed_initialization_trials=struct.min_observed_initialization_trials,  # noqa: E501
                initialize_with_center=struct.initialize_with_center,
                use_existing_trials_for_initialization=struct.use_existing_trials_for_initialization,  # noqa: E501
                allow_exceeding_initialization_budget=struct.allow_exceeding_initialization_budget,  # noqa: E501
                initialization_random_seed=struct.initialization_random_seed,
            ),
            _get_mbm_node(
                method=struct.method,
                torch_device=struct.torch_device,
            ),
        ]
        gs_name = f"Sobol+MBM:{struct.method}"
    if struct.initialize_with_center:
        center_node = CenterGenerationNode(next_node_name=nodes[0].node_name)
        nodes.insert(0, center_node)
        gs_name = f"Center+{gs_name}"
    return GenerationStrategy(name=gs_name, nodes=nodes)

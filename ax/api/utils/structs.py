# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig


@dataclass
class ExperimentStruct:
    """
    Allows specifying the search space of an experiment along
    with other metadata.
    """

    parameters: list[RangeParameterConfig | ChoiceParameterConfig]
    # Parameter constraints will be parsed via SymPy
    # Ex: "num_layers1 <= num_layers2", "compound_a + compound_b <= 1"
    parameter_constraints: list[str]

    name: str | None
    description: str | None
    experiment_type: str | None
    owner: str | None


@dataclass
class GenerationStrategyDispatchStruct:
    """
    Allows the user to configure the way candidates are generated during the experiment.
    This is used, along with the properties of the experiment, to determine the
    ``GenerationStrategy`` to use for candidate generation.

    Args:
        method: The generation method to use. Provides high level guidance to the
            underlying generation strategy dispatch logic, which is responsible for
            determinining the exact details. Available options are:
                - ``"balanced"``, a balanced generation method that may utilize
                    (per-metric) model selection to achieve a good model accuracy.
                - ``"fast"``, a faster generation method that uses the built-in
                    defaults from the Modular BoTorch Model without any model
                    selection.
                - ``"random_search"``, primarily intended for pure exploration
                    experiments, this method utilizes quasi-random Sobol sequences
                    for candidate generation.
        initialization_budget: The number of trials to use for initialization.
            If ``None``, a default budget of 5 trials is used.
        initialization_random_seed: The random seed to use with the Sobol generator
            that generates the initialization trials.
        initialize_with_center: If True, the center of the search space is used as the
            first point. The definition of center respects the scaling of the
            parameters. For discrete parameters, the median value is considered the
            center, with the later points being used to break ties.
        use_existing_trials_for_initialization: Whether to count all trials attached
            to the experiment as part of the initialization budget. For example,
            if 2 trials were manually attached to the experiment and this option
            is set to ``True``, we will only generate `initialization_budget - 2`
            additional trials for initialization.
        min_observed_initialization_trials: The minimum required number of
            initialization trials with observations before the generation strategy
            is allowed to transition away from the initialization phase.
            Defaults to `max(1, initialization_budget // 2)`.
        allow_exceeding_initialization_budget: This option determines the behavior
            of the generation strategy when the ``initialization_budget`` is exhausted
            and ``min_observed_initialization_trials`` is not met. If this is ``True``,
            the generation strategy will generate additional initialization trials when
            a new trial is requested, exceeding the specified ``initialization_budget``.
            If this is ``False``, the generation strategy will raise an error and the
            candidate generation may be continued when additional data is observed
            for the existing trials.
        torch_device: The device to use for model fitting and candidate
            generation in PyTorch / BoTorch based generation nodes.
            NOTE: This option is not validated. Please ensure that the string
            input corresponds to a valid device.
    """

    method: Literal["balanced", "fast", "random_search"] = "fast"
    # Initialization options
    initialization_budget: int | None = None
    initialization_random_seed: int | None = None
    initialize_with_center: bool = True
    use_existing_trials_for_initialization: bool = True
    min_observed_initialization_trials: int | None = None
    allow_exceeding_initialization_budget: bool = False
    # Misc options
    torch_device: str | None = None

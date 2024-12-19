# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Mapping, Sequence

from ax.preview.api.types import TParameterValue
from ax.storage.registry_bundle import RegistryBundleBase


class ParameterType(Enum):
    """
    The ParameterType enum allows users to specify the type of a parameter.
    """

    FLOAT = "float"
    INT = "int"
    STRING = "str"
    BOOL = "bool"


class ParameterScaling(Enum):
    """
    The ParameterScaling enum allows users to specify which scaling to apply during
    candidate generation. This is useful for parameters that should not be explored
    on the same scale, such as learning rates and batch sizes.
    """

    LINEAR = "linear"
    LOG = "log"


@dataclass
class RangeParameterConfig:
    """
    RangeParameterConfig allows users to specify the a continuous dimension of an
    experiment's search space and will internally validate the inputs.
    """

    name: str

    bounds: tuple[float, float]
    parameter_type: ParameterType
    step_size: float | None = None
    scaling: ParameterScaling | None = None


@dataclass
class ChoiceParameterConfig:
    """
    ChoiceParameterConfig allows users to specify the a discrete dimension of an
    experiment's search space and will internally validate the inputs.
    """

    name: str
    values: List[float] | List[int] | List[str] | List[bool]
    parameter_type: ParameterType
    is_ordered: bool | None = None
    dependent_parameters: Mapping[TParameterValue, Sequence[str]] | None = None


@dataclass
class ExperimentConfig:
    """
    ExperimentConfig allows users to specify the SearchSpace of an experiment along
    with other metadata.
    """

    name: str
    parameters: list[RangeParameterConfig | ChoiceParameterConfig]
    # Parameter constraints will be parsed via SymPy
    # Ex: "num_layers1 <= num_layers2", "compound_a + compound_b <= 1"
    parameter_constraints: list[str] = field(default_factory=list)

    description: str | None = None
    experiment_type: str | None = None
    owner: str | None = None


class GenerationMethod(Enum):
    """An enum to specify the desired candidate generation method for the experiment.
    This is used in ``GenerationStrategyConfig``, along with the properties of the
    experiment, to determine the generation strategy to use for candidate generation.

    NOTE: New options should be rarely added to this enum. This is not intended to be
    a list of generation strategies for the user to choose from. Instead, this enum
    should only provide high level guidance to the underlying generation strategy
    dispatch logic, which is responsible for determinining the exact details.

    Available options are:
        BALANCED: A balanced generation method that may utilize (per-metric) model
            selection to achieve a good model accuracy. This method excludes expensive
            methods, such as the fully Bayesian SAASBO model. Used by default.
        FAST: A faster generation method that uses the built-in defaults from the
            Modular BoTorch Model without any model selection.
        RANDOM_SEARCH: Primarily intended for pure exploration experiments, this
            method utilizes quasi-random Sobol sequences for candidate generation.
    """

    BALANCED = "balanced"
    FAST = "fast"
    RANDOM_SEARCH = "random_search"


@dataclass
class GenerationStrategyConfig:
    """A dataclass used to configure the generation strategy used in the experiment.
    This is used, along with the properties of the experiment, to determine the
    generation strategy to use for candidate generation.

    Args:
        method: The generation method to use. See ``GenerationMethod`` for more details.
        initialization_budget: The number of trials to use for initialization.
            If ``None``, a default budget of 5 trials is used.
        initialization_random_seed: The random seed to use with the Sobol generator
            that generates the initialization trials.
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

    method: GenerationMethod = GenerationMethod.BALANCED
    # Initialization options
    initialization_budget: int | None = None
    initialization_random_seed: int | None = None
    use_existing_trials_for_initialization: bool = True
    min_observed_initialization_trials: int | None = None
    allow_exceeding_initialization_budget: bool = False
    # Misc options
    torch_device: str | None = None


@dataclass
class OrchestrationConfig:
    parallelism: int = 1
    tolerated_trial_failure_rate: float = 0.5
    initial_seconds_between_polls: int = 1


@dataclass
class StorageConfig:
    creator: Callable[..., Any] | None = None  # pyre-fixme[4]
    url: str | None = None
    registry_bundle: RegistryBundleBase | None = None

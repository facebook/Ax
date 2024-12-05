# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Mapping, Optional, Sequence

from ax.preview.api.types import TParameterValue


# Note: I'm not sold these should be dataclasses, just using this as a placeholder


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
    owner: str | None = None


@dataclass
class GenerationStrategyConfig:
    # This will hold the args to choose_generation_strategy
    num_trials: Optional[int] = None
    num_initialization_trials: Optional[int] = None
    maximum_parallelism: Optional[int] = None


@dataclass
class OrchestrationConfig:
    parallelism: int = 1
    tolerated_trial_failure_rate: float = 0.5
    seconds_between_polls: float = 1.0


@dataclass
class DatabaseConfig:
    url: str

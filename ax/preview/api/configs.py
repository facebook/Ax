# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

from ax.core.types import TParamValue

# Note: I'm not sold these should be dataclasses, just using this as a placeholder


class DomainType(Enum):
    """
    The DomainType enum allows the ParameterConfig to know whether to expect inputs for
    a RangeParameter or ChoiceParameter (or FixedParameter) during the parameter
    instantiation and validation process.
    """

    RANGE = "range"
    CHOICE = "choice"


class ParameterType(Enum):
    """
    The ParameterType enum allows users to specify the type of a parameter.
    """

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"


class ParameterScaling(Enum):
    """
    The ParameterScaling enum allows users to specify which scaling to apply during
    candidate generation. This is useful for parameters that should not be explored
    on the same scale, such as learning rates and batch sizes.
    """

    LINEAR = "linear"
    LOG = "log"


@dataclass
class ParameterConfig:
    """
    ParameterConfig allows users to specify the parameters of an experiment and will
    internally validate the inputs to ensure they are valid for the given DomainType.
    """

    name: str
    domain_type: DomainType
    parameter_type: ParameterType | None = None

    # Fields for RANGE
    bounds: Optional[tuple[float, float]] = None
    step_size: Optional[float] = None
    scaling: Optional[ParameterScaling] = None

    # Fields for CHOICE ("FIXED" is Choice with len(values) == 1)
    values: Optional[Union[list[float], list[str], list[bool]]] = None
    is_ordered: Optional[bool] = None
    dependent_parameters: Optional[dict[TParamValue, str]] = None


@dataclass
class ExperimentConfig:
    """
    ExperimentConfig allows users to specify the SearchSpace and OptimizationConfig of
    an Experiment and validates their inputs jointly.

    This will also be the construct that handles transforming string-based inputs (the
    objective, parameter constraints, and output constraints) into their corresponding
    Ax class using SymPy.
    """

    name: str
    parameters: list[ParameterConfig]
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

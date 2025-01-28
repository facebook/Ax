# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np

from ax.core.experiment import Experiment

from ax.core.formatting_utils import DataType
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    Parameter,
    ParameterType as CoreParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import validate_constraint_parameters
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace
from ax.exceptions.core import UserInputError
from ax.preview.api.configs import (
    ChoiceParameterConfig,
    ExperimentConfig,
    ParameterScaling,
    ParameterType,
    RangeParameterConfig,
)
from ax.preview.api.utils.instantiation.from_string import parse_parameter_constraint


def parameter_from_config(
    config: RangeParameterConfig | ChoiceParameterConfig,
) -> Parameter:
    """
    Create a RangeParameter, ChoiceParameter, or FixedParameter from a ParameterConfig.
    """

    if isinstance(config, RangeParameterConfig):
        lower, upper = config.bounds

        # TODO[mpolson64] Add support for RangeParameterConfig.step_size native to
        # RangeParameter instead of converting to ChoiceParameter
        if (step_size := config.step_size) is not None:
            if not (
                config.scaling == ParameterScaling.LINEAR or config.scaling is None
            ):
                raise UserInputError(
                    "Non-linear parameter scaling is not supported when using "
                    "step_size."
                )

            if (upper - lower) % step_size != 0:
                raise UserInputError(
                    "The range of the parameter must be evenly divisible by the "
                    "step size."
                )

            return ChoiceParameter(
                name=config.name,
                parameter_type=_parameter_type_converter(config.parameter_type),
                values=[*np.arange(lower, upper + step_size, step_size)],
                is_ordered=True,
            )

        return RangeParameter(
            name=config.name,
            parameter_type=_parameter_type_converter(config.parameter_type),
            lower=lower,
            upper=upper,
            log_scale=config.scaling == ParameterScaling.LOG,
        )

    else:
        # If there is only one value, create a FixedParameter instead of a
        # ChoiceParameter
        if len(config.values) == 1:
            return FixedParameter(
                name=config.name,
                parameter_type=_parameter_type_converter(config.parameter_type),
                value=config.values[0],
                # pyre-fixme[6] Variance issue caused by FixedParameter.dependents
                # using List instead of immutable container type.
                dependents=config.dependent_parameters,
            )

        return ChoiceParameter(
            name=config.name,
            parameter_type=_parameter_type_converter(config.parameter_type),
            # pyre-fixme[6] Variance issue caused by ChoiceParameter.value using List
            # instead of immutable container type.
            values=config.values,
            is_ordered=config.is_ordered,
            # pyre-fixme[6] Variance issue caused by ChoiceParameter.dependents using
            # List instead of immutable container type.
            dependents=config.dependent_parameters,
        )


def experiment_from_config(config: ExperimentConfig) -> Experiment:
    """Create an Experiment from an ExperimentConfig."""
    parameters = [
        parameter_from_config(config=parameter_config)
        for parameter_config in config.parameters
    ]

    constraints = [
        parse_parameter_constraint(constraint_str=constraint_str)
        for constraint_str in config.parameter_constraints
    ]

    # Ensure that all ParameterConstraints are valid and acting on existing parameters
    for constraint in constraints:
        validate_constraint_parameters(
            parameters=[
                parameter
                for parameter in parameters
                if parameter.name in constraint.constraint_dict.keys()
            ]
        )

    if any(p.is_hierarchical for p in parameters):
        search_space = HierarchicalSearchSpace(
            parameters=parameters, parameter_constraints=constraints
        )
    else:
        search_space = SearchSpace(
            parameters=parameters, parameter_constraints=constraints
        )

    return Experiment(
        search_space=search_space,
        name=config.name,
        description=config.description,
        experiment_type=config.experiment_type,
        properties={"owners": [config.owner]},
        default_data_type=DataType.MAP_DATA,
    )


def _parameter_type_converter(parameter_type: ParameterType) -> CoreParameterType:
    """
    Convert from an API ParameterType to a core Ax ParameterType.
    """

    if parameter_type == ParameterType.BOOL:
        return CoreParameterType.BOOL
    elif parameter_type == ParameterType.FLOAT:
        return CoreParameterType.FLOAT
    elif parameter_type == ParameterType.INT:
        return CoreParameterType.INT
    elif parameter_type == ParameterType.STRING:
        return CoreParameterType.STRING
    else:
        raise UserInputError(f"Unsupported parameter type {parameter_type}.")

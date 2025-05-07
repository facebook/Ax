# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    Parameter,
    ParameterType as CoreParameterType,
    RangeParameter,
)
from ax.exceptions.core import UserInputError


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
            if not (config.scaling == "linear" or config.scaling is None):
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
            log_scale=config.scaling == "log",
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


def _parameter_type_converter(parameter_type: str) -> CoreParameterType:
    """
    Convert from an API ParameterType to a core Ax ParameterType.
    """

    if parameter_type == "bool":
        return CoreParameterType.BOOL
    elif parameter_type == "float":
        return CoreParameterType.FLOAT
    elif parameter_type == "int":
        return CoreParameterType.INT
    elif parameter_type == "str":
        return CoreParameterType.STRING
    else:
        raise UserInputError(f"Unsupported parameter type {parameter_type}.")

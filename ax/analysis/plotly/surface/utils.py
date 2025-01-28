# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import math

import numpy as np
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    Parameter,
    RangeParameter,
    TParamValue,
)


def get_parameter_values(parameter: Parameter, density: int = 100) -> list[TParamValue]:
    """
    Get a list of parameter values to predict over for a given parameter.
    """

    # For RangeParameter use linspace for the range of the parameter
    if isinstance(parameter, RangeParameter):
        if parameter.log_scale:
            return np.logspace(
                math.log10(parameter.lower), math.log10(parameter.upper), density
            ).tolist()

        return np.linspace(parameter.lower, parameter.upper, density).tolist()

    # For ChoiceParameter use the values of the parameter directly
    if isinstance(parameter, ChoiceParameter) and parameter.is_ordered:
        return parameter.values

    raise ValueError(
        f"Parameter {parameter.name} must be a RangeParameter or "
        "ChoiceParameter with is_ordered=True to be used in surface plot."
    )


def select_fixed_value(parameter: Parameter) -> TParamValue:
    """
    Select a fixed value for a parameter. Use mean for RangeParameter, "middle" value
    for ChoiceParameter, and value for FixedParameter.
    """
    if isinstance(parameter, RangeParameter):
        return (parameter.lower * 1.0 + parameter.upper) / 2
    elif isinstance(parameter, ChoiceParameter):
        return parameter.values[len(parameter.values) // 2]
    elif isinstance(parameter, FixedParameter):
        return parameter.value
    else:
        raise ValueError(f"Got unexpected parameter type {parameter}.")


def is_axis_log_scale(parameter: Parameter) -> bool:
    """
    Check if the parameter is log scale.
    """
    return isinstance(parameter, RangeParameter) and parameter.log_scale

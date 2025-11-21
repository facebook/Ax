# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import math

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    DerivedParameter,
    FixedParameter,
    Parameter,
    RangeParameter,
    TParamValue,
)
from ax.core.search_space import SearchSpace


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


def get_features_for_slice_or_contour(
    parameters: dict[str, TParamValue], search_space: SearchSpace
) -> ObservationFeatures:
    """Fill missing values for a specific point in the slice/contour.


    For missing parameter values, the value is chosen via `select_fixed_value`.
    For derived parameters, the value is computed from the other parameters.

    Args:
        parameters: Specified values for an individual point in the  slice/contour
            plot.
        search_space: The search space.

    Returns:
        A full parameterization for the point.

    """
    derived_params = [
        p for p in search_space.parameters.values() if isinstance(p, DerivedParameter)
    ]
    params = parameters.copy()
    for parameter in search_space.parameters.values():
        if parameter.name not in parameters:
            if not isinstance(parameter, DerivedParameter):
                params[parameter.name] = select_fixed_value(parameter=parameter)
    for p in derived_params:
        params[p.name] = p.compute(parameters=params)
    return ObservationFeatures(parameters=params)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from __future__ import annotations

import math
from typing import TYPE_CHECKING

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
from ax.core.trial import Trial
from ax.service.utils.best_point import (
    get_best_parameters_from_model_predictions_with_trial_index,
)
from pyre_extensions import assert_is_instance, none_throws

if TYPE_CHECKING:
    from ax.core.experiment import Experiment
    from ax.generation_strategy.generation_strategy import GenerationStrategy


def _get_best_trial_info(
    experiment: Experiment,
    generation_strategy: GenerationStrategy,
) -> tuple[dict[str, TParamValue], int, str] | None:
    """Get parameterization and arm info from the best trial for single-objective.

    This is a private helper function used by get_fixed_values_for_slice_or_contour.

    Args:
        experiment: The experiment to get the best trial from.
        generation_strategy: The generation strategy with a fitted adapter for
            model-based best point estimation.

    Returns:
        A tuple of (parameterization, trial_index, arm_name), or None if the
        experiment is a multi-objective optimization problem or if no best
        trial can be found.
    """

    optimization_config = experiment.optimization_config
    if optimization_config is None:
        return None

    # Best trial is not applicable to multi-objective optimization
    if optimization_config.is_moo_problem:
        return None

    result = get_best_parameters_from_model_predictions_with_trial_index(
        experiment=experiment,
        adapter=generation_strategy.adapter,
        optimization_config=optimization_config,
    )

    if result is None:
        return None

    trial_index, parameterization, _prediction = result
    # Get the arm name from the trial
    trial = assert_is_instance(experiment.trials[trial_index], Trial)
    arm_name = none_throws(trial.arm).name
    return parameterization, trial_index, arm_name


def get_fixed_values_for_slice_or_contour(
    experiment: Experiment,
    generation_strategy: GenerationStrategy | None,
) -> tuple[dict[str, TParamValue], str]:
    """Get complete fixed parameter values for all parameters in slice/contour plots.

    This function computes fixed values for all non-derived parameters in the
    search space using the following priority:
    1. status_quo arm (if it exists and is within the search space)
    2. Best trial parameterization (for single-objective optimization)
    3. Center of search space (fallback)

    Args:
        experiment: The experiment to get fixed values from.
        generation_strategy: The generation strategy with a fitted adapter.

    Returns:
        A tuple of:
        - A dictionary mapping parameter names to their fixed values for all
          non-derived parameters in the search space.
        - A description string for use in subtitles (e.g., "their status_quo
          value (arm_name)", "their best trial value (arm_name)", "the center
          of the search space").
    """
    # Start with center of search space for all non-derived parameters
    fixed_values = {
        p.name: select_fixed_value(p)
        for p in experiment.search_space.parameters.values()
        if not isinstance(p, DerivedParameter)
    }

    # First priority: use status_quo if it exists and is within the search space
    # (If status_quo is outside the search space, the model would be extrapolating)
    if experiment.status_quo is not None and experiment.search_space.check_membership(
        parameterization=experiment.status_quo.parameters,
        raise_error=False,
        check_all_parameters_present=True,
    ):
        fixed_values.update(experiment.status_quo.parameters)
        arm_name = none_throws(experiment.status_quo).name
        return fixed_values, f"their status_quo value (Arm {arm_name})"

    # Second priority: use best trial for single-objective optimization
    if generation_strategy is not None:
        best_trial_info = _get_best_trial_info(
            experiment=experiment,
            generation_strategy=generation_strategy,
        )
        if best_trial_info is not None:
            parameterization, _trial_index, arm_name = best_trial_info
            fixed_values.update(parameterization)
            return fixed_values, f"their best trial value (Arm {arm_name})"

    # Fallback: center of search space (already computed)
    return fixed_values, "the center of the search space"


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
    parameters: dict[str, TParamValue],
    search_space: SearchSpace,
    fixed_values: dict[str, TParamValue],
) -> ObservationFeatures:
    """Fill missing values for a specific point in the slice/contour.

    For missing parameter values, the value is taken from `fixed_values`.
    For derived parameters, the value is computed from the other parameters.

    Args:
        parameters: Specified values for an individual point in the slice/contour
            plot.
        search_space: The search space.
        fixed_values: Pre-computed fixed values for inactive parameters. Should
            be obtained from `get_fixed_values_for_slice_or_contour()`.

    Returns:
        A full parameterization for the point.

    """
    derived_params = [
        p for p in search_space.parameters.values() if isinstance(p, DerivedParameter)
    ]
    params = parameters.copy()
    for parameter in search_space.parameters.values():
        if parameter.name not in parameters and not isinstance(
            parameter, DerivedParameter
        ):
            params[parameter.name] = fixed_values[parameter.name]
    for p in derived_params:
        params[p.name] = p.compute(parameters=params)
    return ObservationFeatures(parameters=params)

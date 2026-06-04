#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import cast, Literal

from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
from ax.core.experiment import Experiment
from ax.core.types import (
    TEvaluationFunction,
    TEvaluationOutcome,
    TModelPredictArm,
    TParamValue,
)
from ax.exceptions.core import UnsupportedError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.instantiation import TParameterRepresentation
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import (
    assert_is_instance_list,
    assert_is_instance_of_tuple,
    assert_is_instance_optional,
)
from pyre_extensions import assert_is_instance


logger: logging.Logger = get_logger(__name__)


def _validate_values_list(
    raw_values: object, value_type: Literal["float", "int", "str", "bool"]
) -> list[float] | list[int] | list[str] | list[bool]:
    """Validate that raw_values is a list whose elements all match value_type."""
    values_seq = assert_is_instance(raw_values, list)
    if value_type == "bool":
        return assert_is_instance_list(values_seq, bool)
    if value_type == "str":
        return assert_is_instance_list(values_seq, str)
    # bool is a subclass of int, so reject bool explicitly before numeric checks.
    for v in values_seq:
        if isinstance(v, bool):
            raise TypeError(
                f"Value {v!r} is a bool but value_type is {value_type!r}. "
                "Use value_type='bool' for boolean choices."
            )
    if value_type == "int":
        return assert_is_instance_list(values_seq, int)
    return [float(assert_is_instance_of_tuple(v, (int, float))) for v in values_seq]


def _narrow_range_parameter_type(value_type: str) -> Literal["float", "int"]:
    if value_type not in ("float", "int"):
        raise ValueError(
            f"Invalid value_type {value_type!r} for range parameter; "
            "expected 'float' or 'int'."
        )
    return value_type


def _narrow_choice_parameter_type(
    value_type: str,
) -> Literal["float", "int", "str", "bool"]:
    if value_type not in ("float", "int", "str", "bool"):
        raise ValueError(
            f"Invalid value_type {value_type!r} for choice parameter; "
            "expected one of 'float', 'int', 'str', 'bool'."
        )
    return value_type


def _param_dict_to_config(
    param: TParameterRepresentation,
) -> RangeParameterConfig | ChoiceParameterConfig:
    """Convert a legacy parameter dict to a typed config."""
    param_type = assert_is_instance(param["type"], str)
    name = assert_is_instance(param["name"], str)

    if param_type == "range":
        bounds = assert_is_instance(param["bounds"], Sequence)
        value_type = _narrow_range_parameter_type(
            assert_is_instance(param.get("value_type", "float"), str)
        )
        log_scale = bool(param.get("log_scale", False))
        return RangeParameterConfig(
            name=name,
            bounds=(
                float(assert_is_instance_of_tuple(bounds[0], (int, float))),
                float(assert_is_instance_of_tuple(bounds[1], (int, float))),
            ),
            parameter_type=value_type,
            scaling="log" if log_scale else None,
        )
    elif param_type == "choice":
        value_type = _narrow_choice_parameter_type(
            assert_is_instance(param.get("value_type", "str"), str)
        )
        values = _validate_values_list(param["values"], value_type)
        is_ordered = assert_is_instance_optional(param.get("is_ordered"), bool)
        return ChoiceParameterConfig(
            name=name,
            values=values,
            parameter_type=value_type,
            is_ordered=is_ordered,
        )
    elif param_type == "fixed":
        value = assert_is_instance_of_tuple(param["value"], (bool, int, float, str))
        default_type_name = type(value).__name__
        value_type = _narrow_choice_parameter_type(
            assert_is_instance(param.get("value_type", default_type_name), str)
        )
        return ChoiceParameterConfig(
            name=name,
            values=_validate_values_list([value], value_type),
            parameter_type=value_type,
        )
    else:
        raise ValueError(f"Unsupported parameter type: {param_type}")


def _call_evaluation_function(
    evaluation_function: TEvaluationFunction,
    parameterization: Mapping[str, TParamValue],
) -> TEvaluationOutcome:
    """Call the evaluation function with the right number of arguments."""
    signature = inspect.signature(evaluation_function)
    num_params = len(signature.parameters)
    if num_params == 1:
        return cast(
            Callable[[Mapping[str, TParamValue]], TEvaluationOutcome],
            evaluation_function,
        )(parameterization)
    elif num_params == 2:
        return cast(
            Callable[[Mapping[str, TParamValue], float | None], TEvaluationOutcome],
            evaluation_function,
        )(parameterization, None)
    else:
        raise ValueError(
            "Evaluation function must take either one parameter "
            "(parameterization) or two parameters (parameterization and weight)."
        )


def _outcome_to_dict(
    outcome: TEvaluationOutcome,
    objective_name: str,
) -> Mapping[str, float | tuple[float, float]]:
    """Convert the various TEvaluationOutcome formats to TOutcome for Client."""
    if isinstance(outcome, dict):
        # dict[str, tuple[float, float]] or dict[str, float] etc.
        result: dict[str, float | tuple[float, float]] = {}
        for k, v in outcome.items():
            if isinstance(v, tuple):
                mean, sem = v
                if sem is None:
                    result[k] = float(mean)
                else:
                    result[k] = (float(mean), float(sem))
            else:
                result[k] = float(v)
        return result
    elif isinstance(outcome, tuple):
        # (float, float) or (float, None)
        mean, sem = outcome
        if sem is None:
            return {objective_name: float(mean)}
        else:
            return {objective_name: (float(mean), float(sem))}
    else:
        # Single float
        return {objective_name: float(outcome)}


def optimize(
    parameters: list[TParameterRepresentation],
    evaluation_function: TEvaluationFunction,
    experiment_name: str | None = None,
    objective_name: str | None = None,
    minimize: bool = False,
    parameter_constraints: list[str] | None = None,
    outcome_constraints: list[str] | None = None,
    total_trials: int = 20,
    arms_per_trial: int = 1,
    random_seed: int | None = None,
    generation_strategy: GenerationStrategy | None = None,
) -> tuple[Mapping[str, TParamValue], TModelPredictArm | None, Experiment, None]:
    """Construct and run a full optimization loop.

    .. deprecated::
        This function is deprecated. Use :class:`ax.api.client.Client` directly.
    """
    warnings.warn(
        "optimize is deprecated and will be removed in a future version of Ax. "
        "Please use Client from ax.api.client instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if arms_per_trial != 1:
        raise UnsupportedError("optimize() only supports arms_per_trial=1. ")

    if objective_name is None:
        objective_name = "objective"

    # Convert legacy parameter dicts to typed configs.
    param_configs = [_param_dict_to_config(p) for p in parameters]

    # Set up Client.
    client = Client(random_seed=random_seed)
    client.configure_experiment(
        parameters=param_configs,
        parameter_constraints=parameter_constraints,
        name=experiment_name,
    )
    objective_str = objective_name if not minimize else f"-{objective_name}"
    client.configure_optimization(
        objective=objective_str,
        outcome_constraints=outcome_constraints,
    )
    if generation_strategy is not None:
        client.set_generation_strategy(generation_strategy)

    # Run optimization loop.
    for _ in range(total_trials):
        try:
            trials = client.get_next_trials(max_trials=1)
        except Exception:
            logger.exception("Encountered exception during trial generation.")
            break

        for trial_index, parameterization in trials.items():
            try:
                raw_outcome = _call_evaluation_function(
                    evaluation_function, parameterization
                )
                outcome_data = _outcome_to_dict(raw_outcome, objective_name)
                client.complete_trial(trial_index=trial_index, raw_data=outcome_data)
            except Exception:
                logger.exception(
                    f"Encountered exception evaluating trial {trial_index}."
                )
                break

    # Get best parameters.
    best_params, prediction, _trial_index, _arm_name = (
        client.get_best_parameterization()
    )

    # Convert prediction (TOutcome) to TModelPredictArm format for
    # backward compatibility.
    means: dict[str, float] = {}
    covariances: dict[str, dict[str, float]] = {}
    for metric_name, value in prediction.items():
        if isinstance(value, tuple):
            means[metric_name] = value[0]
            covariances[metric_name] = {metric_name: value[1] ** 2}
        else:
            means[metric_name] = value
            covariances[metric_name] = {metric_name: 0.0}

    model_predict_arm: TModelPredictArm = (means, covariances)

    return best_params, model_predict_arm, client._experiment, None

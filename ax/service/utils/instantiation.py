#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Union, cast

import numpy as np
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import (
    PARAMETER_PYTHON_TYPE_MAP,
    ChoiceParameter,
    FixedParameter,
    Parameter,
    ParameterType,
    RangeParameter,
    TParameterType,
)
from ax.core.parameter_constraint import OrderConstraint, ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.core.simple_experiment import DEFAULT_OBJECTIVE_NAME
from ax.core.types import (
    ComparisonOp,
    TEvaluationOutcome,
    TFidelityTrialEvaluation,
    TParameterization,
    TParamValue,
    TTrialEvaluation,
)
from ax.utils.common.typeutils import (
    checked_cast,
    checked_cast_to_tuple,
    not_none,
    numpy_type_to_python_type,
)


"""Utilities for RESTful-like instantiation of Ax classes needed in AxClient."""


TParameterRepresentation = Dict[str, Union[TParamValue, List[TParamValue]]]
PARAM_CLASSES = ["range", "choice", "fixed"]
PARAM_TYPES = {"int": int, "float": float, "bool": bool, "str": str}
COMPARISON_OPS = {"<=": ComparisonOp.LEQ, ">=": ComparisonOp.GEQ}


def _get_parameter_type(python_type: TParameterType) -> ParameterType:
    for param_type, py_type in PARAMETER_PYTHON_TYPE_MAP.items():
        if py_type is python_type:
            return param_type
    raise ValueError(f"No AE parameter type corresponding to {python_type}.")


def _to_parameter_type(
    vals: List[TParamValue], typ: Optional[str], param_name: str, field_name: str
) -> ParameterType:
    if typ is None:
        typ = type(not_none(vals[0]))
        parameter_type = _get_parameter_type(typ)  # pyre-ignore[6]
        assert all(isinstance(x, typ) for x in vals), (
            f"Values in `{field_name}` not of the same type and no `value_type` was "
            f"explicitly specified; cannot infer value type for parameter {param_name}."
        )
        return parameter_type
    return _get_parameter_type(PARAM_TYPES[typ])  # pyre-ignore[6]


def _make_range_param(
    name: str, representation: TParameterRepresentation, parameter_type: Optional[str]
) -> RangeParameter:
    assert "bounds" in representation, "Bounds are required for range parameters."
    bounds = representation["bounds"]
    assert isinstance(bounds, list) and len(bounds) == 2, (
        f"Cannot parse parameter {name}: for range parameters, json representation "
        "should include a list of two values, lower and upper bounds of the range."
    )
    return RangeParameter(
        name=name,
        parameter_type=_to_parameter_type(bounds, parameter_type, name, "bounds"),
        lower=checked_cast_to_tuple((float, int), bounds[0]),
        upper=checked_cast_to_tuple((float, int), bounds[1]),
        log_scale=checked_cast(bool, representation.get("log_scale", False)),
        is_fidelity=checked_cast(bool, representation.get("is_fidelity", False)),
        target_value=representation.get("target_value", None),  # pyre-ignore[6]
    )


def _make_choice_param(
    name: str, representation: TParameterRepresentation, parameter_type: Optional[str]
) -> ChoiceParameter:
    assert "values" in representation, "Values are required for choice parameters."
    values = representation["values"]
    assert isinstance(values, list) and len(values) > 1, (
        f"Cannot parse parameter {name}: for choice parameters, json representation"
        " should include a list of two or more values."
    )
    return ChoiceParameter(
        name=name,
        parameter_type=_to_parameter_type(values, parameter_type, name, "values"),
        values=values,
        is_ordered=checked_cast(bool, representation.get("is_ordered", False)),
        is_fidelity=checked_cast(bool, representation.get("is_fidelity", False)),
        is_task=checked_cast(bool, representation.get("is_task", False)),
        target_value=representation.get("target_value", None),  # pyre-ignore[6]
    )


def _make_fixed_param(
    name: str, representation: Dict[str, TParamValue], parameter_type: Optional[str]
) -> FixedParameter:
    assert "value" in representation, "Value is required for fixed parameters."
    value = representation["value"]
    assert type(value) in PARAM_TYPES.values(), (
        f"Cannot parse fixed parameter {name}: for fixed parameters, json "
        "representation should include a single value."
    )
    return FixedParameter(
        name=name,
        parameter_type=_get_parameter_type(type(value))  # pyre-ignore[6]
        if parameter_type is None
        else _get_parameter_type(PARAM_TYPES[parameter_type]),  # pyre-ignore[6]
        value=value,
        is_fidelity=checked_cast(bool, representation.get("is_fidelity", False)),
        target_value=representation.get("target_value", None),
    )


def parameter_from_json(
    representation: Dict[str, Union[TParamValue, List[TParamValue]]]
) -> Parameter:
    """Instantiate a parameter from JSON representation."""
    name = representation["name"]
    assert isinstance(name, str), "Parameter name must be a string."
    parameter_class = representation["type"]
    assert (
        isinstance(parameter_class, str) and parameter_class in PARAM_CLASSES
    ), "Type in parameter JSON representation must be `range`, `choice`, or `fixed`."

    parameter_type = representation.get("value_type", None)
    if parameter_type is not None:
        assert isinstance(parameter_type, str) and parameter_type in PARAM_TYPES, (
            "Value type in parameter JSON representation must be 'int', 'float', "
            "'bool' or 'str'."
        )

    if parameter_class == "range":
        return _make_range_param(
            name=name, representation=representation, parameter_type=parameter_type
        )

    if parameter_class == "choice":
        return _make_choice_param(
            name=name, representation=representation, parameter_type=parameter_type
        )

    if parameter_class == "fixed":
        assert not any(isinstance(val, list) for val in representation.values())
        return _make_fixed_param(
            name=name,
            representation=cast(Dict[str, TParamValue], representation),
            parameter_type=parameter_type,
        )
    else:
        raise ValueError(  # pragma: no cover (this is unreachable)
            f"Unrecognized parameter type {parameter_class}."
        )


def constraint_from_str(
    representation: str, parameters: Dict[str, Parameter]
) -> ParameterConstraint:
    """Parse string representation of a parameter constraint."""
    tokens = representation.split()
    parameter_names = parameters.keys()
    order_const = len(tokens) == 3 and tokens[1] in COMPARISON_OPS
    sum_const = (
        len(tokens) >= 5 and len(tokens) % 2 == 1 and tokens[-2] in COMPARISON_OPS
    )
    if not (order_const or sum_const):
        raise ValueError(
            "Parameter constraint should be of form <parameter_name> >= "
            "<other_parameter_name> for order constraints or `<parameter_name> "
            "+ <other_parameter_name> >= x, where any number of terms can be "
            "added and `x` is a float bound. Acceptable comparison operators "
            'are ">=" and "<=".'
        )

    if len(tokens) == 3:  # Case "x1 >= x2" => order constraint.
        left, right = tokens[0], tokens[2]
        assert left in parameter_names, f"Parameter {left} not in {parameter_names}."
        assert right in parameter_names, f"Parameter {right} not in {parameter_names}."
        return (
            OrderConstraint(
                lower_parameter=parameters[left], upper_parameter=parameters[right]
            )
            if COMPARISON_OPS[tokens[1]] is ComparisonOp.LEQ
            else OrderConstraint(
                lower_parameter=parameters[right], upper_parameter=parameters[left]
            )
        )
    try:  # Case "x1 - 2*x2 + x3 >= 2" => parameter constraint.
        bound = float(tokens[-1])
    except ValueError:
        raise ValueError(f"Bound for the constraint must be a number; got {tokens[-1]}")
    if any(token[0] == "*" or token[-1] == "*" for token in tokens):
        raise ValueError(
            "A linear constraint should be the form a*x + b*y - c*z <= d"
            ", where a,b,c,d are float constants and x,y,z are parameters. "
            "There should be no space in each term around the operator * while "
            "there should be a single space around each operator +, -, <= and >=."
        )
    parameter_weight = {}
    comparison_multiplier = (
        1.0 if COMPARISON_OPS[tokens[-2]] is ComparisonOp.LEQ else -1.0
    )
    operator_sign = 1.0  # Determines whether the operator is + or -
    for idx, token in enumerate(tokens[:-2]):
        if idx % 2 == 0:
            split_token = token.split("*")
            parameter = ""  # Initializing the parameter
            multiplier = 1.0  # Initializing the multiplier
            if len(split_token) == 2:  # There is a non-unit multiplier
                try:
                    multiplier = float(split_token[0])
                except ValueError:
                    raise ValueError(
                        f"Multiplier should be float; got {split_token[0]}"
                    )
                parameter = split_token[1]
            elif len(split_token) == 1:  # The multiplier is either -1 or 1
                parameter = split_token[0]
                if parameter[0] == "-":  # The multiplier is -1
                    parameter = parameter[1:]
                    multiplier = -1.0
                else:
                    multiplier = 1.0
            assert (
                parameter in parameter_names
            ), f"Parameter {parameter} not in {parameter_names}."
            parameter_weight[parameter] = operator_sign * multiplier
        else:
            assert (
                token == "+" or token == "-"
            ), f"Expected a mixed constraint, found operator {token}."
            operator_sign = 1.0 if token == "+" else -1.0
    return ParameterConstraint(
        constraint_dict={
            p: comparison_multiplier * parameter_weight[p] for p in parameter_weight
        },
        bound=comparison_multiplier * bound,
    )


def outcome_constraint_from_str(representation: str) -> OutcomeConstraint:
    """Parse string representation of an outcome constraint."""
    tokens = representation.split()
    assert len(tokens) == 3 and tokens[1] in COMPARISON_OPS, (
        "Outcome constraint should be of form `metric_name >= x`, where x is a "
        "float bound and comparison operator is >= or <=."
    )
    op = COMPARISON_OPS[tokens[1]]
    rel = False
    try:
        bound_repr = tokens[2]
        if bound_repr[-1] == "%":
            rel = True
            bound_repr = bound_repr[:-1]
        bound = float(bound_repr)
    except ValueError:
        raise ValueError("Outcome constraint bound should be a float.")
    return OutcomeConstraint(Metric(name=tokens[0]), op=op, bound=bound, relative=rel)


def make_experiment(
    parameters: List[TParameterRepresentation],
    name: Optional[str] = None,
    objective_name: Optional[str] = None,
    minimize: bool = False,
    parameter_constraints: Optional[List[str]] = None,
    outcome_constraints: Optional[List[str]] = None,
    status_quo: Optional[TParameterization] = None,
    experiment_type: Optional[str] = None,
) -> Experiment:
    """Instantiation wrapper that allows for creation of SimpleExperiment
    without importing or instantiating any Ax classes."""

    exp_parameters: List[Parameter] = [parameter_from_json(p) for p in parameters]
    status_quo_arm = None if status_quo is None else Arm(parameters=status_quo)
    parameter_map = {p.name: p for p in exp_parameters}
    ocs = [outcome_constraint_from_str(c) for c in (outcome_constraints or [])]
    if status_quo_arm is None and any(oc.relative for oc in ocs):
        raise ValueError("Must set status_quo to have relative outcome constraints.")
    return Experiment(
        name=name,
        search_space=SearchSpace(
            parameters=exp_parameters,
            parameter_constraints=None
            if parameter_constraints is None
            else [constraint_from_str(c, parameter_map) for c in parameter_constraints],
        ),
        optimization_config=OptimizationConfig(
            objective=Objective(
                metric=Metric(
                    name=objective_name or DEFAULT_OBJECTIVE_NAME,
                    lower_is_better=minimize,
                ),
                minimize=minimize,
            ),
            outcome_constraints=ocs,
        ),
        status_quo=status_quo_arm,
        experiment_type=experiment_type,
    )


def raw_data_to_evaluation(
    raw_data: TEvaluationOutcome,
    objective_name: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> TEvaluationOutcome:
    """Format the trial evaluation data to a standard `TTrialEvaluation`
    (mapping from metric names to a tuple of mean and SEM) representation, or
    to a TFidelityTrialEvaluation.

    Note: this function expects raw_data to be data for a `Trial`, not a
    `BatchedTrial`.
    """
    if isinstance(raw_data, dict):
        if any(isinstance(x, dict) for x in raw_data.values()):  # pragma: no cover
            raise ValueError("Raw data is expected to be just for one arm.")
        return raw_data
    elif isinstance(raw_data, list):
        return raw_data
    elif isinstance(raw_data, tuple):
        return {objective_name: raw_data}
    elif isinstance(raw_data, (float, int)):
        return {objective_name: (raw_data, None)}
    elif isinstance(raw_data, (np.float32, np.float64, np.int32, np.int64)):
        return {objective_name: (numpy_type_to_python_type(raw_data), None)}
    else:
        raise ValueError(
            "Raw data has an invalid type. The data must either be in the form "
            "of a dictionary of metric names to mean, sem tuples, "
            "or a single mean, sem tuple, or a single mean."
        )


def data_from_evaluations(
    evaluations: Dict[str, TEvaluationOutcome],
    trial_index: int,
    sample_sizes: Dict[str, int],
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> Data:
    """Transforms evaluations into Ax Data.

    Each evaluation is either a trial evaluation: {metric_name -> (mean, SEM)}
    or a fidelity trial evaluation for multi-fidelity optimizations:
    [(fidelities, {metric_name -> (mean, SEM)})].

    Args:
        evalutions: Mapping from arm name to evaluation.
        trial_index: Index of the trial, for which the evaluations are.
        sample_sizes: Number of samples collected for each arm, may be empty
            if unavailable.
        start_time: Optional start time of run of the trial that produced this
            data, in milliseconds.
        end_time: Optional end time of run of the trial that produced this
            data, in milliseconds.
    """
    if all(isinstance(evaluations[x], dict) for x in evaluations.keys()):
        # All evaluations are no-fidelity evaluations.
        data = Data.from_evaluations(
            evaluations=cast(Dict[str, TTrialEvaluation], evaluations),
            trial_index=trial_index,
            sample_sizes=sample_sizes,
            start_time=start_time,
            end_time=end_time,
        )
    elif all(isinstance(evaluations[x], list) for x in evaluations.keys()):
        # All evaluations are with-fidelity evaluations.
        data = Data.from_fidelity_evaluations(
            evaluations=cast(Dict[str, TFidelityTrialEvaluation], evaluations),
            trial_index=trial_index,
            sample_sizes=sample_sizes,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        raise ValueError(  # pragma: no cover
            "Evaluations included a mixture of no-fidelity and with-fidelity "
            "evaluations, which is not currently supported."
        )
    return data

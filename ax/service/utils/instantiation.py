#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict, List, Optional, Union, cast

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
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.core.search_space import SearchSpace
from ax.core.simple_experiment import DEFAULT_OBJECTIVE_NAME
from ax.core.types import ComparisonOp, TParamValue
from ax.utils.common.typeutils import not_none


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
        "should include a list of two values, lower and upper bounds of the bounds."
    )
    return RangeParameter(
        name=name,
        parameter_type=_to_parameter_type(bounds, parameter_type, name, "bounds"),
        lower=bounds[0],
        upper=bounds[1],
        log_scale=representation.get("log_scale", False),
    )


def _make_choice_param(
    name: str, representation: TParameterRepresentation, parameter_type: Optional[str]
) -> ChoiceParameter:
    assert "values" in representation, "Values are required for choice parameters."
    values = representation["values"]
    assert isinstance(values, list) and len(values) > 1, (
        f"Cannot parse parameter {name}: for choice parameters, json representation"
        " should include a list values, lower and upper bounds of the range."
    )
    return ChoiceParameter(
        name=name,
        parameter_type=_to_parameter_type(values, parameter_type, name, "values"),
        values=values,
        is_ordered=representation.get("is_ordered", False),
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
            "Value type in parameter JSON representation must be `int`, `float`, "
            "`bool` or `str`."
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
        raise ValueError(f"Unrecognized parameter type {parameter_class}.")


def constraint_from_str(
    representation: str, parameters: Dict[str, Parameter]
) -> ParameterConstraint:
    """Parse string representation of a parameter constraint."""
    tokens = representation.split()
    assert (len(tokens) == 3 and tokens[1] in COMPARISON_OPS) or (
        len(tokens) == 5 and tokens[3] in COMPARISON_OPS
    ), (
        "Parameter constraint should be of form `metric_name` >= `other_metric_name` "
        "for order constraints or `metric_name` + `other_metric_name` >= x, where "
        "x is a float bound, and acceptable comparison operators are >= and <=."
    )
    one, other = tokens[0], tokens[2]
    parameter_names = parameters.keys()
    assert one in parameter_names, f"Parameter {one} not in {parameter_names}."
    assert other in parameter_names, f"Parameter {other} not in {parameter_names}."

    if len(tokens) == 3:  # Case "x1 >= x2" => order constraint.
        return (
            OrderConstraint(
                lower_parameter=parameters[one], upper_parameter=parameters[other]
            )
            if COMPARISON_OPS[tokens[1]] is ComparisonOp.LEQ
            else OrderConstraint(
                lower_parameter=parameters[other], upper_parameter=parameters[one]
            )
        )

    try:  # Case "x1 + x3 >= 2" => sum constraint.
        bound = float(tokens[4])
    except ValueError:
        raise ValueError(f"Bound for sum constraint must be a number; got {tokens[4]}")
    return SumConstraint(
        parameters=[parameters[one], parameters[other]],
        is_upper_bound=COMPARISON_OPS[tokens[3]] is ComparisonOp.LEQ,
        bound=bound,
    )


def outcome_constraint_from_str(representation: str) -> OutcomeConstraint:
    """Parse string representation of an outcome constraint."""
    tokens = representation.split()
    assert len(tokens) == 3 and tokens[1] in COMPARISON_OPS, (
        "Outcome constraint should be of form `metric_name >= x`, where x is a "
        "float bound and comparison operator is >= or <=."
    )
    op = COMPARISON_OPS[tokens[1]]
    try:
        bound = float(tokens[2])
    except ValueError:
        raise ValueError("Outcome constraint bound should be a float.")
    return OutcomeConstraint(Metric(name=tokens[0]), op=op, bound=bound, relative=False)


def make_experiment(
    parameters: List[TParameterRepresentation],
    name: Optional[str] = None,
    objective_name: Optional[str] = None,
    minimize: bool = False,
    parameter_constraints: Optional[List[str]] = None,
    outcome_constraints: Optional[List[str]] = None,
) -> Experiment:
    """Instantiation wrapper that allows for creation of SimpleExperiment without
    importing or instantiating any Ax classes."""

    exp_parameters: List[Parameter] = [parameter_from_json(p) for p in parameters]
    parameter_map = {p.name: p for p in exp_parameters}
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
                metric=Metric(name=objective_name or DEFAULT_OBJECTIVE_NAME),
                minimize=minimize,
            ),
            outcome_constraints=None
            if outcome_constraints is None
            else [outcome_constraint_from_str(c) for c in outcome_constraints],
        ),
    )

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Union, cast

import numpy as np
from ax.core.abstract_data import AbstractDataFrameData
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import DataType, Experiment
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric
from ax.core.objective import Objective, MultiObjective
from ax.core.optimization_config import (
    ObjectiveThreshold,
    OptimizationConfig,
    MultiObjectiveOptimizationConfig,
)
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
    TMapTrialEvaluation,
    TParameterization,
    TParamValue,
    TTrialEvaluation,
)
from ax.exceptions.core import UnsupportedError
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import (
    checked_cast,
    checked_cast_to_tuple,
    not_none,
    numpy_type_to_python_type,
)

logger = get_logger(__name__)


"""Utilities for RESTful-like instantiation of Ax classes needed in AxClient."""


TParameterRepresentation = Dict[str, Union[TParamValue, List[TParamValue]]]
PARAM_CLASSES = ["range", "choice", "fixed"]
PARAM_TYPES = {"int": int, "float": float, "bool": bool, "str": str}
COMPARISON_OPS = {"<=": ComparisonOp.LEQ, ">=": ComparisonOp.GEQ}
EXPECTED_KEYS_IN_PARAM_REPR = {
    "name",
    "type",
    "values",
    "bounds",
    "value",
    "value_type",
    "log_scale",
    "target_value",
    "is_fidelity",
    "is_ordered",
    "is_task",
    "digits",
}


class MetricObjective(enum.Enum):
    MINIMIZE = enum.auto()
    MAXIMIZE = enum.auto()


@dataclass
class ObjectiveProperties:
    minimize: bool
    threshold: Optional[float] = None


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
        logger.info(
            f"Inferred value type of {parameter_type} for parameter {param_name}. "
            "If that is not the expected value type, you can explicity specify "
            "'value_type' ('int', 'float', 'bool' or 'str') in parameter dict."
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
        digits=representation.get("digits", None),  # pyre-ignore[6]
        is_fidelity=checked_cast(bool, representation.get("is_fidelity", False)),
        # pyre-fixme[6]: Expected `Union[None, bool, float, int, str]` for 8th param
        #  but got `Union[None, List[Union[None, bool, float, int, str]], bool, float,
        #  int, str]`.
        target_value=representation.get("target_value", None),
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
    if "parameter_type" in representation:
        raise ValueError(
            "'parameter_type' is not an expected key in parameter dictionary. "
            "If you are looking to specify the type of values that this "
            "parameter should take, use 'value_type' (expects 'int', 'float', "
            "'str' or 'bool')."
        )
    unexpected_keys = set(representation.keys()) - EXPECTED_KEYS_IN_PARAM_REPR
    if unexpected_keys:
        raise ValueError(
            f"Unexpected keys {unexpected_keys} in parameter representation."
            f"Exhaustive set of expected keys: {EXPECTED_KEYS_IN_PARAM_REPR}."
        )
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


def objective_threshold_constraint_from_str(
    representation: str,
) -> ObjectiveThreshold:
    oc = outcome_constraint_from_str(representation)
    return ObjectiveThreshold(
        metric=oc.metric.clone(),
        bound=oc.bound,
        relative=oc.relative,
        op=oc.op,
    )


def make_objectives(objectives: Dict[str, str]) -> List[Objective]:
    try:
        return [
            Objective(
                metric=Metric(
                    name=metric_name,
                ),
                minimize=(
                    MetricObjective[min_or_max.upper()] == MetricObjective.MINIMIZE
                ),
            )
            for metric_name, min_or_max in objectives.items()
        ]
    except KeyError as k:
        raise ValueError(
            f"Objective values should specify '{MetricObjective.MINIMIZE.name.lower()}'"
            f" or '{MetricObjective.MAXIMIZE.name.lower()}', got {k} in"
            f" objectives({objectives})"
        )


def make_outcome_constraints(
    outcome_constraints: List[str], status_quo_defined: bool
) -> List[OutcomeConstraint]:

    typed_outcome_constraints = [
        outcome_constraint_from_str(c) for c in outcome_constraints
    ]

    if status_quo_defined is False and any(
        oc.relative for oc in typed_outcome_constraints
    ):
        raise ValueError("Must set status_quo to have relative outcome constraints.")

    return typed_outcome_constraints


def make_objective_thresholds(
    objective_thresholds: List[str], status_quo_defined: bool
) -> List[ObjectiveThreshold]:

    typed_objective_thresholds = (
        [objective_threshold_constraint_from_str(c) for c in objective_thresholds]
        if objective_thresholds is not None
        else []
    )

    if status_quo_defined is False and any(
        oc.relative for oc in typed_objective_thresholds
    ):
        raise ValueError("Must set status_quo to have relative objective thresholds.")

    return typed_objective_thresholds


def optimization_config_from_objectives(
    objectives: List[Objective],
    objective_thresholds: List[ObjectiveThreshold],
    outcome_constraints: List[OutcomeConstraint],
) -> OptimizationConfig:
    """Parse objectives and constraints to define optimization config.

    The resulting optimization config will be regular single-objective config
    if `objectives` is a list of one element and a multi-objective config
    otherwise.

    NOTE: If passing in multiple objectives, `objective_thresholds` must be a
    non-empty list definining constraints for each objective.
    """
    if len(objectives) == 1:
        if objective_thresholds:
            raise ValueError(
                "Single-objective optimizations must not specify objective thresholds."
            )
        return OptimizationConfig(
            objective=objectives[0],
            outcome_constraints=outcome_constraints,
        )
    else:
        objective_names = {m.metric.name for m in objectives}
        threshold_names = {oc.metric.name for oc in objective_thresholds}
        if objective_names != threshold_names:
            logger.info(
                (
                    "Due to non-specification, we will use the heuristic for selecting "
                    "thresholds for these metrics: %s"
                ),
                objective_names.symmetric_difference(threshold_names),
            )

        return MultiObjectiveOptimizationConfig(
            objective=MultiObjective(objectives=objectives),
            outcome_constraints=outcome_constraints,
            objective_thresholds=objective_thresholds,
        )


def make_optimization_config(
    objectives: Dict[str, str],
    objective_thresholds: List[str],
    outcome_constraints: List[str],
    status_quo_defined: bool,
) -> OptimizationConfig:

    return optimization_config_from_objectives(
        make_objectives(objectives),
        make_objective_thresholds(objective_thresholds, status_quo_defined),
        make_outcome_constraints(outcome_constraints, status_quo_defined),
    )


def make_search_space(
    parameters: List[TParameterRepresentation],
    parameter_constraints: List[str],
) -> SearchSpace:

    typed_parameters = [parameter_from_json(p) for p in parameters]
    parameter_map = {p.name: p for p in typed_parameters}

    typed_parameter_constraints = [
        constraint_from_str(c, parameter_map) for c in parameter_constraints
    ]

    return SearchSpace(
        parameters=typed_parameters,
        parameter_constraints=typed_parameter_constraints,
    )


def make_experiment(
    parameters: List[TParameterRepresentation],
    name: Optional[str] = None,
    parameter_constraints: Optional[List[str]] = None,
    outcome_constraints: Optional[List[str]] = None,
    status_quo: Optional[TParameterization] = None,
    experiment_type: Optional[str] = None,
    tracking_metric_names: Optional[List[str]] = None,
    # Single-objective optimization arguments:
    objective_name: Optional[str] = None,
    minimize: bool = False,
    # Multi-objective optimization arguments:
    objectives: Optional[Dict[str, str]] = None,
    objective_thresholds: Optional[List[str]] = None,
    support_intermediate_data: bool = False,
    immutable_search_space_and_opt_config: bool = True,
    is_test: bool = False,
) -> Experiment:
    """Instantiation wrapper that allows for Ax `Experiment` creation
    without importing or instantiating any Ax classes.

    Args:
        parameters: List of dictionaries representing parameters in the
            experiment search space.
            Required elements in the dictionaries are:
            1. "name" (name of parameter, string),
            2. "type" (type of parameter: "range", "fixed", or "choice", string),
            and one of the following:
            3a. "bounds" for range parameters (list of two values, lower bound
            first),
            3b. "values" for choice parameters (list of values), or
            3c. "value" for fixed parameters (single value).
            Optional elements are:
            1. "log_scale" (for float-valued range parameters, bool),
            2. "value_type" (to specify type that values of this parameter should
            take; expects "float", "int", "bool" or "str"),
            3. "is_fidelity" (bool) and "target_value" (float) for fidelity
            parameters,
            4. "is_ordered" (bool) for choice parameters,
            5. "is_task" (bool) for task parameters, and
            6. "digits" (int) for float-valued range parameters.
        name: Name of the experiment to be created.
        parameter_constraints: List of string representation of parameter
            constraints, such as "x3 >= x4" or "-x3 + 2*x4 - 3.5*x5 >= 2". For
            the latter constraints, any number of arguments is accepted, and
            acceptable operators are "<=" and ">=".
        parameter_constraints: List of string representation of parameter
                constraints, such as "x3 >= x4" or "-x3 + 2*x4 - 3.5*x5 >= 2". For
                the latter constraints, any number of arguments is accepted, and
                acceptable operators are "<=" and ">=".
        outcome_constraints: List of string representation of outcome
            constraints of form "metric_name >= bound", like "m1 <= 3."
        status_quo: Parameterization of the current state of the system.
            If set, this will be added to each trial to be evaluated alongside
            test configurations.
        experiment_type: String indicating type of the experiment (e.g. name of
            a product in which it is used), if any.
        tracking_metric_names: Names of additional tracking metrics not used for
            optimization.
        objective_name: Name of the metric used as objective in this experiment,
            if experiment is single-objective optimization.
        minimize: Whether this experiment represents a minimization problem, if
            experiment is a single-objective optimization.
        objectives: Mapping from an objective name to "minimize" or "maximize"
            representing the direction for that objective. Used only for
            multi-objective optimization experiments.
        objective_thresholds: A list of objective threshold constraints for multi-
            objective optimization, in the same string format as `outcome_constraints`
            argument.
        support_intermediate_data: Whether trials may report metrics results for
            incomplete runs.
        immutable_search_space_and_opt_config: Whether it's possible to update the
            search space and optimization config on this experiment after creation.
            Defaults to True. If set to True, we won't store or load copies of the
            search space and optimization config on each generator run, which will
            improve storage performance.
        is_test: Whether this experiment will be a test experiment (useful for
            marking test experiments in storage etc). Defaults to False.
    """
    if objective_name is not None and (
        objectives is not None or objective_thresholds is not None
    ):
        raise UnsupportedError(
            "Ambiguous objective definition: for single-objective optimization "
            "`objective_name` and `minimize` arguments expected. For multi-objective "
            "optimization `objectives` and `objective_thresholds` arguments expected."
        )

    status_quo_arm = None if status_quo is None else Arm(parameters=status_quo)

    # TODO(jej): Needs to be decided per-metric when supporting heterogenous data.
    metric_cls = MapMetric if support_intermediate_data else Metric
    if objectives is None:
        optimization_config = OptimizationConfig(
            objective=Objective(
                metric=metric_cls(
                    name=objective_name or DEFAULT_OBJECTIVE_NAME,
                    lower_is_better=minimize,
                ),
                minimize=minimize,
            ),
            outcome_constraints=make_outcome_constraints(
                outcome_constraints or [], status_quo_arm is not None
            ),
        )
    else:
        optimization_config = make_optimization_config(
            objectives,
            objective_thresholds or [],
            outcome_constraints or [],
            status_quo_arm is not None,
        )

    tracking_metrics = (
        None
        if tracking_metric_names is None
        else [Metric(name=metric_name) for metric_name in tracking_metric_names]
    )

    default_data_type = (
        DataType.MAP_DATA if support_intermediate_data else DataType.DATA
    )

    immutable_ss_and_oc = immutable_search_space_and_opt_config
    properties = (
        {}
        if not immutable_search_space_and_opt_config
        else {Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF.value: immutable_ss_and_oc}
    )

    return Experiment(
        name=name,
        search_space=make_search_space(parameters, parameter_constraints or []),
        optimization_config=optimization_config,
        status_quo=status_quo_arm,
        experiment_type=experiment_type,
        tracking_metrics=tracking_metrics,
        default_data_type=default_data_type,
        properties=properties,
        is_test=is_test,
    )


def raw_data_to_evaluation(
    raw_data: TEvaluationOutcome,
    metric_names: List[str],
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> TEvaluationOutcome:
    """Format the trial evaluation data to a standard `TTrialEvaluation`
    (mapping from metric names to a tuple of mean and SEM) representation, or
    to a TMapTrialEvaluation.

    Note: this function expects raw_data to be data for a `Trial`, not a
    `BatchedTrial`.
    """
    if isinstance(raw_data, dict):
        if any(isinstance(x, dict) for x in raw_data.values()):  # pragma: no cover
            raise ValueError("Raw data is expected to be just for one arm.")
        for metric_name, dat in raw_data.items():
            if not isinstance(dat, tuple):
                if not isinstance(dat, (float, int)):
                    raise ValueError(
                        "Raw data for an arm is expected to either be a tuple of "
                        "numerical mean and SEM or just a numerical mean."
                        f"Got: {dat} for metric '{metric_name}'."
                    )
                raw_data[metric_name] = (float(dat), None)
        return raw_data
    elif len(metric_names) > 1:
        raise ValueError(
            "Raw data must be a dictionary of metric names to mean "
            "for multi-objective optimizations."
        )
    elif isinstance(raw_data, list):
        return raw_data
    elif isinstance(raw_data, tuple):
        return {metric_names[0]: raw_data}
    elif isinstance(raw_data, (float, int)):
        return {metric_names[0]: (raw_data, None)}
    elif isinstance(raw_data, (np.float32, np.float64, np.int32, np.int64)):
        return {metric_names[0]: (numpy_type_to_python_type(raw_data), None)}
    else:
        raise ValueError(
            "Raw data has an invalid type. The data must either be in the form "
            "of a dictionary of metric names to mean, sem tuples, "
            "or a single mean, sem tuple, or a single mean."
        )


def data_and_evaluations_from_raw_data(
    raw_data: Dict[str, TEvaluationOutcome],
    metric_names: List[str],
    trial_index: int,
    sample_sizes: Dict[str, int],
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> Tuple[Dict[str, TEvaluationOutcome], AbstractDataFrameData]:
    """Transforms evaluations into Ax Data.

    Each evaluation is either a trial evaluation: {metric_name -> (mean, SEM)}
    or a fidelity trial evaluation for multi-fidelity optimizations:
    [(fidelities, {metric_name -> (mean, SEM)})].

    Args:
        raw_data: Mapping from arm name to raw_data.
        metric_names: Names of metrics used to transform raw data to evaluations.
        trial_index: Index of the trial, for which the evaluations are.
        sample_sizes: Number of samples collected for each arm, may be empty
            if unavailable.
        start_time: Optional start time of run of the trial that produced this
            data, in milliseconds.
        end_time: Optional end time of run of the trial that produced this
            data, in milliseconds.
    """
    evaluations = {
        arm_name: raw_data_to_evaluation(
            raw_data=raw_data[arm_name],
            metric_names=metric_names,
            start_time=start_time,
            end_time=end_time,
        )
        for arm_name in raw_data
    }
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
        # All evaluations are map evaluations.
        data = MapData.from_map_evaluations(
            evaluations=cast(Dict[str, TMapTrialEvaluation], evaluations),
            trial_index=trial_index,
        )
    else:
        raise ValueError(  # pragma: no cover
            "Evaluations included a mixture of no-fidelity and with-fidelity "
            "evaluations, which is not currently supported."
        )
    return evaluations, data


def build_objective_threshold(
    objective: str, objective_properties: ObjectiveProperties
) -> str:
    """
    Constructs constraint string for an objective threshold interpretable
    by `make_experiment()`

    Args:
        objective: Name of the objective
        objective_properties: Object containing:
            minimize: Whether this experiment represents a minimization problem.
            threshold: The bound in the objective's threshold constraint.
    """
    operator = "<=" if objective_properties.minimize else ">="
    return f"{objective} {operator} {objective_properties.threshold}"

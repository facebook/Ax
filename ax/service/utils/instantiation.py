#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import enum
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

from logging import Logger
from typing import Any, Union

from ax.core.arm import Arm
from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
from ax.core.experiment import DataType, Experiment
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import MultiObjective, Objective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    Parameter,
    PARAMETER_PYTHON_TYPE_MAP,
    ParameterType,
    RangeParameter,
    TParameterType,
)
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    validate_constraint_parameters,
)
from ax.core.runner import Runner
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace
from ax.core.types import ComparisonOp, TParameterization, TParamValue
from ax.exceptions.core import UnsupportedError
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import (
    assert_is_instance_of_tuple,
    assert_is_instance_optional,
)
from pyre_extensions import assert_is_instance, none_throws

DEFAULT_OBJECTIVE_NAME = "objective"

logger: Logger = get_logger(__name__)


"""Utilities for RESTful-like instantiation of Ax classes needed in AxClient."""


TParameterRepresentation = dict[
    str, Union[TParamValue, Sequence[TParamValue], dict[str, list[str]]]
]
PARAM_CLASSES = ["range", "choice", "fixed"]
PARAM_TYPES = {"int": int, "float": float, "bool": bool, "str": str}
COMPARISON_OPS: dict[str, ComparisonOp] = {
    "<=": ComparisonOp.LEQ,
    ">=": ComparisonOp.GEQ,
}
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
    "sort_values",
    "is_ordered",
    "is_task",
    "digits",
    "dependents",
}


class MetricObjective(enum.Enum):
    MINIMIZE = enum.auto()
    MAXIMIZE = enum.auto()


@dataclass
class ObjectiveProperties:
    r"""Class that holds properties of objective functions. Can be used to define an
    the `objectives` argument of ax_client.create_experiment, e.g.:

        ax_client.create_experiment(
            name="moo_experiment",
            parameters=[...],
            objectives={
                # `threshold` arguments are optional
                "a": ObjectiveProperties(minimize=False, threshold=ref_point[0]),
                "b": ObjectiveProperties(minimize=False, threshold=ref_point[1]),
            },
        )

    Args:
        - minimize: Boolean indicating whether the objective is to be minimized
            or maximized.
        - threshold: Optional `float` representing the smallest objective value
            (resp. largest if minimize=True) that is considered valuable in the context
            of multi-objective optimization. In BoTorch and in the literature, this is
            also known as an element of the reference point vector that defines the
            hyper-volume of the Pareto front.
    """

    minimize: bool
    threshold: float | None = None


@dataclass(frozen=True)
class FixedFeatures:
    """Class for representing fixed features via the Service API."""

    parameters: TParameterization
    trial_index: int | None = None


class InstantiationBase:
    """
    This is a lightweight stateless class that bundles together instantiation utils.
    It is used both on its own and as a mixin to AxClient, with the intent that
    these methods can be overridden by its subclasses for specific use cases.
    """

    @staticmethod
    def _get_deserialized_metric_kwargs(
        metric_class: type[Metric],
        name: str,
        metric_definitions: dict[str, dict[str, Any]] | None,
    ) -> tuple[type[Metric], dict[str, Any]]:
        """Get metric kwargs from metric_definitions if available and deserialize
        if so.  Deserialization is necessary because they were serialized on creation"""
        # deepcopy is used because of subsequent modifications to the dict
        metric_kwargs = deepcopy((metric_definitions or {}).get(name, {}))
        metric_class = metric_kwargs.pop("metric_class", metric_class)
        # this is necessary before deserialization because name will be required
        metric_kwargs["name"] = metric_kwargs.get("name", name)
        metric_kwargs = metric_class.deserialize_init_args(metric_kwargs)
        return metric_class, metric_kwargs

    @classmethod
    def _make_metric(
        cls,
        name: str,
        lower_is_better: bool | None = None,
        metric_class: type[Metric] = Metric,
        for_opt_config: bool = False,
        metric_definitions: dict[str, dict[str, Any]] | None = None,
    ) -> Metric:
        if " " in name:
            raise ValueError(
                "Metric names cannot contain spaces when used with AxClient. Got "
                f"{name!r}."
            )

        metric_definitions = metric_definitions or {}

        metric_class, kwargs = cls._get_deserialized_metric_kwargs(
            name=name,
            metric_definitions=metric_definitions,
            metric_class=metric_class,
        )
        # avoid conflict is lower_is_better is specified in kwargs
        kwargs["lower_is_better"] = kwargs.get("lower_is_better", lower_is_better)
        return metric_class(
            **kwargs,
        )

    @staticmethod
    def _get_parameter_type(python_type: TParameterType) -> ParameterType:
        for param_type, py_type in PARAMETER_PYTHON_TYPE_MAP.items():
            if py_type is python_type:
                return param_type
        raise ValueError(f"No AE parameter type corresponding to {python_type}.")

    @classmethod
    def _to_parameter_type(
        cls,
        vals: list[TParamValue],
        typ: str | None,
        param_name: str,
        field_name: str,
    ) -> ParameterType:
        if typ is None:
            typ = type(none_throws(vals[0]))
            parameter_type = cls._get_parameter_type(typ)  # pyre-ignore[6]
            assert all(isinstance(x, typ) for x in vals), (
                f"Values in `{field_name}` not of the same type and no "
                "`value_type` was explicitly specified; cannot infer "
                f"value type for parameter {param_name}."
            )
            logger.info(
                f"Inferred value type of {parameter_type} for parameter {param_name}. "
                "If that is not the expected value type, you can explicitly specify "
                "'value_type' ('int', 'float', 'bool' or 'str') in parameter dict."
            )
            return parameter_type
        return cls._get_parameter_type(PARAM_TYPES[typ])  # pyre-ignore[6]

    @classmethod
    def _make_range_param(
        cls,
        name: str,
        representation: TParameterRepresentation,
        parameter_type: str | None,
    ) -> RangeParameter:
        assert "bounds" in representation, "Bounds are required for range parameters."
        bounds = representation["bounds"]
        assert isinstance(bounds, list) and len(bounds) == 2, (
            f"Cannot parse parameter {name}: for range parameters, json representation "
            "should include a list of two values, lower and upper bounds of the range."
        )
        return RangeParameter(
            name=name,
            parameter_type=cls._to_parameter_type(
                bounds, parameter_type, name, "bounds"
            ),
            lower=assert_is_instance_of_tuple(bounds[0], (float, int)),
            upper=assert_is_instance_of_tuple(bounds[1], (float, int)),
            log_scale=assert_is_instance(representation.get("log_scale", False), bool),
            digits=representation.get("digits", None),  # pyre-ignore[6]
            is_fidelity=assert_is_instance(
                representation.get("is_fidelity", False), bool
            ),
            # pyre-ignore[6]: Expected `Union[None, bool, float, int, str]`
            #  for 8th param but got `Union[None, List[
            #  Union[None, bool, float, int, str]], bool, float, int, str]`.
            target_value=representation.get("target_value", None),
        )

    @classmethod
    def _make_choice_param(
        cls,
        name: str,
        representation: TParameterRepresentation,
        parameter_type: str | None,
    ) -> ChoiceParameter:
        values = representation["values"]
        assert isinstance(values, list) and len(values) > 1, (
            f"Cannot parse parameter {name}: for choice parameters, json representation"
            " should include a list of two or more values."
        )
        return ChoiceParameter(
            name=name,
            parameter_type=cls._to_parameter_type(
                values, parameter_type, name, "values"
            ),
            values=values,
            is_ordered=assert_is_instance_optional(
                representation.get("is_ordered"), bool
            ),
            is_fidelity=assert_is_instance(
                representation.get("is_fidelity", False), bool
            ),
            is_task=assert_is_instance(representation.get("is_task", False), bool),
            target_value=representation.get("target_value", None),  # pyre-ignore[6]
            sort_values=assert_is_instance_optional(
                representation.get("sort_values", None), bool
            ),
            dependents=assert_is_instance_optional(
                representation.get("dependents", None), dict
            ),
        )

    @classmethod
    def _make_fixed_param(
        cls,
        name: str,
        representation: TParameterRepresentation,
        parameter_type: str | None,
    ) -> FixedParameter:
        assert "value" in representation, "Value is required for fixed parameters."
        value = representation["value"]
        assert type(value) in PARAM_TYPES.values(), (
            f"Cannot parse fixed parameter {name}: for fixed parameters, json "
            "representation should include a single value."
        )
        return FixedParameter(
            name=name,
            parameter_type=(
                cls._get_parameter_type(type(value))  # pyre-ignore[6]
                if parameter_type is None
                # pyre-ignore[6]
                else cls._get_parameter_type(PARAM_TYPES[parameter_type])
            ),
            value=value,  # pyre-ignore[6]
            is_fidelity=assert_is_instance(
                representation.get("is_fidelity", False), bool
            ),
            target_value=representation.get("target_value", None),  # pyre-ignore[6]
            dependents=representation.get("dependents", None),  # pyre-ignore[6]
        )

    @classmethod
    def parameter_from_json(
        cls,
        representation: TParameterRepresentation,
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
        assert isinstance(parameter_class, str) and parameter_class in PARAM_CLASSES, (
            "Type in parameter JSON representation must be "
            "`range`, `choice`, or `fixed`."
        )

        parameter_type = representation.get("value_type", None)
        if parameter_type is not None:
            assert isinstance(parameter_type, str) and parameter_type in PARAM_TYPES, (
                "Value type in parameter JSON representation must be 'int', 'float', "
                "'bool' or 'str'."
            )

        if " " in name:
            raise ValueError(
                "Parameter names cannot contain spaces when used with AxClient. Got "
                f"{name!r}."
            )

        if parameter_class == "range":
            return cls._make_range_param(
                name=name,
                representation=representation,
                parameter_type=parameter_type,
            )

        if parameter_class == "choice":
            assert (
                "values" in representation
            ), "Values are required for choice parameters."
            values = representation["values"]
            if isinstance(values, list) and len(values) == 1:
                logger.info(
                    f"Choice parameter {name} contains only one value, converting to a"
                    + " fixed parameter instead."
                )
                # update the representation to a fixed parameter class
                parameter_class = "fixed"
                representation["type"] = parameter_class
                representation["value"] = values[0]
                del representation["values"]
            else:
                return cls._make_choice_param(
                    name=name,
                    representation=representation,
                    parameter_type=parameter_type,
                )

        if parameter_class == "fixed":
            assert not any(isinstance(val, list) for val in representation.values())
            return cls._make_fixed_param(
                name=name,
                representation=representation,
                parameter_type=parameter_type,
            )
        else:
            raise ValueError(f"Unrecognized parameter type {parameter_class}.")

    @staticmethod
    def constraint_from_str(
        representation: str, parameters: dict[str, Parameter]
    ) -> ParameterConstraint:
        """Parse string representation of a parameter constraint."""
        tokens = representation.split()
        parameter_names = parameters.keys()
        try:
            float(tokens[-1])
            last_token_is_numeric = True
        except ValueError:
            last_token_is_numeric = False
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

        # Case "x1 >= x2" => order constraint.
        if len(tokens) == 3 and not last_token_is_numeric:
            left, right = tokens[0], tokens[2]
            assert (
                left in parameter_names
            ), f"Parameter {left} not in {parameter_names}."
            assert (
                right in parameter_names
            ), f"Parameter {right} not in {parameter_names}."
            validate_constraint_parameters(
                parameters=[parameters[left], parameters[right]]
            )
            return (
                OrderConstraint(
                    lower_parameter=parameters[left], upper_parameter=parameters[right]
                )
                if COMPARISON_OPS[tokens[1]] is ComparisonOp.LEQ
                else OrderConstraint(
                    lower_parameter=parameters[right], upper_parameter=parameters[left]
                )
            )
        if not last_token_is_numeric:
            raise ValueError(
                f"Bound for the constraint must be a number; got {tokens[-1]}"
            )
        bound = float(tokens[-1])
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
        # tokens are alternating monomials and operators
        for idx, token in enumerate(tokens[:-2]):
            # for monomials
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
                validate_constraint_parameters(parameters=[parameters[parameter]])

                parameter_weight[parameter] = operator_sign * multiplier
            # for operators
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

    @classmethod
    def outcome_constraint_from_str(
        cls,
        representation: str,
        metric_definitions: dict[str, dict[str, Any]] | None = None,
    ) -> OutcomeConstraint:
        """Parse string representation of an outcome constraint."""
        tokens = representation.split()
        assert len(tokens) == 3 and tokens[1] in COMPARISON_OPS, (
            f"Outcome constraint '{representation}' should be of "
            "form `metric_name >= x`, where x is a "
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
            raise ValueError(
                f"Outcome constraint bound should be a float for '{representation}'."
            )
        return OutcomeConstraint(
            cls._make_metric(
                name=tokens[0],
                for_opt_config=True,
                metric_definitions=metric_definitions,
                lower_is_better=op is ComparisonOp.LEQ,
            ),
            op=op,
            bound=bound,
            relative=rel,
        )

    @classmethod
    def objective_threshold_constraint_from_str(
        cls,
        representation: str,
        metric_definitions: dict[str, dict[str, Any]] | None = None,
    ) -> ObjectiveThreshold:
        oc = cls.outcome_constraint_from_str(
            representation, metric_definitions=metric_definitions
        )
        return ObjectiveThreshold(
            metric=oc.metric.clone(),
            bound=oc.bound,
            relative=oc.relative,
            op=oc.op,
        )

    @classmethod
    def make_objectives(
        cls,
        objectives: dict[str, str],
        metric_definitions: dict[str, dict[str, Any]] | None = None,
    ) -> list[Objective]:
        try:
            output_objectives = []
            for metric_name, min_or_max in objectives.items():
                minimize = (
                    MetricObjective[min_or_max.upper()] == MetricObjective.MINIMIZE
                )
                objective = Objective(
                    metric=cls._make_metric(
                        name=metric_name,
                        for_opt_config=True,
                        lower_is_better=minimize,
                        metric_definitions=metric_definitions,
                    ),
                    minimize=minimize,
                )
                output_objectives.append(objective)
            return output_objectives
        except KeyError as k:
            raise ValueError(
                "Objective values should specify "
                f"'{MetricObjective.MINIMIZE.name.lower()}' or "
                f"'{MetricObjective.MAXIMIZE.name.lower()}', got {k} in"
                f" objectives({objectives})"
            )

    @classmethod
    def make_outcome_constraints(
        cls,
        outcome_constraints: list[str],
        status_quo_defined: bool,
        metric_definitions: dict[str, dict[str, Any]] | None = None,
    ) -> list[OutcomeConstraint]:
        typed_outcome_constraints = [
            cls.outcome_constraint_from_str(c, metric_definitions=metric_definitions)
            for c in outcome_constraints
        ]

        if status_quo_defined is False and any(
            oc.relative for oc in typed_outcome_constraints
        ):
            raise ValueError(
                "Must set status_quo to have relative outcome constraints."
            )

        return typed_outcome_constraints

    @classmethod
    def make_objective_thresholds(
        cls,
        objective_thresholds: list[str],
        status_quo_defined: bool,
        metric_definitions: dict[str, dict[str, Any]] | None = None,
    ) -> list[ObjectiveThreshold]:
        typed_objective_thresholds = (
            [
                cls.objective_threshold_constraint_from_str(
                    c, metric_definitions=metric_definitions
                )
                for c in objective_thresholds
            ]
            if objective_thresholds is not None
            else []
        )

        if status_quo_defined is False and any(
            oc.relative for oc in typed_objective_thresholds
        ):
            raise ValueError(
                "Must set status_quo to have relative objective thresholds."
            )

        return typed_objective_thresholds

    @staticmethod
    def optimization_config_from_objectives(
        objectives: list[Objective],
        objective_thresholds: list[ObjectiveThreshold],
        outcome_constraints: list[OutcomeConstraint],
    ) -> OptimizationConfig:
        """Parse objectives and constraints to define optimization config.

        The resulting optimization config will be regular single-objective config
        if `objectives` is a list of one element and a multi-objective config
        otherwise.

        NOTE: If passing in multiple objectives, `objective_thresholds` must be a
        non-empty list defining constraints for each objective.
        """
        if len(objectives) == 1:
            if objective_thresholds:
                raise ValueError(
                    "Single-objective optimizations must not specify objective "
                    "thresholds."
                )
            return OptimizationConfig(
                objective=objectives[0],
                outcome_constraints=outcome_constraints,
            )

        if not objective_thresholds:
            logger.info(
                "Due to non-specification, we will use the heuristic for selecting "
                "objective thresholds."
            )

        return MultiObjectiveOptimizationConfig(
            objective=MultiObjective(objectives=objectives),
            outcome_constraints=outcome_constraints,
            objective_thresholds=objective_thresholds,
        )

    @classmethod
    def make_optimization_config(
        cls,
        objectives: dict[str, str],
        objective_thresholds: list[str],
        outcome_constraints: list[str],
        status_quo_defined: bool,
        metric_definitions: dict[str, dict[str, Any]] | None = None,
    ) -> OptimizationConfig:
        return cls.optimization_config_from_objectives(
            cls.make_objectives(objectives, metric_definitions=metric_definitions),
            cls.make_objective_thresholds(
                objective_thresholds,
                status_quo_defined,
                metric_definitions=metric_definitions,
            ),
            cls.make_outcome_constraints(
                outcome_constraints,
                status_quo_defined,
                metric_definitions=metric_definitions,
            ),
        )

    @classmethod
    def make_optimization_config_from_properties(
        cls,
        objectives: dict[str, ObjectiveProperties] | None = None,
        outcome_constraints: list[str] | None = None,
        metric_definitions: dict[str, dict[str, Any]] | None = None,
        status_quo_defined: bool = False,
    ) -> OptimizationConfig | None:
        """Makes optimization config based on ObjectiveProperties objects

        Args:
            objectives: Mapping from an objective name to object containing:
                minimize: Whether this experiment represents a minimization problem.
                threshold: The bound in the objective's threshold constraint.
            outcome_constraints: List of string representation of outcome
                constraints of form "metric_name >= bound", like "m1 <= 3."
            status_quo_defined: bool for whether the experiment has a status quo
            metric_definitions: A mapping of metric names to extra kwargs to pass
                to that metric
        """
        if objectives is not None:
            objective_thresholds = (
                cls.build_objective_thresholds(objectives)
                if objectives is not None
                else []
            )
            simple_objectives = {
                objective: ("minimize" if properties.minimize else "maximize")
                for objective, properties in objectives.items()
            }
            return cls.make_optimization_config(
                objectives=simple_objectives,
                objective_thresholds=objective_thresholds,
                outcome_constraints=outcome_constraints or [],
                status_quo_defined=status_quo_defined,
                metric_definitions=metric_definitions,
            )
        return None

    @classmethod
    def make_search_space(
        cls,
        parameters: list[TParameterRepresentation],
        parameter_constraints: list[str] | None,
    ) -> SearchSpace:
        parameter_constraints = (
            parameter_constraints if parameter_constraints is not None else []
        )
        typed_parameters = [cls.parameter_from_json(p) for p in parameters]
        is_hss = any(p.is_hierarchical for p in typed_parameters)
        search_space_cls = HierarchicalSearchSpace if is_hss else SearchSpace

        parameter_map = {p.name: p for p in typed_parameters}

        typed_parameter_constraints = [
            cls.constraint_from_str(c, parameter_map) for c in parameter_constraints
        ]

        if any(
            any(
                isinstance(parameter_map[parameter], ChoiceParameter)
                for parameter in constraint.constraint_dict
            )
            for constraint in typed_parameter_constraints
        ):
            raise UnsupportedError(
                "Constraints on ChoiceParameters are not allowed. Try absorbing "
                "this constraint into the associated range parameter's bounds."
            )

        if any(
            any(
                isinstance(parameter_map[parameter], FixedParameter)
                for parameter in constraint.constraint_dict
            )
            for constraint in typed_parameter_constraints
        ):
            raise UnsupportedError(
                "Constraints on FixedParameters are not allowed. Try absorbing "
                "this constraint into the associated range parameter's bounds."
            )

        ss = search_space_cls(
            parameters=typed_parameters,
            parameter_constraints=typed_parameter_constraints,
        )

        logger.info(f"Created search space: {ss}.")
        if is_hss:
            hss = assert_is_instance(ss, HierarchicalSearchSpace)
            logger.info(
                "Hieararchical structure of the search space: \n"
                f"{hss.hierarchical_structure_str(parameter_names_only=True)}"
            )

        return search_space_cls(
            parameters=typed_parameters,
            parameter_constraints=typed_parameter_constraints,
        )

    @classmethod
    def _get_default_objectives(cls) -> dict[str, str] | None:
        """Get the default objective and its optimization direction.

        The return type is optional since some subclasses may not wish to
        use any optimization config by default.
        """
        return {DEFAULT_OBJECTIVE_NAME: "maximize"}

    @classmethod
    def make_experiment(
        cls,
        parameters: list[TParameterRepresentation],
        name: str | None = None,
        description: str | None = None,
        owners: list[str] | None = None,
        parameter_constraints: list[str] | None = None,
        outcome_constraints: list[str] | None = None,
        status_quo: TParameterization | None = None,
        experiment_type: str | None = None,
        tracking_metric_names: list[str] | None = None,
        metric_definitions: dict[str, dict[str, Any]] | None = None,
        objectives: dict[str, str] | None = None,
        objective_thresholds: list[str] | None = None,
        support_intermediate_data: bool = False,
        immutable_search_space_and_opt_config: bool = True,
        auxiliary_experiments_by_purpose: None
        | (dict[AuxiliaryExperimentPurpose, list[AuxiliaryExperiment]]) = None,
        default_trial_type: str | None = None,
        default_runner: Runner | None = None,
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
            outcome_constraints: List of string representation of outcome
                constraints of form "metric_name >= bound", like "m1 <= 3."
            status_quo: Parameterization of the current state of the system.
                If set, this will be added to each trial to be evaluated alongside
                test configurations.
            experiment_type: String indicating type of the experiment (e.g. name of
                a product in which it is used), if any.
            tracking_metric_names: Names of additional tracking metrics not used for
                optimization.
            metric_definitions: A mapping of metric names to extra kwargs to pass
                to that metric
            objectives: Mapping from an objective name to "minimize" or "maximize"
                representing the direction for that objective.
            objective_thresholds: A list of objective threshold constraints for multi-
                objective optimization, in the same string format as
                `outcome_constraints` argument.
            support_intermediate_data: Whether trials may report metrics results for
                incomplete runs.
            immutable_search_space_and_opt_config: Whether it's possible to update the
                search space and optimization config on this experiment after creation.
                Defaults to True. If set to True, we won't store or load copies of the
                search space and optimization config on each generator run, which will
                improve storage performance.
            auxiliary_experiments_by_purpose: Dictionary of auxiliary experiments for
                different use cases (e.g., transfer learning).
            default_trial_type: The default trial type if multiple
                trial types are intended to be used in the experiment.  If specified,
                a MultiTypeExperiment will be created. Otherwise, a single-type
                Experiment will be created.
            default_runner: The default runner in this experiment.
                This only applies to MultiTypeExperiment (when default_trial_type
                is specified).
            is_test: Whether this experiment will be a test experiment (useful for
                marking test experiments in storage etc). Defaults to False.

        """
        if (default_trial_type is None) != (default_runner is None):
            raise ValueError(
                "Must specify both default_trial_type and default_runner if "
                "using a MultiTypeExperiment."
            )

        status_quo_arm = None if status_quo is None else Arm(parameters=status_quo)

        objectives = objectives or cls._get_default_objectives()
        if objectives:
            optimization_config = cls.make_optimization_config(
                objectives=objectives,
                objective_thresholds=objective_thresholds or [],
                outcome_constraints=outcome_constraints or [],
                status_quo_defined=status_quo_arm is not None,
                metric_definitions=metric_definitions,
            )
        else:
            optimization_config = None

        tracking_metrics = (
            None
            if tracking_metric_names is None
            else [
                cls._make_metric(
                    name=metric_name, metric_definitions=metric_definitions
                )
                for metric_name in tracking_metric_names
            ]
        )

        default_data_type = (
            DataType.MAP_DATA if support_intermediate_data else DataType.DATA
        )

        properties: dict[str, Any] = {}

        if immutable_search_space_and_opt_config:
            properties[Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF] = (
                immutable_search_space_and_opt_config
            )

        if owners is not None:
            properties["owners"] = owners
        if default_trial_type is not None:
            return MultiTypeExperiment(
                name=none_throws(name),
                search_space=cls.make_search_space(parameters, parameter_constraints),
                default_trial_type=none_throws(default_trial_type),
                default_runner=none_throws(default_runner),
                optimization_config=optimization_config,
                tracking_metrics=tracking_metrics,
                status_quo=status_quo_arm,
                description=description,
                is_test=is_test,
                experiment_type=experiment_type,
                properties=properties,
                default_data_type=default_data_type,
            )

        return Experiment(
            name=name,
            description=description,
            search_space=cls.make_search_space(parameters, parameter_constraints),
            optimization_config=optimization_config,
            status_quo=status_quo_arm,
            experiment_type=experiment_type,
            tracking_metrics=tracking_metrics,
            default_data_type=default_data_type,
            properties=properties,
            auxiliary_experiments_by_purpose=auxiliary_experiments_by_purpose,
            is_test=is_test,
        )

    @classmethod
    def build_objective_thresholds(
        cls, objectives: dict[str, ObjectiveProperties]
    ) -> list[str]:
        """Construct a list of constraint string for an objective thresholds
        interpretable by `make_experiment()`

        Args:
            objectives: Mapping of name of the objective to Object containing:
                minimize: Whether this experiment represents a minimization problem.
                threshold: The bound in the objective's threshold constraint.
        """
        return [
            cls.build_objective_threshold(objective, properties)
            for objective, properties in objectives.items()
            if properties.threshold is not None
        ]

    @staticmethod
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

    @staticmethod
    def make_fixed_observation_features(
        fixed_features: FixedFeatures,
    ) -> ObservationFeatures:
        """Construct ObservationFeatures from FixedFeatures.

        Args:
            fixed_features: The fixed features for generation.

        Returns:
            The new ObservationFeatures object.
        """
        return ObservationFeatures(
            parameters=fixed_features.parameters,
            trial_index=(
                None
                if fixed_features.trial_index is None
                else fixed_features.trial_index
            ),
        )

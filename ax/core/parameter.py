#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod, abstractproperty
from copy import deepcopy
from enum import Enum
from logging import Logger
from math import inf
from typing import Any, cast, Union
from warnings import warn

from ax.core.types import TNumeric, TParameterization, TParamValue
from ax.exceptions.core import AxParameterWarning, UnsupportedError, UserInputError
from ax.utils.common.base import SortableBase
from ax.utils.common.logger import get_logger
from ax.utils.common.string_utils import sanitize_name, unsanitize_name
from pyre_extensions import assert_is_instance, none_throws
from scipy.special import expit, logit
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify

logger: Logger = get_logger(__name__)

# Tolerance for floating point comparisons. This is relatively permissive,
# and allows for serializing at rather low numerical precision.
# TODO: Do a more comprehensive audit of how floating point precision issues
# may creep up and implement a more principled fix
EPS = 1.5e-7
MAX_VALUES_CHOICE_PARAM = 1000
FIXED_CHOICE_PARAM_ERROR = (
    "ChoiceParameters require multiple feasible values. "
    "Please use FixedParameter instead when setting a single possible value."
)
REPR_FLAGS_IF_TRUE_ONLY = [
    "is_fidelity",
    "is_task",
    "is_hierarchical",
    "log_scale",
    "logit_scale",
]


class ParameterType(Enum):
    # pyre-fixme[35]: Target cannot be annotated.
    BOOL: int = 0
    # pyre-fixme[35]: Target cannot be annotated.
    INT: int = 1
    # pyre-fixme[35]: Target cannot be annotated.
    FLOAT: int = 2
    # pyre-fixme[35]: Target cannot be annotated.
    STRING: int = 3

    @property
    def is_numeric(self) -> bool:
        return self == ParameterType.INT or self == ParameterType.FLOAT


TParameterType = Union[type[int], type[float], type[str], type[bool]]

# pyre: PARAMETER_PYTHON_TYPE_MAP is declared to have type
# pyre: `Dict[ParameterType, Union[Type[bool], Type[float], Type[int],
# pyre: Type[str]]]` but is used as type `Dict[ParameterType,
# pyre-fixme[9]: Type[Union[float, str]]]`.
PARAMETER_PYTHON_TYPE_MAP: dict[ParameterType, TParameterType] = {
    ParameterType.INT: int,
    ParameterType.FLOAT: float,
    ParameterType.STRING: str,
    ParameterType.BOOL: bool,
}

SUPPORTED_PARAMETER_TYPES: tuple[
    type[bool] | type[float] | type[int] | type[str], ...
] = tuple(PARAMETER_PYTHON_TYPE_MAP.values())


# pyre-fixme[24]: Generic type `type` expects 1 type parameter, use `typing.Type` to
#  avoid runtime subscripting errors.
def _get_parameter_type(python_type: type) -> ParameterType:
    """Given a Python type, retrieve corresponding Ax ``ParameterType``."""
    for param_type, py_type in PARAMETER_PYTHON_TYPE_MAP.items():
        if py_type == python_type:
            return param_type
    raise ValueError(f"No Ax parameter type corresponding to {python_type}.")


class Parameter(SortableBase, metaclass=ABCMeta):
    _is_fidelity: bool = False
    _name: str
    _target_value: TParamValue = None
    _parameter_type: ParameterType
    _backfill_value: TParamValue = None
    _default_value: TParamValue = None

    def cast(self, value: TParamValue) -> TParamValue:
        if value is None:
            raise UnsupportedError("None values are not supported.")
        return self.python_type(value)

    @abstractmethod
    def validate(self, value: TParamValue, raises: bool = False) -> bool:
        """Returns True if input is a valid value for the parameter.

        Args:
            value: Value being checked.
            raises: If true, and validation fails, raises a UserInputError.

        Raises:
            UserInputError: If validation fails and raises is True.

        Returns:
            True if valid, False otherwise.
        """

    @abstractmethod
    def cardinality(self) -> float:
        pass

    @property
    def python_type(self) -> TParameterType:
        """The python type for the corresponding ParameterType enum.

        Used primarily for casting values of unknown type to conform
        to that of the parameter.
        """
        return PARAMETER_PYTHON_TYPE_MAP[self.parameter_type]

    def is_valid_type(self, value: TParamValue) -> bool:
        """Whether a given value's type is allowed by this parameter."""
        return type(value) is self.python_type or (
            # ints are floats
            type(value) is int and self.python_type is float
        )

    @property
    def is_numeric(self) -> bool:
        return self.parameter_type.is_numeric

    @property
    def is_fidelity(self) -> bool:
        return self._is_fidelity

    @property
    def is_hierarchical(self) -> bool:
        return isinstance(self, (ChoiceParameter, FixedParameter)) and bool(
            self._dependents
        )

    @property
    def target_value(self) -> TParamValue:
        return self._target_value

    @property
    def backfill_value(self) -> TParamValue:
        return self._backfill_value

    @property
    def default_value(self) -> TParamValue:
        return self._default_value

    @property
    def is_disabled(self) -> bool:
        return self.default_value is not None

    @property
    def parameter_type(self) -> ParameterType:
        return self._parameter_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def dependents(self) -> dict[TParamValue, list[str]]:
        raise NotImplementedError(
            "Only fixed and choice hierarchical parameters are currently supported."
        )

    # pyre-fixme[7]: Expected `Parameter` but got implicit return value of `None`.
    def clone(self) -> Parameter:
        pass

    def disable(self, default_value: TParamValue) -> None:
        """
        Effectively remove parameter from the search space for future trial generation.
        Existing trials remain valid, and the disabled parameter is replaced with the
        default_value for all subsequent trials.
        """
        if self.is_disabled:
            logger.warning(
                f"Parameter {self.name} is already disabled with "
                f"default value {self.default_value}. "
                f"Updating default value to {default_value}."
            )
        self._default_value = default_value

    @property
    def _unique_id(self) -> str:
        return str(self)

    def _base_repr(self) -> str:
        ret_val = (
            f"{self.__class__.__name__}("
            f"name='{self._name}', "
            f"parameter_type={self.parameter_type.name}, "
            f"{self.domain_repr}"
        )

        # Add binary flags.
        for flag in self.available_flags:
            val = getattr(self, flag, False)
            if flag not in REPR_FLAGS_IF_TRUE_ONLY or (
                flag in REPR_FLAGS_IF_TRUE_ONLY and val is True
            ):
                ret_val += f", {flag}={val}"

        # Add target_value if one exists.
        if self.target_value is not None:
            tval_rep = self.target_value
            if self.parameter_type == ParameterType.STRING:
                tval_rep = f"'{tval_rep}'"
            ret_val += f", target_value={tval_rep}"

        return ret_val

    @abstractproperty
    def domain_repr(self) -> str:
        """Returns a string representation of the domain."""
        pass

    @property
    def available_flags(self) -> list[str]:
        """List of boolean attributes that can be set on this parameter."""
        return ["is_fidelity"]

    @property
    def summary_dict(
        self,
    ) -> dict[str, Any]:
        # Assemble dict.
        summary_dict: dict[str, Any] = {
            "name": self.name,
            "type": self.__class__.__name__.removesuffix("Parameter"),
            "domain": self.domain_repr,
            "parameter_type": self.parameter_type.name.lower(),
        }

        # Extract flags.
        flags = []
        for flag in self.available_flags:
            flag_val = getattr(self, flag, None)
            flag_repr = flag.removeprefix("is_")
            if flag == "sort_values":
                flag_repr = "sorted"
            if flag_val is True:
                flags.append(flag_repr)
            elif flag_val is False and flag not in REPR_FLAGS_IF_TRUE_ONLY:
                flags.append("un" + flag_repr)

        # Add flags, target_values, and dependents if present.
        if flags:
            summary_dict["flags"] = ", ".join(flags)
        if getattr(self, "is_fidelity", False) or getattr(self, "is_task", False):
            summary_dict["target_value"] = self.target_value
        if getattr(self, "is_hierarchical", False):
            summary_dict["dependents"] = self.dependents
        if getattr(self, "backfill_value", None) is not None:
            summary_dict["backfill_value"] = self.backfill_value
        if getattr(self, "default_value", None) is not None:
            summary_dict["default_value"] = self.default_value

        return summary_dict


class RangeParameter(Parameter):
    """Parameter object that specifies a range of values."""

    def __init__(
        self,
        name: str,
        parameter_type: ParameterType,
        lower: float,
        upper: float,
        log_scale: bool = False,
        logit_scale: bool = False,
        digits: int | None = None,
        is_fidelity: bool = False,
        target_value: TParamValue = None,
        backfill_value: TParamValue = None,
        default_value: TParamValue = None,
    ) -> None:
        """Initialize RangeParameter

        Args:
            name: Name of the parameter.
            parameter_type: Enum indicating the type of parameter
                value (e.g. string, int).
            lower: Lower bound of the parameter range (inclusive).
            upper: Upper bound of the parameter range (inclusive).
            log_scale: Whether to sample in the log space when drawing
                random values of the parameter.
            logit_scale: Whether to sample in logit space when drawing
                random values of the parameter.
            digits: Number of digits to round values to for float type.
            is_fidelity: Whether this parameter is a fidelity parameter.
            target_value: Target value of this parameter if it is a fidelity.
            backfill_value: For parameters added to experiments that have already run
                trials.
                Used to backfill trials missing the parameter.
            default_value: For parameters disabled in experiments that have already
                run trials. Used as default value in modeling for future trials.
        """
        if is_fidelity and (target_value is None):
            raise UserInputError(
                "`target_value` should not be None for the fidelity parameter: "
                "{}".format(name)
            )

        self._name = name
        if parameter_type not in (ParameterType.INT, ParameterType.FLOAT):
            raise UserInputError("RangeParameter type must be int or float.")
        self._parameter_type = parameter_type
        self._digits = digits
        self._lower: TNumeric = self.cast(lower)
        self._upper: TNumeric = self.cast(upper)
        self._log_scale = log_scale
        self._logit_scale = logit_scale
        self._is_fidelity = is_fidelity
        self._target_value: TNumeric | None = (
            self.cast(target_value) if target_value is not None else None
        )
        self._backfill_value: TNumeric | None = (
            self.cast(backfill_value) if backfill_value is not None else None
        )
        self._default_value: TNumeric | None = (
            self.cast(default_value) if default_value is not None else None
        )

        self._validate_range_param(
            parameter_type=parameter_type,
            lower=lower,
            upper=upper,
            log_scale=log_scale,
            logit_scale=logit_scale,
        )

    def cardinality(self) -> TNumeric:
        if self.parameter_type == ParameterType.FLOAT:
            return inf

        return int(self.upper) - int(self.lower) + 1

    def _validate_range_param(
        self,
        lower: TNumeric,
        upper: TNumeric,
        log_scale: bool,
        logit_scale: bool,
        parameter_type: ParameterType | None = None,
    ) -> None:
        if parameter_type and parameter_type not in (
            ParameterType.INT,
            ParameterType.FLOAT,
        ):
            raise UserInputError(
                f"RangeParameter {self.name}type must be int or float."
            )

        upper = float(upper)
        if lower >= upper:
            raise UserInputError(
                f"Upper bound of {self.name} must be strictly larger than lower."
                f"Got: ({lower}, {upper})."
            )
        width: float = upper - lower
        if width < 100 * EPS:
            raise UserInputError(
                f"Parameter {self.name}'s range ({width}) is very small and likely "
                "to cause numerical errors. Consider reparameterizing your "
                "problem by scaling the parameter."
            )
        if log_scale and logit_scale:
            raise UserInputError(f"{self.name} can't use both log and logit.")
        if log_scale and lower <= 0:
            raise UserInputError(f"{self.name} cannot take log when min <= 0.")
        if logit_scale and (lower <= 0 or upper >= 1):
            raise UserInputError(f"{self.name} logit requires lower > 0 and upper < 1")
        if not (self.is_valid_type(lower)) or not (self.is_valid_type(upper)):
            raise UserInputError(
                f"[{lower}, {upper}] is an invalid range for {self.name}."
            )

    @property
    def upper(self) -> TNumeric:
        """Upper bound of the parameter range.

        Value is cast to parameter type upon set and also validated
        to ensure the bound is strictly greater than lower bound.
        """
        return self._upper

    @upper.setter
    def upper(self, value: TNumeric) -> None:
        self._validate_range_param(
            lower=self.lower,
            upper=value,
            log_scale=self.log_scale,
            logit_scale=self.logit_scale,
        )
        self._upper = self.cast(value)

    @property
    def lower(self) -> TNumeric:
        """Lower bound of the parameter range.

        Value is cast to parameter type upon set and also validated
        to ensure the bound is strictly less than upper bound.
        """
        return self._lower

    @lower.setter
    def lower(self, value: TNumeric) -> None:
        self._validate_range_param(
            lower=value,
            upper=self.upper,
            log_scale=self.log_scale,
            logit_scale=self.logit_scale,
        )
        self._lower = self.cast(value)

    @property
    def digits(self) -> int | None:
        """Number of digits to round values to for float type.

        Upper and lower bound are re-cast after this property is changed.
        """
        return self._digits

    @property
    def log_scale(self) -> bool:
        """Whether the parameter's values should be sampled from log space."""
        return self._log_scale

    @property
    def logit_scale(self) -> bool:
        """Whether the parameter's random values should be sampled from logit space."""
        return self._logit_scale

    def update_range(
        self, lower: float | None = None, upper: float | None = None
    ) -> RangeParameter:
        """Set the range to the given values.

        If lower or upper is not provided, it will be left at its current value.

        Args:
            lower: New value for the lower bound.
            upper: New value for the upper bound.
        """
        if lower is None:
            lower = self._lower
        if upper is None:
            upper = self._upper

        cast_lower = self.cast(lower)
        cast_upper = self.cast(upper)
        self._validate_range_param(
            lower=cast_lower,
            upper=cast_upper,
            log_scale=self.log_scale,
            logit_scale=self.logit_scale,
        )
        self._lower = cast_lower
        self._upper = cast_upper
        return self

    def set_digits(self, digits: int | None) -> RangeParameter:
        self._digits = digits

        # Re-scale min and max to new digits definition
        cast_lower = self.cast(self._lower)
        cast_upper = self.cast(self._upper)
        # `<=` is not supported for operand types `Union[float, int]` and `int`.
        # pyre-ignore [58]
        if cast_lower >= cast_upper:
            raise UserInputError(
                f"Lower bound {cast_lower} is >= upper bound {cast_upper}."
            )

        self._lower = cast_lower
        self._upper = cast_upper
        return self

    def set_log_scale(self, log_scale: bool) -> RangeParameter:
        self._log_scale = log_scale
        return self

    def set_logit_scale(self, logit_scale: bool) -> RangeParameter:
        self._logit_scale = logit_scale
        return self

    def validate(
        self, value: TParamValue, raises: bool = False, tol: float = EPS
    ) -> bool:
        """Returns True if input is a valid value for the parameter.

        Checks that value is of the right type and within
        the valid range for the parameter. Returns False if value is None.

        Args:
            value: Value being checked.
            raises: If true, and validation fails, raises a UserInputError.
            tol: Absolute tolerance for floating point comparisons.

        Raises:
            UserInputError: If validation fails and raises is True.

        Returns:
            True if valid, False otherwise.
        """

        def return_false_or_raise(msg: str) -> bool:
            if raises:
                raise UserInputError(msg)
            return False

        if value is None:
            msg = (
                f"Value of parameter {self.name} is `None` but the parameter "
                f"type is {self.parameter_type}."
            )
            return return_false_or_raise(msg)

        if not self.is_valid_type(value):
            msg = (
                f"Value ({value}) of parameter {self.name} has type ({type(value)}), "
                "which is not valid for a RangeParameter with parameter type "
                f"{self.parameter_type}."
            )
            return return_false_or_raise(msg)

        value = self.cast(value)
        if value < self.lower - tol or value > self.upper + tol:
            interval = (self.lower, self.upper)
            msg = (
                f"Value ({value}) of parameter {self.name} is not within the range of "
                f"the parameter {interval}, even with a tolerance of {tol}."
            )
            return return_false_or_raise(msg)

        return True

    def is_valid_type(self, value: TParamValue) -> bool:
        """Same as default except allows floats whose value is an int
        for Int parameters.
        """
        if not (isinstance(value, float) or isinstance(value, int)):
            return False

        # This might have issues with ints > 2^24
        if self.parameter_type is ParameterType.INT:
            return isinstance(value, int) or float(none_throws(value)).is_integer()
        return True

    def clone(self) -> RangeParameter:
        return RangeParameter(
            name=self._name,
            parameter_type=self._parameter_type,
            lower=self._lower,
            upper=self._upper,
            log_scale=self._log_scale,
            logit_scale=self._logit_scale,
            digits=self._digits,
            is_fidelity=self._is_fidelity,
            target_value=self._target_value,
            backfill_value=self._backfill_value,
            default_value=self._default_value,
        )

    def cast(self, value: TParamValue) -> TNumeric:
        value = super().cast(value=value)
        if self.parameter_type is ParameterType.FLOAT and self._digits is not None:
            return round(float(value), none_throws(self._digits))
        return assert_is_instance(value, TNumeric)

    def __repr__(self) -> str:
        ret_val = self._base_repr()

        if self._digits is not None:
            ret_val += f", digits={self._digits}"

        return ret_val + ")"

    @property
    def available_flags(self) -> list[str]:
        """List of boolean attributes that can be set on this parameter."""
        return super().available_flags + ["log_scale", "logit_scale"]

    @property
    def domain_repr(self) -> str:
        """Returns a string representation of the domain."""
        return f"range={[self.lower, self.upper]}"


class ChoiceParameter(Parameter):
    """Parameter object that specifies a discrete set of values.

    Args:
        name: Name of the parameter.
        parameter_type: Enum indicating the type of parameter
            value (e.g. string, int).
        values: List of allowed values for the parameter.
        is_ordered: If False, the parameter is a categorical variable.
            Defaults to False if parameter_type is STRING and ``values``
            is longer than 2, else True.
        is_task: Treat the parameter as a task parameter for modeling.
        is_fidelity: Whether this parameter is a fidelity parameter.
        target_value: Target value of this parameter if it's a fidelity or
            task parameter.
        sort_values: Whether to sort ``values`` before encoding.
            Defaults to False if ``parameter_type`` is STRING, else
            True. Note: Numeric ordered parameters (int or float with
            ``is_ordered=True``) must have ``sort_values=True``.
        log_scale: Whether to sample choice values from log space. Only valid
            for numerical (int or float) parameters with all positive values.
        dependents: Optional mapping for parameters in hierarchical search
            spaces; format is { value -> list of dependent parameter names }.
        bypass_cardinality_check: Whether to bypass the cardinality check
            that restricts the number of distinct values. This should only be
            set to True when constructing parameters within the modeling layer.
        backfill_value: For parameters added to experiments that have already run.
                Used to backfill trials missing the parameter.
        default_value: For parameters disabled in experiments that have already
                run. Used as default value in modeling for future trials.
    """

    def __init__(
        self,
        name: str,
        parameter_type: ParameterType,
        values: list[TParamValue],
        is_ordered: bool | None = None,
        is_task: bool = False,
        is_fidelity: bool = False,
        target_value: TParamValue = None,
        sort_values: bool | None = None,
        log_scale: bool | None = None,
        dependents: dict[TParamValue, list[str]] | None = None,
        bypass_cardinality_check: bool = False,
        backfill_value: TParamValue = None,
        default_value: TParamValue = None,
    ) -> None:
        if (is_fidelity or is_task) and (target_value is None):
            ptype = "fidelity" if is_fidelity else "task"
            raise UserInputError(
                f"`target_value` should not be None for the {ptype} parameter: "
                "{}".format(name)
            )

        self._name = name
        self._parameter_type = parameter_type
        self._is_task = is_task
        self._is_fidelity = is_fidelity
        self._target_value: TParamValue = (
            self.cast(target_value) if target_value is not None else None
        )
        self._backfill_value: TParamValue = (
            self.cast(backfill_value) if backfill_value is not None else None
        )
        self._default_value: TParamValue = (
            self.cast(default_value) if default_value is not None else None
        )
        # A choice parameter with only one value is a FixedParameter.
        if not len(values) > 1:
            raise UserInputError(f"{self._name}({values}): {FIXED_CHOICE_PARAM_ERROR}")
        # Cap the number of possible values.
        if not bypass_cardinality_check and len(values) > MAX_VALUES_CHOICE_PARAM:
            raise UserInputError(
                f"`ChoiceParameter` with more than {MAX_VALUES_CHOICE_PARAM} values "
                "is not supported! Use a `RangeParameter` instead."
            )
        self._bypass_cardinality_check = bypass_cardinality_check
        # Remove duplicate values.
        # Using dict to deduplicate here since set doesn't preserve order but dict does.
        dict_values = dict.fromkeys(values)
        if len(values) != len(dict_values):
            warn(
                f"Duplicate values found for ChoiceParameter {name}. "
                "Initializing the parameter with duplicate values removed. ",
                AxParameterWarning,
                stacklevel=2,
            )
            values = list(dict_values)

        if is_ordered is False and len(values) == 2:
            is_ordered = True
            logger.debug(
                f"Changing `is_ordered` to `True` for `ChoiceParameter` '{name}' since "
                "there are only two possible values.",
                AxParameterWarning,
                stacklevel=3,
            )
        self._is_ordered: bool = (
            is_ordered
            if is_ordered is not None
            else self._get_default_is_ordered_and_warn(num_choices=len(values))
        )
        # sort_values defaults to True if the parameter is not a string
        self._sort_values: bool = (
            sort_values
            if sort_values is not None
            else self._get_default_sort_values_and_warn()
        )
        # Validate that numeric ordered parameters have sort_values=True
        if self._is_ordered and parameter_type.is_numeric and not self._sort_values:
            raise UserInputError(
                f"Numeric ordered choice parameters must have sort_values=True. "
                f"Parameter {name} is ordered with type {parameter_type.name} but "
                f"has sort_values=False."
            )
        if self.sort_values:
            values = cast(list[TParamValue], sorted([none_throws(v) for v in values]))
        self._values: list[TParamValue] = self._cast_values(values)

        # Auto-detect log_scale if not explicitly set
        if log_scale is None:
            log_scale = self._get_default_log_scale(
                values=self._values, parameter_type=parameter_type
            )

        # Validate log_scale constraints
        if log_scale:
            if not parameter_type.is_numeric:
                raise UserInputError(
                    f"log_scale is only supported for numerical parameters. "
                    f"Parameter {name} has type {parameter_type.name}."
                )
            # Check that all values are positive
            for value in self._values:
                if float(value) <= 0:
                    raise UserInputError(
                        f"log_scale requires all values to be positive. "
                        f"Parameter {name} has value {value} which is <= 0."
                    )
            # Check that parameter is ordered -- doesn't make sense for categoricals.
            if not self._is_ordered:
                raise UserInputError(
                    f"log_scale is only supported for ordered parameters. "
                    f"Parameter {name} has is_ordered=False."
                )
        self._log_scale: bool = log_scale

        if dependents:
            for value in dependents:
                if value not in self.values:
                    raise UserInputError(
                        f"Value {value} in `dependents` "
                        f"argument is not among the parameter values: {self.values}."
                    )
        # NOTE: We don't need to check that dependent parameters actually exist as
        # that is done in `HierarchicalSearchSpace` constructor.
        self._dependents = dependents

    def _get_default_is_ordered_and_warn(self, num_choices: int) -> bool:
        default_bool = self._parameter_type != ParameterType.STRING or num_choices == 2
        if self._parameter_type == ParameterType.STRING and num_choices > 2:
            motivation = " since the parameter is a string with more than 2 choices."
        elif num_choices == 2:
            motivation = " since there are exactly two choices."
        else:
            motivation = " since the parameter is not of type string."
        warn(
            f'`is_ordered` is not specified for `ChoiceParameter` "{self._name}". '
            f"Defaulting to `{default_bool}` {motivation}. To override this behavior "
            f"(or avoid this warning), specify `is_ordered` during `ChoiceParameter` "
            "construction. Note that choice parameters with exactly 2 choices are "
            "always considered ordered and that the user-supplied `is_ordered` has no "
            "effect in this particular case.",
            AxParameterWarning,
            stacklevel=3,
        )
        return default_bool

    def _get_default_sort_values_and_warn(self) -> bool:
        default_bool = self._parameter_type != ParameterType.STRING
        # Don't warn for numeric ordered parameters since we enforce sort_values=True
        if not (self._is_ordered and self._parameter_type.is_numeric):
            warn(
                f'`sort_values` is not specified for `ChoiceParameter` "{self._name}". '
                f"Defaulting to `{default_bool}` for parameters of `ParameterType` "
                f"{self.parameter_type.name}. To override this behavior (or avoid this "
                "warning), specify `sort_values` during `ChoiceParameter` "
                "construction.",
                AxParameterWarning,
                stacklevel=3,
            )
        return default_bool

    def _get_default_log_scale(
        self, values: list[TParamValue], parameter_type: ParameterType
    ) -> bool:
        """Get the default value for log_scale.

        Returns True if all values are positive and any of the following
        heuristics is satisfied:
        1. Exponential spacing (generalized): Values follow the pattern c * base^p
           where c is a constant, base is inferred from the data, and p are integers
           (possibly with some skipped). This handles:
           - Equal ratios: [2, 4, 8, 16] = [2^1, 2^2, 2^3, 2^4]
           - Skipped powers: [64, 128, 512] = [2^6, 2^7, 2^9]
           - Constant factor: [10, 20, 40, 80] = 10 * [2^0, 2^1, 2^2, 2^3]
           - Any base: [3, 9, 27] = [3^1, 3^2, 3^3]
        2. Spans orders of magnitude: Values span at least 2 orders of magnitude
           (e.g., 0.01 to 1.0 or 1 to 100). This also captures cases where
           max/min >= 100.

        Args:
            values: List of parameter values to check.
            parameter_type: The parameter type.

        Returns:
            True if values should be modeled in log-scale, False otherwise.
        """
        if not parameter_type.is_numeric or not self._is_ordered:
            # Only numeric types & ordered parameters can have log-scale.
            return False
        if len(values) < 3:
            # Need at least 3 values to detect a pattern.
            return False
        vals = [float(v) for v in values]  # refine type.
        if any(v <= 0.0 for v in vals):
            # All values must be positive.
            return False

        # Heuristic 1: Generalized exponential spacing
        # Infer the base from the ratio of first two values, then check if all
        # values follow the pattern c * base^p for some constant c and integer powers p.
        # If values are of the form c * base^p, then log_base(v) = log_base(c) + p.
        # The fractional parts of log_base(v) should all be approximately equal.
        inferred_base = vals[1] / vals[0]
        log_vals = [math.log(val) / math.log(inferred_base) for val in vals]
        fractional_parts = [log_val - round(log_val) for log_val in log_vals]

        # Check if all fractional parts are approximately equal
        first_frac = fractional_parts[0]
        # Allow 0.1 tolerance in the fractional part
        if all(abs(frac - first_frac) < 0.1 for frac in fractional_parts):
            return True

        # Heuristic 2: Spans orders of magnitude
        # Check if values span at least 2 orders of magnitude.
        log_min = math.floor(math.log10(vals[0]))
        log_max = math.floor(math.log10(vals[-1]))
        orders_spanned = log_max - log_min
        if orders_spanned >= 2:
            return True

        return False

    def cardinality(self) -> float:
        return len(self.values)

    @property
    def sort_values(self) -> bool:
        return self._sort_values

    @property
    def is_ordered(self) -> bool:
        return self._is_ordered

    @property
    def is_task(self) -> bool:
        return self._is_task

    @property
    def values(self) -> list[TParamValue]:
        return self._values

    @property
    def log_scale(self) -> bool:
        """Whether the parameter's values should be sampled from log space."""
        return self._log_scale

    def set_values(self, values: list[TParamValue]) -> ChoiceParameter:
        """Set the list of allowed values for parameter.

        Cast all input values to the parameter type.

        Args:
            values: New list of allowed values.
        """
        # A choice parameter with only one value is a FixedParameter.
        if not len(values) > 1:
            raise UserInputError(FIXED_CHOICE_PARAM_ERROR)
        self._values = self._cast_values(values)
        return self

    def add_values(self, values: list[TParamValue]) -> ChoiceParameter:
        """Add input list to the set of allowed values for parameter.

        Cast all input values to the parameter type.

        Args:
            values: Values being added to the allowed list.
        """
        self._values.extend(self._cast_values(values))
        return self

    def validate(self, value: TParamValue, raises: bool = False) -> bool:
        """Checks that the input is in the list of allowed values.

        Args:
            value: Value being checked.
            raises: If true, and validation fails, raises a UserInputError.

        Raises:
            UserInputError: If validation fails and raises is True.

        Returns:
            True if valid, False otherwise.
        """
        is_valid = value in self._values
        if raises and not is_valid:
            raise UserInputError(
                f"Value {value} is not in the list of allowed values: {self._values}."
            )
        return is_valid

    @property
    def dependents(self) -> dict[TParamValue, list[str]]:
        if not self.is_hierarchical:
            raise NotImplementedError(
                "Only hierarchical parameters support the `dependents` property."
            )
        return none_throws(self._dependents)

    @dependents.setter
    def dependents(self, dependents: dict[TParamValue, list[str]] | None) -> None:
        self._dependents = dependents

    def _cast_values(self, values: list[TParamValue]) -> list[TParamValue]:
        return [self.cast(value) for value in values]

    def clone(self) -> ChoiceParameter:
        return ChoiceParameter(
            name=self._name,
            parameter_type=self._parameter_type,
            values=self._values,
            is_ordered=self._is_ordered,
            is_task=self._is_task,
            is_fidelity=self._is_fidelity,
            target_value=self._target_value,
            sort_values=self._sort_values,
            log_scale=self._log_scale,
            dependents=deepcopy(self._dependents),
            bypass_cardinality_check=self._bypass_cardinality_check,
            backfill_value=self._backfill_value,
            default_value=self._default_value,
        )

    def __repr__(self) -> str:
        ret_val = self._base_repr()

        if self._dependents:
            ret_val += f", dependents={self._dependents}"

        return ret_val + ")"

    @property
    def available_flags(self) -> list[str]:
        """List of boolean attributes that can be set on this parameter."""
        return super().available_flags + [
            "is_ordered",
            "is_hierarchical",
            "is_task",
            "sort_values",
            "log_scale",
        ]

    @property
    def domain_repr(self) -> str:
        """Returns a string representation of the domain."""
        return f"values={self.values}"


class FixedParameter(Parameter):
    """Parameter object that specifies a single fixed value."""

    def __init__(
        self,
        name: str,
        parameter_type: ParameterType,
        value: TParamValue,
        is_fidelity: bool = False,
        target_value: TParamValue = None,
        dependents: dict[TParamValue, list[str]] | None = None,
        backfill_value: TParamValue = None,
        default_value: TParamValue = None,
    ) -> None:
        """Initialize FixedParameter

        Args:
            name: Name of the parameter.
            parameter_type: Enum indicating the type of parameter
                value (e.g. string, int).
            value: The fixed value of the parameter.
            is_fidelity: Whether this parameter is a fidelity parameter.
            target_value: Target value of this parameter if it is a fidelity.
            dependents: Optional mapping for parameters in hierarchical search
                spaces; format is { value -> list of dependent parameter names }.
            backfill_value: For parameters added to experiments that have already run.
                Used to backfill trials missing the parameter.
            default_value: For parameters disabled in experiments that have already
                run. Used as default value in modeling for future trials.
        """
        if is_fidelity and (target_value is None):
            raise UserInputError(
                "`target_value` should not be None for the fidelity parameter: "
                "{}".format(name)
            )

        self._name = name
        self._parameter_type = parameter_type
        self._value: TParamValue = self.cast(value)
        self._is_fidelity = is_fidelity
        self._target_value: TParamValue = (
            self.cast(target_value) if target_value is not None else None
        )
        self._backfill_value: TParamValue = (
            self.cast(backfill_value) if backfill_value is not None else None
        )
        self._default_value: TParamValue = (
            self.cast(default_value) if default_value is not None else None
        )
        # NOTE: We don't need to check that dependent parameters actually exist as
        # that is done in `SearchSpace` constructor.
        if dependents:
            if len(dependents) > 1 or next(iter(dependents.keys())) != self.value:
                raise UserInputError(
                    "The only expected key in `dependents` for fixed parameter "
                    f"{self.name}: {self.value}; got: {dependents}."
                )
        self._dependents = dependents

    def cardinality(self) -> float:
        return 1.0

    @property
    def value(self) -> TParamValue:
        return self._value

    def set_value(self, value: TParamValue) -> FixedParameter:
        self._value = self.cast(value)
        return self

    def validate(self, value: TParamValue, raises: bool = False) -> bool:
        """Checks that the input is equal to the fixed value.

        Args:
            value: Value being checked.
            raises: If true, and validation fails, raises a UserInputError.

        Raises:
            UserInputError: If validation fails and raises is True.

        Returns:
            True if valid, False otherwise.
        """
        is_valid = value == self._value
        if raises and not is_valid:
            raise UserInputError(
                f"Value {value} is not equal to the fixed value: {self._value}."
            )
        return is_valid

    @property
    def dependents(self) -> dict[TParamValue, list[str]]:
        if not self.is_hierarchical:
            raise NotImplementedError(
                "Only hierarchical parameters support the `dependents` property."
            )
        return none_throws(self._dependents)

    @dependents.setter
    def dependents(self, dependents: dict[TParamValue, list[str]] | None) -> None:
        self._dependents = dependents

    def clone(self) -> FixedParameter:
        return FixedParameter(
            name=self._name,
            parameter_type=self._parameter_type,
            value=self._value,
            is_fidelity=self._is_fidelity,
            target_value=self._target_value,
            dependents=self._dependents,
            backfill_value=self._backfill_value,
            default_value=self._default_value,
        )

    def __repr__(self) -> str:
        ret_val = self._base_repr()
        return ret_val + ")"

    @property
    def available_flags(self) -> list[str]:
        """List of boolean attributes that can be set on this parameter."""
        return super().available_flags + ["is_hierarchical"]

    @property
    def domain_repr(self) -> str:
        """Returns a string representation of the domain."""
        if self._parameter_type == ParameterType.STRING:
            return f"value='{self._value}'"
        else:
            return f"value={self._value}"


def get_dummy_value_for_parameter(param: Parameter) -> TParamValue:
    """Calculate the dummy value for a parameter (middle of domain).

    This is used when flattening hierarchical search spaces or filling in
    missing parameter values. The dummy value represents a "neutral" or
    "middle" value in the parameter's domain.

    Args:
        param: Parameter to calculate dummy value for.

    Returns:
        The middle of the parameter domain as the dummy value for the parameter.
    """

    if isinstance(param, FixedParameter):
        return param.value
    elif isinstance(param, ChoiceParameter):
        return param.values[len(param.values) // 2]
    elif isinstance(param, RangeParameter):
        lower, upper = float(param.lower), float(param.upper)
        if param.log_scale:
            log_lower, log_upper = math.log10(lower), math.log10(upper)
            log_mid = (log_upper + log_lower) / 2.0
            val = math.pow(10, log_mid)
        elif param.logit_scale:
            logit_lower, logit_upper = logit(lower).item(), logit(upper).item()
            logit_mid = (logit_upper + logit_lower) / 2.0
            val = expit(logit_mid).item()
        else:
            val = (upper + lower) / 2.0
        if param.parameter_type is ParameterType.INT:
            # This makes the distribution uniform after casting to int.
            val += 0.5
        return param.cast(val)
    else:
        raise NotImplementedError(f"Unhandled parameter type on parameter {param}.")


class DerivedParameter(Parameter):
    """A parameter that is a deterministic function of other parameters.

    Currently, only support for linear functions are implemented, but this is
    extendable to non-linear functions.
    """

    # pyre-fixme [13]: Uninitialized attribute [13]: Attribute `_intercept` is
    # declared in class `DerivedParameter` to have type `float` but is never
    # initialized.
    _intercept: float
    # pyre-fixme [13]: Uninitialized attribute [13]: Attribute
    # `_parameter_names_to_weights` is declared in class `DerivedParameter` to#
    # have type `typing.Dict[str, float]` but is never initialized.
    _parameter_names_to_weights: dict[str, float]

    def __init__(
        self,
        name: str,
        parameter_type: ParameterType,
        expression_str: str,
        is_fidelity: bool = False,
        target_value: TParamValue = None,
    ) -> None:
        """Initialize DerivedParameter

        Args:
            name: Name of the parameter.
            parameter_type: Enum indicating the type of parameter value. Expects
                "float", or "int". "bool" and "str" are not supported.
            expression_str: A string expression of the derived parameter definition.
            is_fidelity: Whether this parameter is a fidelity parameter.
            target_value: Target value of this parameter if it is a fidelity.
        """
        if is_fidelity:
            raise UnsupportedError("Derived parameters cannot be fidelity parameters.")
        elif target_value is not None:
            raise UnsupportedError(
                "Derived parameters do not support specifying a target value."
            )
        elif parameter_type not in (ParameterType.FLOAT, ParameterType.INT):
            raise UserInputError(
                "Derived parameters must be of type float or int, but got "
                f"{parameter_type}."
            )

        self.set_expression_str(expression_str=expression_str)
        self._name = name
        self._parameter_type = parameter_type
        self._is_fidelity = is_fidelity
        self._target_value = target_value

    def _parse_expression_str(self, expression_str: str) -> None:
        """Parse the expression str into parameter names and coefficients.

        Currently only linear functions are supported.
        """
        expression = sympify(sanitize_name(expression_str))
        if isinstance(expression, (Float, Integer)):
            raise UserInputError(
                "Derived parameters must have at least one parameter in "
                "`expression_str`."
            )
        elif not isinstance(expression, (Add, Mul, Symbol)):
            raise UnsupportedError("Only linear expressions are currently supported.")
        coefficient_dict = expression.as_coefficients_dict()
        self._intercept = float(coefficient_dict.pop(1, 0.0))
        parameter_names_to_weights = {}
        for name, coef in coefficient_dict.items():
            if not isinstance(name, Symbol):
                raise UnsupportedError(
                    "Only linear expressions are currently supported."
                )
            parameter_names_to_weights[unsanitize_name(str(name))] = float(coef)
        self._parameter_names_to_weights = parameter_names_to_weights

    @property
    def domain_repr(self) -> str:
        """Returns a string representation of the derived parameter."""
        terms = [
            f"{weight} * {name}"
            for name, weight in self._parameter_names_to_weights.items()
        ]
        if self._intercept != 0.0:
            terms.append(str(self._intercept))
        return "value=" + " + ".join(terms)

    @property
    def parameter_names_to_weights(self) -> dict[str, float]:
        return self._parameter_names_to_weights

    @property
    def expression_str(self) -> str:
        return self._expression_str

    def set_expression_str(self, expression_str: str) -> None:
        self._expression_str = expression_str
        self._parse_expression_str(expression_str=expression_str)

    @property
    def intercept(self) -> float:
        return self._intercept

    def cardinality(self) -> float:
        if self.parameter_type == ParameterType.FLOAT:
            return inf
        raise UnsupportedError(
            "cardinality for an integer DerivedParameter is not supported."
        )

    def compute(self, parameters: TParameterization) -> TParamValue:
        """Compute the value of the derived parameter.

        Args:
            parameterization: A dictionary mapping parameter names to values.

        Returns:
            The value of the derived parameter.
        """
        return self.cast(
            self._intercept
            + sum(
                self._parameter_names_to_weights[parameter_name]
                * float(parameters[parameter_name])
                for parameter_name in self._parameter_names_to_weights
            )
        )

    def validate(
        self,
        value: TParamValue,
        raises: bool = False,
        parameters: TParameterization | None = None,
    ) -> bool:
        """Checks that the input is equal to the derived value.

        Args:
            value: Value being checked.
            raises: If true, and validation fails, raises a UserInputError.
            parameters: A dictionary mapping parameter names to values. Used to
                compute the expected derived parameter value.

        Raises:
            UserInputError: If validation fails and raises is True.

        Returns:
            True if valid, False otherwise.
        """
        is_valid = False
        if parameters is None:
            if raises:
                raise UserInputError(
                    "Must specify `parameters` to validate a derived parameter"
                )
            return False
        expected_value = self.compute(parameters=parameters)
        is_valid = (
            abs(
                assert_is_instance(expected_value, TNumeric)
                - assert_is_instance(value, TNumeric)
            )
            < EPS
        )
        if raises and not is_valid:
            raise UserInputError(
                f"Value {value} is not equal to the expected derived"
                f" value: {expected_value}."
            )
        return is_valid

    def clone(self) -> DerivedParameter:
        return DerivedParameter(
            name=self._name,
            parameter_type=self._parameter_type,
            expression_str=self._expression_str,
            is_fidelity=self._is_fidelity,
            target_value=self._target_value,
        )

    def __repr__(self) -> str:
        ret_val = self._base_repr()
        return ret_val + ")"

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from abc import ABCMeta, abstractmethod, abstractproperty
from copy import deepcopy
from enum import Enum
from math import inf
from typing import cast, Dict, List, Optional, Tuple, Type, Union
from warnings import warn

from ax.core.types import TNumeric, TParamValue, TParamValueList
from ax.exceptions.core import AxParameterWarning, UserInputError
from ax.utils.common.base import SortableBase
from ax.utils.common.typeutils import not_none
from pyre_extensions import assert_is_instance

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
    BOOL: int = 0
    INT: int = 1
    FLOAT: int = 2
    STRING: int = 3

    @property
    def is_numeric(self) -> bool:
        return self == ParameterType.INT or self == ParameterType.FLOAT


TParameterType = Union[Type[int], Type[float], Type[str], Type[bool]]

# pyre: PARAMETER_PYTHON_TYPE_MAP is declared to have type
# pyre: `Dict[ParameterType, Union[Type[bool], Type[float], Type[int],
# pyre: Type[str]]]` but is used as type `Dict[ParameterType,
# pyre-fixme[9]: Type[Union[float, str]]]`.
PARAMETER_PYTHON_TYPE_MAP: Dict[ParameterType, TParameterType] = {
    ParameterType.INT: int,
    ParameterType.FLOAT: float,
    ParameterType.STRING: str,
    ParameterType.BOOL: bool,
}

SUPPORTED_PARAMETER_TYPES: Tuple[
    Union[Type[bool], Type[float], Type[int], Type[str]], ...
] = tuple(PARAMETER_PYTHON_TYPE_MAP.values())


# pyre-fixme[24]: Generic type `type` expects 1 type parameter, use `typing.Type` to
#  avoid runtime subscripting errors.
def _get_parameter_type(python_type: Type) -> ParameterType:
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

    def cast(self, value: TParamValue) -> TParamValue:
        if value is None:
            return None
        return self.python_type(value)

    @abstractmethod
    def validate(self, value: TParamValue) -> bool:
        pass

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
        return type(value) is self.python_type

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
    def parameter_type(self) -> ParameterType:
        return self._parameter_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def dependents(self) -> Dict[TParamValue, List[str]]:
        raise NotImplementedError(
            "Only choice hierarchical parameters are currently supported."
        )

    def clone(self) -> Parameter:
        # pyre-fixme[7]: Expected `Parameter` but got implicit return value of `None`.
        pass

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
    def available_flags(self) -> List[str]:
        """List of boolean attributes that can be set on this parameter."""
        return ["is_fidelity"]

    @property
    def summary_dict(
        self,
    ) -> Dict[str, Union[TParamValueList, TParamValue, str, List[str]]]:

        # Assemble dict.
        summary_dict = {
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
        digits: Optional[int] = None,
        is_fidelity: bool = False,
        target_value: TParamValue = None,
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
        self._lower: TNumeric = not_none(self.cast(lower))
        self._upper: TNumeric = not_none(self.cast(upper))
        self._log_scale = log_scale
        self._logit_scale = logit_scale
        self._is_fidelity = is_fidelity
        self._target_value: Optional[TNumeric] = self.cast(target_value)

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
        parameter_type: Optional[ParameterType] = None,
    ) -> None:
        if parameter_type and parameter_type not in (
            ParameterType.INT,
            ParameterType.FLOAT,
        ):
            raise UserInputError("RangeParameter type must be int or float.")

        upper = float(upper)
        if lower >= upper:
            raise UserInputError(
                f"Upper bound of {self.name} must be strictly larger than lower."
                f"Got: ({lower}, {upper})."
            )
        width: float = upper - lower
        if width < 100 * EPS:
            raise UserInputError(
                f"Parameter range ({width}) is very small and likely "
                "to cause numerical errors. Consider reparameterizing your "
                "problem by scaling the parameter."
            )
        if log_scale and logit_scale:
            raise UserInputError("Can't use both log and logit.")
        if log_scale and lower <= 0:
            raise UserInputError("Cannot take log when min <= 0.")
        if logit_scale and (lower <= 0 or upper >= 1):
            raise UserInputError("Logit requires lower > 0 and upper < 1")
        if not (self.is_valid_type(lower)) or not (self.is_valid_type(upper)):
            raise UserInputError(
                f"[{lower}, {upper}] is an invalid range for this parameter."
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
        self._upper = not_none(self.cast(value))

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
        self._lower = not_none(self.cast(value))

    @property
    def digits(self) -> Optional[int]:
        """Number of digits to round values to for float type.

        Upper and lower bound are re-cast after this property is changed.
        """
        return self._digits

    @property
    def log_scale(self) -> bool:
        """Whether the parameter's random values should be sampled from log space."""
        return self._log_scale

    @property
    def logit_scale(self) -> bool:
        """Whether the parameter's random values should be sampled from logit space."""
        return self._logit_scale

    def update_range(
        self, lower: Optional[float] = None, upper: Optional[float] = None
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

        cast_lower = not_none(self.cast(lower))
        cast_upper = not_none(self.cast(upper))
        self._validate_range_param(
            lower=cast_lower,
            upper=cast_upper,
            log_scale=self.log_scale,
            logit_scale=self.logit_scale,
        )
        self._lower = cast_lower
        self._upper = cast_upper
        return self

    def set_digits(self, digits: int) -> RangeParameter:
        self._digits = digits

        # Re-scale min and max to new digits definition
        cast_lower = not_none(self.cast(self._lower))
        cast_upper = not_none(self.cast(self._upper))
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

    def validate(self, value: TParamValue, tol: float = EPS) -> bool:
        """Returns True if input is a valid value for the parameter.

        Checks that value is of the right type and within
        the valid range for the parameter. Returns False if value is None.

        Args:
            value: Value being checked.
            tol: Absolute tolerance for floating point comparisons.

        Returns:
            True if valid, False otherwise.
        """
        if value is None:
            return False

        if not self.is_valid_type(value):
            return False
        # pyre-fixme[58]: `>=` is not supported for operand types `Union[bool,
        #  float, int, str]` and `float`.
        # pyre-fixme[58]: `<=` is not supported for operand types `Union[bool,
        #  float, int, str]` and `float`.
        return value >= self._lower - tol and value <= self._upper + tol

    def is_valid_type(self, value: TParamValue) -> bool:
        """Same as default except allows floats whose value is an int
        for Int parameters.
        """
        if not (isinstance(value, float) or isinstance(value, int)):
            return False

        # This might have issues with ints > 2^24
        if self.parameter_type is ParameterType.INT:
            return isinstance(value, int) or float(not_none(value)).is_integer()
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
        )

    def cast(self, value: TParamValue) -> Optional[TNumeric]:
        if value is None:
            return None
        if self.parameter_type is ParameterType.FLOAT and self._digits is not None:
            return round(float(value), not_none(self._digits))
        return assert_is_instance(self.python_type(value), TNumeric)

    def __repr__(self) -> str:
        ret_val = self._base_repr()

        if self._digits is not None:
            ret_val += f", digits={self._digits}"

        return ret_val + ")"

    @property
    def available_flags(self) -> List[str]:
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
            True.
        dependents: Optional mapping for parameters in hierarchical search
            spaces; format is { value -> list of dependent parameter names }.
    """

    def __init__(
        self,
        name: str,
        parameter_type: ParameterType,
        values: List[TParamValue],
        is_ordered: Optional[bool] = None,
        is_task: bool = False,
        is_fidelity: bool = False,
        target_value: TParamValue = None,
        sort_values: Optional[bool] = None,
        dependents: Optional[Dict[TParamValue, List[str]]] = None,
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
        self._target_value: TParamValue = self.cast(target_value)
        # A choice parameter with only one value is a FixedParameter.
        if not len(values) > 1:
            raise UserInputError(f"{self._name}({values}): {FIXED_CHOICE_PARAM_ERROR}")
        # Cap the number of possible values.
        if len(values) > MAX_VALUES_CHOICE_PARAM:
            raise UserInputError(
                f"`ChoiceParameter` with more than {MAX_VALUES_CHOICE_PARAM} values "
                "is not supported! Use a `RangeParameter` instead."
            )
        # Remove duplicate values.
        # Using dict to deduplicate here since set doesn't preserve order but dict does.
        if len(values) != len(dict_values := dict.fromkeys(values)):
            warn(
                f"Duplicate values found for ChoiceParameter {name}. "
                "Initializing the parameter with duplicate values removed. ",
                AxParameterWarning,
                stacklevel=2,
            )
            values = list(dict_values)

        self._is_ordered: bool = (
            is_ordered
            if is_ordered is not None
            else self._get_default_bool_and_warn(param_string="is_ordered")
        )
        # sort_values defaults to True if the parameter is not a string
        self._sort_values: bool = (
            sort_values
            if sort_values is not None
            else self._get_default_bool_and_warn(param_string="sort_values")
        )
        if self.sort_values:
            values = cast(List[TParamValue], sorted([not_none(v) for v in values]))
        self._values: List[TParamValue] = self._cast_values(values)

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

    def _get_default_bool_and_warn(self, param_string: str) -> bool:
        default_bool = self._parameter_type != ParameterType.STRING
        warn(
            f'`{param_string}` is not specified for `ChoiceParameter` "{self._name}". '
            f"Defaulting to `{default_bool}` for parameters of `ParameterType` "
            f"{self.parameter_type.name}. To override this behavior (or avoid this "
            f"warning), specify `{param_string}` during `ChoiceParameter` "
            "construction.",
            AxParameterWarning,
            stacklevel=3,
        )
        return default_bool

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
    def values(self) -> List[TParamValue]:
        return self._values

    def set_values(self, values: List[TParamValue]) -> ChoiceParameter:
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

    def add_values(self, values: List[TParamValue]) -> ChoiceParameter:
        """Add input list to the set of allowed values for parameter.

        Cast all input values to the parameter type.

        Args:
            values: Values being added to the allowed list.
        """
        self._values.extend(self._cast_values(values))
        return self

    def validate(self, value: TParamValue) -> bool:
        """Checks that the input is in the list of allowed values.

        Args:
            value: Value being checked.

        Returns:
            True if valid, False otherwise.
        """
        return value in self._values

    @property
    def dependents(self) -> Dict[TParamValue, List[str]]:
        if not self.is_hierarchical:
            raise NotImplementedError(
                "Only hierarchical parameters support the `dependents` property."
            )
        return not_none(self._dependents)

    def _cast_values(self, values: List[TParamValue]) -> List[TParamValue]:
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
            dependents=deepcopy(self._dependents),
        )

    def __repr__(self) -> str:
        ret_val = self._base_repr()

        if self._dependents:
            ret_val += f", dependents={self._dependents}"

        return ret_val + ")"

    @property
    def available_flags(self) -> List[str]:
        """List of boolean attributes that can be set on this parameter."""
        return super().available_flags + [
            "is_ordered",
            "is_hierarchical",
            "is_task",
            "sort_values",
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
        dependents: Optional[Dict[TParamValue, List[str]]] = None,
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
        self._target_value: TParamValue = self.cast(target_value)
        # NOTE: We don't need to check that dependent parameters actually exist as
        # that is done in `HierarchicalSearchSpace` constructor.
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

    def validate(self, value: TParamValue) -> bool:
        """Checks that the input is equal to the fixed value.

        Args:
            value: Value being checked.

        Returns:
            True if valid, False otherwise.
        """
        return value == self._value

    @property
    def dependents(self) -> Dict[TParamValue, List[str]]:
        if not self.is_hierarchical:
            raise NotImplementedError(
                "Only hierarchical parameters support the `dependents` property."
            )
        return not_none(self._dependents)

    def clone(self) -> FixedParameter:
        return FixedParameter(
            name=self._name,
            parameter_type=self._parameter_type,
            value=self._value,
            is_fidelity=self._is_fidelity,
            target_value=self._target_value,
            dependents=self._dependents,
        )

    def __repr__(self) -> str:
        ret_val = self._base_repr()
        return ret_val + ")"

    @property
    def available_flags(self) -> List[str]:
        """List of boolean attributes that can be set on this parameter."""
        return super().available_flags + ["is_hierarchical"]

    @property
    def domain_repr(self) -> str:
        """Returns a string representation of the domain."""
        if self._parameter_type == ParameterType.STRING:
            return f"value='{self._value}'"
        else:
            return f"value={self._value}"

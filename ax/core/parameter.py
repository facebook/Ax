#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from abc import ABCMeta, abstractmethod, abstractproperty
from enum import Enum
from typing import Dict, List, Optional, Type, Union

from ax.core.base import Base
from ax.core.types import TParamValue


FIXED_CHOICE_PARAM_ERROR = (
    "ChoiceParameters require multiple feasible values. "
    "Please use FixedParameter instead when setting a single possible value."
)


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


class Parameter(Base, metaclass=ABCMeta):
    _is_fidelity: bool = False

    def _cast(self, value: TParamValue) -> TParamValue:
        if value is None:
            return None
        return self.python_type(value)

    @abstractmethod
    def validate(self, value: TParamValue) -> bool:
        pass  # pragma: no cover

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

    @abstractproperty
    def parameter_type(self) -> ParameterType:
        pass  # pragma: no cover

    @abstractproperty
    def name(self) -> str:
        pass  # pragma: no cover

    def clone(self) -> "Parameter":
        pass  # pragma: no cover


class RangeParameter(Parameter):
    """Parameter object that specifies a continuous numerical range of values."""

    def __init__(
        self,
        name: str,
        parameter_type: ParameterType,
        lower: float,
        upper: float,
        log_scale: bool = False,
        digits: Optional[int] = None,
        is_fidelity: bool = False,
    ) -> None:
        """Initialize RangeParameter

        Args:
            name: Name of the parameter.
            parameter_type: Enum indicating the type of parameter
                value (e.g. string, int).
            lower: Lower bound of the parameter range.
            upper: Upper bound of the parameter range.
            log_scale: Whether to sample in the log space when drawing
                random values of the parameter.
            digits: Number of digits to round values to for float type.
            is_fidelity: Whether this parameter is a fidelity parameter.
        """
        self._name = name
        self._parameter_type = parameter_type
        self._digits = digits
        self._lower = self._cast(lower)
        self._upper = self._cast(upper)
        self._log_scale = log_scale
        self._is_fidelity = is_fidelity

        self._validate_range_param(
            parameter_type=parameter_type, lower=lower, upper=upper, log_scale=log_scale
        )

    def _validate_range_param(
        self,
        lower: TParamValue,
        upper: TParamValue,
        log_scale: bool,
        parameter_type: Optional[ParameterType] = None,
    ) -> None:
        if parameter_type and parameter_type not in (
            ParameterType.INT,
            ParameterType.FLOAT,
        ):
            raise ValueError("RangeParameter type must be int or float.")
        # pyre-fixme[16]: Optional type has no attribute `__ge__`.
        if lower >= upper:
            raise ValueError("max must be strictly larger than min.")
        # pyre-fixme[16]: Optional type has no attribute `__gt__`.
        if log_scale and lower <= 0:
            raise ValueError("Cannot take log when min <= 0.")
        if not (self.is_valid_type(lower)) or not (self.is_valid_type(upper)):
            raise ValueError(
                f"[{lower}, {upper}] is an invalid range for this parameter."
            )

    @property
    def parameter_type(self) -> ParameterType:
        return self._parameter_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def upper(self) -> float:
        """Upper bound of the parameter range.

        Value is cast to parameter type upon set and also validated
        to ensure the bound is strictly greater than lower bound.
        """
        return self._upper

    @property
    def lower(self) -> float:
        """Lower bound of the parameter range.

        Value is cast to parameter type upon set and also validated
        to ensure the bound is strictly less than upper bound.
        """
        return self._lower

    @property
    def digits(self) -> Optional[int]:
        """Number of digits to round values to for float type.

        Upper and lower bound are re-cast after this property is changed.
        """
        return self._digits

    @property
    def log_scale(self) -> bool:
        """Whether to sample in log space when drawing random values of the parameter.
        """
        return self._log_scale

    def update_range(
        self, lower: Optional[float] = None, upper: Optional[float] = None
    ) -> "RangeParameter":
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
        casted_lower = self._cast(lower)
        casted_upper = self._cast(upper)
        self._validate_range_param(
            lower=casted_lower, upper=casted_upper, log_scale=self.log_scale
        )
        self._lower = casted_lower
        self._upper = casted_upper
        return self

    def set_digits(self, digits: int) -> "RangeParameter":
        self._digits = digits

        # Re-scale min and max to new digits definition
        casted_lower = self._cast(self._lower)
        casted_upper = self._cast(self._upper)
        # pyre-fixme[16]: Optional type has no attribute `__ge__`.
        if casted_lower >= casted_upper:
            raise ValueError(
                f"Lower bound {casted_lower} is >= upper bound {casted_upper}."
            )

        self._lower = casted_lower
        self._upper = casted_upper
        return self

    def set_log_scale(self, log_scale: bool) -> "RangeParameter":
        self._log_scale = log_scale
        return self

    def validate(self, value: TParamValue) -> bool:
        """Returns True if input is a valid value for the parameter.

        Checks that value is of the right type and within
        the valid range for the parameter. Returns False if value is None.

        Args:
            value: Value being checked.

        Returns:
            True if valid, False otherwise.
        """
        if value is None:
            return False

        if not self.is_valid_type(value):
            return False
        return value >= self._lower and value <= self._upper

    def is_valid_type(self, value: TParamValue) -> bool:
        """Same as default except allows floats whose value is an int
           for Int parameters.
        """
        if not (isinstance(value, float) or isinstance(value, int)):
            return False

        # This might have issues with ints > 2^24
        if self.parameter_type is ParameterType.INT:
            return isinstance(value, int) or float(value).is_integer()
        return True

    def clone(self) -> "RangeParameter":
        return RangeParameter(
            name=self._name,
            parameter_type=self._parameter_type,
            lower=self._lower,
            upper=self._upper,
            log_scale=self._log_scale,
            digits=self._digits,
            is_fidelity=self._is_fidelity,
        )

    def _cast(self, value: TParamValue) -> TParamValue:
        if value is None:
            return None
        if self.parameter_type is ParameterType.FLOAT and self._digits is not None:
            return round(float(value), self._digits)
        return self.python_type(value)

    def __repr__(self) -> str:
        ret_val = (
            f"RangeParameter("
            f"name='{self._name}', "
            f"parameter_type={self.parameter_type.name}, "
            f"range=[{self._lower}, {self._upper}]"
        )
        if self._log_scale:
            ret_val += f", log_scale={self._log_scale}"

        if self._digits:
            ret_val += f", digits={self._digits}"

        return ret_val + ")"


class ChoiceParameter(Parameter):
    """Parameter object that specifies a discrete set of values."""

    def __init__(
        self,
        name: str,
        parameter_type: ParameterType,
        values: List[TParamValue],
        is_ordered: bool = False,
        is_task: bool = False,
        is_fidelity: bool = False,
    ) -> None:
        """Initialize ChoiceParameter.

        Args:
            name: Name of the parameter.
            parameter_type: Enum indicating the type of parameter
                value (e.g. string, int).
            values: List of allowed values for the parameter.
            is_ordered: If False, the parameter is a categorical variable.
            is_task: Treat the parameter as a task parameter for modeling.
        """
        self._is_ordered = is_ordered
        self._is_task = is_task
        self._is_fidelity = is_fidelity
        self._name = name
        self._parameter_type = parameter_type
        # A choice parameter with only one value is a FixedParameter.
        if not len(values) > 1:
            raise ValueError(FIXED_CHOICE_PARAM_ERROR)
        self._values = self._cast_values(values)

    @property
    def is_ordered(self) -> bool:
        return self._is_ordered

    @property
    def is_task(self) -> bool:
        return self._is_task

    @property
    def values(self) -> List[TParamValue]:
        return self._values

    @property
    def parameter_type(self) -> ParameterType:
        return self._parameter_type

    @property
    def name(self) -> str:
        return self._name

    def set_values(self, values: List[TParamValue]) -> "ChoiceParameter":
        """Set the list of allowed values for parameter.

        Cast all input values to the parameter type.

        Args:
            values: New list of allowed values.
        """
        # A choice parameter with only one value is a FixedParameter.
        if not len(values) > 1:
            raise ValueError(FIXED_CHOICE_PARAM_ERROR)
        self._values = self._cast_values(values)
        return self

    def add_values(self, values: List[TParamValue]) -> "ChoiceParameter":
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

    def _cast_values(self, values: List[TParamValue]) -> List[TParamValue]:
        return [self._cast(value) for value in values]

    def clone(self) -> "ChoiceParameter":
        return ChoiceParameter(
            name=self._name,
            parameter_type=self._parameter_type,
            values=self._values,
            is_task=self._is_task,
            is_fidelity=self._is_fidelity,
        )

    def __repr__(self) -> str:
        return (
            f"ChoiceParameter("
            f"name='{self._name}', "
            f"parameter_type={self.parameter_type.name}, "
            f"values={self._values})"
        )


class FixedParameter(Parameter):
    """Parameter object that specifies a single fixed value."""

    def __init__(
        self,
        name: str,
        parameter_type: ParameterType,
        value: TParamValue,
        is_fidelity: bool = False,
    ) -> None:
        """Initialize FixedParameter

        Args:
            name: Name of the parameter.
            parameter_type: Enum indicating the type of parameter
                value (e.g. string, int).
            value: The fixed value of the parameter.
        """
        self._name = name
        self._parameter_type = parameter_type
        self._value = self._cast(value)
        self._is_fidelity = is_fidelity

    @property
    def value(self) -> TParamValue:
        return self._value

    @property
    def parameter_type(self) -> ParameterType:
        return self._parameter_type

    @property
    def name(self) -> str:
        return self._name

    def set_value(self, value: TParamValue) -> "FixedParameter":
        self._value = self._cast(value)
        return self

    def validate(self, value: TParamValue) -> bool:
        """Checks that the input is equal to the fixed value.

        Args:
            value: Value being checked.

        Returns:
            True if valid, False otherwise.
        """
        return value == self._value

    def clone(self) -> "FixedParameter":
        return FixedParameter(
            name=self._name,
            parameter_type=self._parameter_type,
            value=self._value,
            is_fidelity=self._is_fidelity,
        )

    def __repr__(self) -> str:
        return (
            f"FixedParameter("
            f"name='{self._name}', "
            f"parameter_type={self.parameter_type.name}, "
            f"value={self._value})"
        )

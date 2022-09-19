#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type, Union
from warnings import warn

from ax.core.types import TParamValue
from ax.exceptions.core import UserInputError
from ax.utils.common.base import SortableBase
from ax.utils.common.typeutils import not_none


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
    _target_value: Optional[TParamValue] = None
    _parameter_type: ParameterType

    def cast(self, value: TParamValue) -> TParamValue:
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

    @property
    def is_hierarchical(self) -> bool:
        return (
            isinstance(self, (ChoiceParameter, FixedParameter))
            and self._dependents is not None
        )

    @property
    def target_value(self) -> Optional[TParamValue]:
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
        pass  # pragma: no cover

    @property
    def _unique_id(self) -> str:
        return str(self)

    def _base_repr(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self._name}', "
            f"parameter_type={self.parameter_type.name}, "
        )


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
        target_value: Optional[TParamValue] = None,
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
        self._parameter_type = parameter_type
        self._digits = digits
        # pyre-fixme[4]: Attribute must be annotated.
        self._lower = self.cast(lower)
        # pyre-fixme[4]: Attribute must be annotated.
        self._upper = self.cast(upper)
        self._log_scale = log_scale
        self._logit_scale = logit_scale
        self._is_fidelity = is_fidelity
        # pyre-fixme[4]: Attribute must be annotated.
        self._target_value = self.cast(target_value)

        self._validate_range_param(
            parameter_type=parameter_type,
            lower=lower,
            upper=upper,
            log_scale=log_scale,
            logit_scale=logit_scale,
        )

    def _validate_range_param(
        self,
        lower: TParamValue,
        upper: TParamValue,
        log_scale: bool,
        logit_scale: bool,
        parameter_type: Optional[ParameterType] = None,
    ) -> None:
        if parameter_type and parameter_type not in (
            ParameterType.INT,
            ParameterType.FLOAT,
        ):
            raise UserInputError("RangeParameter type must be int or float.")
        # pyre-fixme[58]: `>=` is not supported for operand types `Union[None, bool,
        #  float, int, str]` and `Union[None, bool, float, int, str]`.
        if lower >= upper:
            raise UserInputError(
                f"Upper bound of {self.name} must be strictly larger than lower."
                f"Got: ({lower}, {upper})."
            )
        if log_scale and logit_scale:
            raise UserInputError("Can't use both log and logit.")
        # pyre-fixme[58]: `<=` is not supported for operand types `Union[None, bool,
        #  float, int, str]` and `int`.
        if log_scale and lower <= 0:
            raise UserInputError("Cannot take log when min <= 0.")
        # pyre-fixme[58]: `<=` is not supported for operand types `Union[None, bool,
        #  float, int, str]` and `int`.
        if logit_scale and (lower <= 0 or upper >= 1):
            raise UserInputError("Logit requires lower > 0 and upper < 1")
        if not (self.is_valid_type(lower)) or not (self.is_valid_type(upper)):
            raise UserInputError(
                f"[{lower}, {upper}] is an invalid range for this parameter."
            )

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

    def set_digits(self, digits: int) -> RangeParameter:
        self._digits = digits

        # Re-scale min and max to new digits definition
        cast_lower = self.cast(self._lower)
        cast_upper = self.cast(self._upper)
        # pyre-fixme[58]: `>=` is not supported for operand types `Union[None, bool,
        #  float, int, str]` and `Union[None, bool, float, int, str]`.
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
            # pyre-fixme[6]: Expected `Union[_SupportsIndex, bytearray, bytes, str,
            #  typing.SupportsFloat]` for 1st param but got `Union[None, float, str]`.
            return isinstance(value, int) or float(value).is_integer()
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

    def cast(self, value: TParamValue) -> TParamValue:
        if value is None:
            return None
        if self.parameter_type is ParameterType.FLOAT and self._digits is not None:
            # pyre-fixme[6]: Expected `None` for 2nd param but got `Optional[int]`.
            return round(float(value), self._digits)
        return self.python_type(value)

    def __repr__(self) -> str:
        ret_val = self._base_repr()
        ret_val += f"range=[{self._lower}, {self._upper}]"

        if self._log_scale:
            ret_val += f", log_scale={self._log_scale}"

        if self._logit_scale:
            ret_val += f", logit_scale={self._logit_scale}"

        if self._digits:
            ret_val += f", digits={self._digits}"

        if self.is_fidelity:
            ret_val += (
                f", fidelity={self.is_fidelity}, target_value={self.target_value}"
            )

        return ret_val + ")"


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
        target_value: Target value of this parameter if it's fidelity.
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
        target_value: Optional[TParamValue] = None,
        sort_values: Optional[bool] = None,
        dependents: Optional[Dict[TParamValue, List[str]]] = None,
    ) -> None:
        if is_fidelity and (target_value is None):
            raise UserInputError(
                "`target_value` should not be None for the fidelity parameter: "
                "{}".format(name)
            )

        self._name = name
        self._parameter_type = parameter_type
        self._is_task = is_task
        self._is_fidelity = is_fidelity
        # pyre-fixme[4]: Attribute must be annotated.
        self._target_value = self.cast(target_value)
        # A choice parameter with only one value is a FixedParameter.
        if not len(values) > 1:
            raise UserInputError(f"{self._name}({values}): {FIXED_CHOICE_PARAM_ERROR}")
        # pyre-fixme[4]: Attribute must be annotated.
        self._values = self._cast_values(values)
        # pyre-fixme[4]: Attribute must be annotated.
        self._is_ordered = (
            is_ordered
            if is_ordered is not None
            else self._get_default_bool_and_warn(param_string="is_ordered")
        )
        # sort_values defaults to True if the parameter is not a string
        # pyre-fixme[4]: Attribute must be annotated.
        self._sort_values = (
            sort_values
            if sort_values is not None
            else self._get_default_bool_and_warn(param_string="sort_values")
        )
        if self.sort_values:
            # pyre-ignore[6]: values/self._values expects List[Union[None, bool, float,
            # int, str]] but sorted() takes/returns
            # List[Variable[_typeshed.SupportsLessThanT (bound to
            # _typeshed.SupportsLessThan)]]
            self._values = self._cast_values(sorted(values))
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
            "construction."
        )
        return default_bool

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
            dependents=self._dependents,
        )

    def __repr__(self) -> str:
        ret_val = self._base_repr()
        ret_val += f"values={self._values}, "
        ret_val += f"is_ordered={self._is_ordered}, "
        ret_val += f"sort_values={self._sort_values}"

        if self._is_task:
            ret_val += f", is_task={self._is_task}"

        if self._is_fidelity:
            tval_rep = self.target_value
            if self.parameter_type == ParameterType.STRING:
                tval_rep = f"'{tval_rep}'"
            ret_val += f", is_fidelity={self.is_fidelity}, target_value={tval_rep}"

        if self._dependents:
            ret_val += f", dependents={self._dependents}"

        return ret_val + ")"


class FixedParameter(Parameter):
    """Parameter object that specifies a single fixed value."""

    def __init__(
        self,
        name: str,
        parameter_type: ParameterType,
        value: TParamValue,
        is_fidelity: bool = False,
        target_value: Optional[TParamValue] = None,
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
        # pyre-fixme[4]: Attribute must be annotated.
        self._value = self.cast(value)
        self._is_fidelity = is_fidelity
        # pyre-fixme[4]: Attribute must be annotated.
        self._target_value = self.cast(target_value)
        # NOTE: We don't need to check that dependent parameters actually exist as
        # that is done in `HierarchicalSearchSpace` constructor.
        if dependents:
            if len(dependents) > 1 or next(iter(dependents.keys())) != self.value:
                raise UserInputError(
                    "The only expected key in `dependents` for fixed parameter "
                    f"{self.name}: {self.value}; got: {dependents}."
                )
        self._dependents = dependents

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

        if self._parameter_type == ParameterType.STRING:
            ret_val += f"value='{self._value}'"
        else:
            ret_val += f"value={self._value}"

        if self._is_fidelity:
            ret_val += (
                f", fidelity={self.is_fidelity}, target_value={self.target_value}"
            )
        return ret_val + ")"

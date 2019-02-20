#!/usr/bin/env python3

from datetime import datetime
from typing import Any

import numpy as np
from ae.lazarus.ae.utils.common.equality import (
    datetime_equals,
    equality_typechecker,
    list_equals,
)


class Base(object):
    """Metaclass for core AE classes."""

    @equality_typechecker
    def __eq__(self, other: "Base"):
        for field in self.__dict__.keys():
            self_val = getattr(self, field)
            other_val = getattr(other, field)

            self_val = _numpy_type_to_python_type(self_val)
            other_val = _numpy_type_to_python_type(other_val)

            if type(self_val) != type(other_val):
                return False

            if field == "_experiment":
                # prevent infinite loop when checking equality of Trials
                equal = self_val.name == other_val.name
            elif isinstance(self_val, list):
                equal = list_equals(self_val, other_val)
            elif isinstance(self_val, np.ndarray):
                equal = np.array_equal(self_val, other_val)
            elif isinstance(self_val, datetime):
                equal = datetime_equals(self_val, other_val)
            else:
                equal = self_val == other_val
            if not equal:
                return False
        return True


def _numpy_type_to_python_type(value: Any) -> Any:
    """If `value` is a numpy int or float, coerce to a python int or float.
    This is necessary because some of our transforms return numpy values.
    """
    # pyre-fixme[20]: Call `object.__eq__` expects argument `o`.
    if type(value) == np.int64:
        value = int(value)  # pragma: nocover (covered by generator tests)
    # pyre-fixme[20]: Call `object.__eq__` expects argument `o`.
    if type(value) == np.float64:
        value = float(value)  # pragma: nocover  (covered by generator tests)
    return value

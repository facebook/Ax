#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from datetime import datetime

import numpy as np
import pandas as pd
from ax.utils.common.equality import (
    datetime_equals,
    equality_typechecker,
    same_elements,
)
from ax.utils.common.typeutils import numpy_type_to_python_type


class Base(object):
    """Metaclass for core Ax classes."""

    @equality_typechecker
    def __eq__(self, other: "Base"):
        for field in self.__dict__.keys():
            self_val = getattr(self, field)
            other_val = getattr(other, field)

            self_val = numpy_type_to_python_type(self_val)
            other_val = numpy_type_to_python_type(other_val)

            if type(self_val) != type(other_val):
                return False

            if field == "_experiment":
                # prevent infinite loop when checking equality of Trials
                equal = self_val.name == other_val.name
            elif isinstance(self_val, list):
                equal = same_elements(self_val, other_val)
            elif isinstance(self_val, np.ndarray):
                equal = np.array_equal(self_val, other_val)
            elif isinstance(self_val, datetime):
                equal = datetime_equals(self_val, other_val)
            elif isinstance(self_val, pd.DataFrame):
                equal = self_val.equals(other_val)
            else:
                equal = self_val == other_val
            if not equal:
                return False
        return True

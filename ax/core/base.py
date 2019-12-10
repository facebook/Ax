#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    def __eq__(self, other: "Base") -> bool:
        for field in self.__dict__.keys():
            self_val = getattr(self, field)
            other_val = getattr(other, field)
            self_val = numpy_type_to_python_type(self_val)
            other_val = numpy_type_to_python_type(other_val)

            if type(self_val) != type(other_val):
                return False

            if field == "_experiment":
                # prevent infinite loop when checking equality of Trials
                equal = self_val is other_val is None or (
                    self_val._name == other_val._name
                )
            elif field == "_model":  # pragma: no cover (tested in modelbridge)
                # TODO[T52643706]: replace with per-`ModelBridge` method like
                # `equivalent_models`, to compare models more meaningfully.
                if not hasattr(self_val, "model"):
                    equal = not hasattr(other_val, "model")
                else:
                    # If model bridges have a `model` attribute, the types of the
                    # values of those attributes should be equal if the model
                    # bridge is the same.
                    equal = isinstance(self_val.model, type(other_val.model))
            elif isinstance(self_val, list):
                equal = same_elements(self_val, other_val)
            elif isinstance(self_val, dict):
                equal = sorted(self_val.keys()) == sorted(other_val.keys())
                equal = equal and same_elements(
                    list(self_val.values()), list(other_val.values())
                )
            elif isinstance(self_val, np.ndarray):
                equal = np.array_equal(self_val, other_val)
            elif isinstance(self_val, datetime):
                equal = datetime_equals(self_val, other_val)
            elif isinstance(self_val, float):
                equal = np.isclose(self_val, other_val)
            elif isinstance(self_val, pd.DataFrame):
                try:
                    if self_val.empty and other_val.empty:
                        equal = True
                    else:
                        pd.testing.assert_frame_equal(
                            self_val.sort_index(axis=1),
                            other_val.sort_index(axis=1),
                            check_exact=False,
                        )
                        equal = True
                except AssertionError:
                    equal = False
            else:
                equal = self_val == other_val
            if not equal:
                return False
        return True

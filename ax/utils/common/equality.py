#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from ax.utils.common.typeutils import numpy_type_to_python_type


def equality_typechecker(eq_func: Callable) -> Callable:
    """A decorator to wrap all __eq__ methods to ensure that the inputs
    are of the right type.
    """

    # no type annotation for now; breaks sphinx-autodoc-typehints
    def _type_safe_equals(self, other):
        if not isinstance(other, self.__class__):
            return False
        return eq_func(self, other)

    return _type_safe_equals


def same_elements(list1: List[Any], list2: List[Any]) -> bool:
    """Compare equality of two lists of core Ax objects.

    Assumptions:
        -- The contents of each list are types that implement __eq__
        -- The lists do not contain duplicates

    Checking equality is then the same as checking that the lists are the same
    length, and that one is a subset of the other.
    """

    if len(list1) != len(list2):
        return False

    for item1 in list1:
        found = False
        for item2 in list2:
            if isinstance(item1, np.ndarray) or isinstance(item2, np.ndarray):
                if (
                    isinstance(item1, np.ndarray)
                    and isinstance(item2, np.ndarray)
                    and np.array_equal(item1, item2)
                ):
                    found = True
                    break
            elif item1 == item2:
                found = True
                break
        if not found:
            return False

    return True


def datetime_equals(dt1: Optional[datetime], dt2: Optional[datetime]) -> bool:
    """Compare equality of two datetimes, ignoring microseconds."""
    if not dt1 and not dt2:
        return True
    if not (dt1 and dt2):
        return False
    return dt1.replace(microsecond=0) == dt2.replace(microsecond=0)


def dataframe_equals(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """Compare equality of two pandas dataframes."""
    try:
        if df1.empty and df2.empty:
            equal = True
        else:
            pd.testing.assert_frame_equal(
                df1.sort_index(axis=1), df2.sort_index(axis=1), check_exact=False
            )
            equal = True
    except AssertionError:
        equal = False

    return equal


def object_attribute_dicts_equal(
    one_dict: Dict[str, Any], other_dict: Dict[str, Any]
) -> bool:
    """Utility to check if all items in attribute dicts of two Ax objects
    are the same.


    NOTE: Special-cases some Ax object attributes, like "_experiment" or
    "_model", where full equality is hard to check.
    """
    for field in one_dict:
        one_val = one_dict.get(field)
        other_val = other_dict.get(field)
        one_val = numpy_type_to_python_type(one_val)
        other_val = numpy_type_to_python_type(other_val)

        if type(one_val) != type(other_val):
            return False

        if field == "_experiment":
            # prevent infinite loop when checking equality of Trials
            equal = one_val is other_val is None or (one_val._name == other_val._name)
        elif field == "_model":  # pragma: no cover (tested in modelbridge)
            # TODO[T52643706]: replace with per-`ModelBridge` method like
            # `equivalent_models`, to compare models more meaningfully.
            if not hasattr(one_val, "model"):
                equal = not hasattr(other_val, "model")
            else:
                # If model bridges have a `model` attribute, the types of the
                # values of those attributes should be equal if the model
                # bridge is the same.
                equal = isinstance(one_val.model, type(other_val.model))
        elif isinstance(one_val, list):
            equal = same_elements(one_val, other_val)
        elif isinstance(one_val, dict):
            equal = sorted(one_val.keys()) == sorted(other_val.keys())
            equal = equal and same_elements(
                list(one_val.values()), list(other_val.values())
            )
        elif isinstance(one_val, np.ndarray):
            equal = np.array_equal(one_val, other_val)
        elif isinstance(one_val, datetime):
            equal = datetime_equals(one_val, other_val)
        elif isinstance(one_val, float):
            equal = np.isclose(one_val, other_val)
        elif isinstance(one_val, pd.DataFrame):
            equal = dataframe_equals(one_val, other_val)
        else:
            equal = one_val == other_val
        if not equal:
            return False
    return True

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ax.utils.common.typeutils import numpy_type_to_python_type


# pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
def equality_typechecker(eq_func: Callable) -> Callable:
    """A decorator to wrap all __eq__ methods to ensure that the inputs
    are of the right type.
    """

    # no type annotation for now; breaks sphinx-autodoc-typehints
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _type_safe_equals(self, other):
        if not isinstance(other, self.__class__):
            return False
        return eq_func(self, other)

    return _type_safe_equals


# pyre-fixme[2]: Parameter annotation cannot contain `Any`.
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
    one_dict: Dict[str, Any], other_dict: Dict[str, Any], skip_db_id_check: bool = False
) -> bool:
    """Utility to check if all items in attribute dicts of two Ax objects
    are the same.


    NOTE: Special-cases some Ax object attributes, like "_experiment" or
    "_model", where full equality is hard to check.

    Args:
        one_dict: First object's attribute dict (``obj.__dict__``).
        other_dict: Second object's attribute dict (``obj.__dict__``).
        skip_db_id_check: If ``True``, will exclude the ``db_id`` attributes from the
            equality check. Useful for ensuring that all attributes of an object are
            equal except the ids, with which one or both of them are saved to the
            database (e.g. if confirming an object before it was saved, to the version
            reloaded from the DB).
    """
    unequal_type, unequal_value = object_attribute_dicts_find_unequal_fields(
        one_dict=one_dict, other_dict=other_dict, skip_db_id_check=skip_db_id_check
    )
    return not bool(unequal_type or unequal_value)


# pyre-fixme[3]: Return annotation cannot contain `Any`.
def object_attribute_dicts_find_unequal_fields(
    one_dict: Dict[str, Any],
    other_dict: Dict[str, Any],
    fast_return: bool = True,
    skip_db_id_check: bool = False,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Tuple[Any, Any]]]:
    """Utility for finding out what attributes of two objects' attribute dicts
    are unequal.

    Args:
        one_dict: First object's attribute dict (``obj.__dict__``).
        other_dict: Second object's attribute dict (``obj.__dict__``).
        fast_return: Boolean representing whether to return as soon as a
            single unequal attribute was found or to iterate over all attributes
            and collect all unequal ones.
        skip_db_id_check: If ``True``, will exclude the ``db_id`` attributes from the
            equality check. Useful for ensuring that all attributes of an object are
            equal except the ids, with which one or both of them are saved to the
            database (e.g. if confirming an object before it was saved, to the version
            reloaded from the DB).

    Returns:
        Two dictionaries:
            - attribute name to attribute values of unequal type (as a tuple),
            - attribute name to attribute values of unequal value (as a tuple).
    """
    unequal_type, unequal_value = {}, {}
    for field in one_dict:
        one_val = one_dict.get(field)
        other_val = other_dict.get(field)
        one_val = numpy_type_to_python_type(one_val)
        other_val = numpy_type_to_python_type(other_val)

        if type(one_val) != type(other_val):
            unequal_type[field] = (one_val, other_val)
            if fast_return:
                return unequal_type, unequal_value

        if field == "_experiment":
            # prevent infinite loop when checking equality of Trials
            equal = one_val is other_val is None or (one_val._name == other_val._name)
        elif field == "analysis_scheduler":
            # prevent infinite loop when checking equality of analysis runs
            equal = one_val is other_val is None or (one_val.db_id == other_val.db_id)
        elif field == "_db_id":
            equal = skip_db_id_check or one_val == other_val
        elif field == "_model":  # pragma: no cover (tested in modelbridge)
            # TODO[T52643706]: replace with per-`ModelBridge` method like
            # `equivalent_models`, to compare models more meaningfully.
            if not hasattr(one_val, "model") or not hasattr(other_val, "model"):
                equal = not hasattr(other_val, "model") and not hasattr(
                    other_val, "model"
                )
            else:
                # If model bridges have a `model` attribute, the types of the
                # values of those attributes should be equal if the model
                # bridge is the same.
                equal = (
                    hasattr(one_val, "model")
                    and hasattr(other_val, "model")
                    and isinstance(one_val.model, type(other_val.model))
                )

        elif isinstance(one_val, list):
            equal = isinstance(other_val, list) and same_elements(one_val, other_val)
        elif isinstance(one_val, dict):
            equal = isinstance(other_val, dict) and sorted(one_val.keys()) == sorted(
                other_val.keys()
            )
            equal = equal and same_elements(
                list(one_val.values()), list(other_val.values())
            )
        elif isinstance(one_val, np.ndarray):
            equal = np.array_equal(one_val, other_val, equal_nan=True)
        elif isinstance(one_val, datetime):
            equal = datetime_equals(one_val, other_val)
        elif isinstance(one_val, float):
            equal = np.isclose(one_val, other_val)
        elif isinstance(one_val, pd.DataFrame):
            equal = dataframe_equals(one_val, other_val)
        else:
            equal = one_val == other_val
        if not equal:
            unequal_value[field] = (one_val, other_val)
            if fast_return:
                return unequal_type, unequal_value
    return unequal_type, unequal_value

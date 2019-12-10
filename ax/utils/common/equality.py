#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime
from typing import Any, Callable, List, Optional

import numpy as np


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

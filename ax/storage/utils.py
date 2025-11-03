#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import enum
from hashlib import md5

from ax.core.evaluations_to_data import DataType  # noqa F401


class DomainType(enum.Enum):
    """Class for enumerating domain types."""

    FIXED = 0
    RANGE = 1
    CHOICE = 2
    ENVIRONMENTAL_RANGE = 3
    DERIVED = 4


class MetricIntent(enum.Enum):
    """Class for enumerating metric use types."""

    OBJECTIVE = "objective"
    MULTI_OBJECTIVE = "multi_objective"
    SCALARIZED_OBJECTIVE = "scalarized_objective"
    # Additional objective is not yet supported in Ax open-source.
    ADDITIONAL_OBJECTIVE = "additional_objective"
    OUTCOME_CONSTRAINT = "outcome_constraint"
    SCALARIZED_OUTCOME_CONSTRAINT = "scalarized_outcome_constraint"
    OBJECTIVE_THRESHOLD = "objective_threshold"
    TRACKING = "tracking"
    RISK_MEASURE = "risk_measure"  # DEPRECATED


class ParameterConstraintType(enum.Enum):
    """Class for enumerating parameter constraint types.

    Linear constraint is base type whereas other constraint types are
    special types of linear constraints.
    """

    LINEAR = 0
    ORDER = 1
    SUM = 2
    DISTRIBUTION = 3  # DEPRECATED


def stable_hash(s: str) -> int:
    """Return an integer hash of a string that is consistent across re-invocations
    of the interpreter (unlike the built-in hash, which is salted by default).

    Args:
        s (str): String to hash.
    Returns:

        int: Hash, converted to an integer.
    """
    return int(md5(s.encode("utf-8")).hexdigest(), 16)

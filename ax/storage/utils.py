#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import enum
from hashlib import md5

from ax.core.formatting_utils import DataType  # noqa F401


class DomainType(enum.Enum):
    """Class for enumerating domain types."""

    # pyre-fixme[35]: Target cannot be annotated.
    FIXED: int = 0
    # pyre-fixme[35]: Target cannot be annotated.
    RANGE: int = 1
    # pyre-fixme[35]: Target cannot be annotated.
    CHOICE: int = 2
    # pyre-fixme[35]: Target cannot be annotated.
    ENVIRONMENTAL_RANGE: int = 3


class MetricIntent(enum.Enum):
    """Class for enumerating metric use types."""

    # pyre-fixme[35]: Target cannot be annotated.
    OBJECTIVE: str = "objective"
    # pyre-fixme[35]: Target cannot be annotated.
    MULTI_OBJECTIVE: str = "multi_objective"
    # pyre-fixme[35]: Target cannot be annotated.
    SCALARIZED_OBJECTIVE: str = "scalarized_objective"
    # Additional objective is not yet supported in Ax open-source.
    # pyre-fixme[35]: Target cannot be annotated.
    ADDITIONAL_OBJECTIVE: str = "additional_objective"
    # pyre-fixme[35]: Target cannot be annotated.
    OUTCOME_CONSTRAINT: str = "outcome_constraint"
    # pyre-fixme[35]: Target cannot be annotated.
    SCALARIZED_OUTCOME_CONSTRAINT: str = "scalarized_outcome_constraint"
    # pyre-fixme[35]: Target cannot be annotated.
    OBJECTIVE_THRESHOLD: str = "objective_threshold"
    # pyre-fixme[35]: Target cannot be annotated.
    TRACKING: str = "tracking"
    # pyre-fixme[35]: Target cannot be annotated.
    RISK_MEASURE: str = "risk_measure"


class ParameterConstraintType(enum.Enum):
    """Class for enumerating parameter constraint types.

    Linear constraint is base type whereas other constraint types are
    special types of linear constraints.
    """

    # pyre-fixme[35]: Target cannot be annotated.
    LINEAR: int = 0
    # pyre-fixme[35]: Target cannot be annotated.
    ORDER: int = 1
    # pyre-fixme[35]: Target cannot be annotated.
    SUM: int = 2
    # pyre-fixme[35]: Target cannot be annotated.
    DISTRIBUTION: int = 3


def stable_hash(s: str) -> int:
    """Return an integer hash of a string that is consistent across re-invocations
    of the interpreter (unlike the built-in hash, which is salted by default).

    Args:
        s (str): String to hash.
    Returns:

        int: Hash, converted to an integer.
    """
    return int(md5(s.encode("utf-8")).hexdigest(), 16)

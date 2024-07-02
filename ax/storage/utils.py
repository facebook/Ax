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

    FIXED: int = 0
    RANGE: int = 1
    CHOICE: int = 2
    ENVIRONMENTAL_RANGE: int = 3


class MetricIntent(enum.Enum):
    """Class for enumerating metric use types."""

    OBJECTIVE: str = "objective"
    MULTI_OBJECTIVE: str = "multi_objective"
    SCALARIZED_OBJECTIVE: str = "scalarized_objective"
    # Additional objective is not yet supported in Ax open-source.
    ADDITIONAL_OBJECTIVE: str = "additional_objective"
    OUTCOME_CONSTRAINT: str = "outcome_constraint"
    SCALARIZED_OUTCOME_CONSTRAINT: str = "scalarized_outcome_constraint"
    OBJECTIVE_THRESHOLD: str = "objective_threshold"
    TRACKING: str = "tracking"
    RISK_MEASURE: str = "risk_measure"


class ParameterConstraintType(enum.Enum):
    """Class for enumerating parameter constraint types.

    Linear constraint is base type whereas other constraint types are
    special types of linear constraints.
    """

    LINEAR: int = 0
    ORDER: int = 1
    SUM: int = 2
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


class AnalysisType(enum.Enum):
    """Class for enumerating different experiment analysis types."""

    ANALYSIS: str = "analysis"
    PLOTLY_VISUALIZATION: str = "plotly_visualization"

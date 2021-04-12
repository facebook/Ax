#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum

from ax.core.experiment import DataType  # noqa F401


class DomainType(enum.Enum):
    """Class for enumerating domain types."""

    FIXED: int = 0
    RANGE: int = 1
    CHOICE: int = 2


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


class ParameterConstraintType(enum.Enum):
    """Class for enumerating parameter constraint types.

    Linear constraint is base type whereas other constraint types are
    special types of linear constraints.
    """

    LINEAR: int = 0
    ORDER: int = 1
    SUM: int = 2

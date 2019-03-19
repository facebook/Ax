#!/usr/bin/env python3

import enum
from typing import Dict, List, NamedTuple


class DomainType(enum.Enum):
    """Class for enumerating domain types."""

    FIXED: int = 0
    RANGE: int = 1
    CHOICE: int = 2


class MetricIntent(enum.Enum):
    """Class for enumerating metric use types."""

    OBJECTIVE: str = "objective"
    OUTCOME_CONSTRAINT: str = "outcome_constraint"
    TRACKING: str = "tracking"


class ParameterConstraintType(enum.Enum):
    """Class for enumerating parameter constraint types.

    Linear constraint is base type whereas other constraint types are
    special types of linear constraints.
    """

    LINEAR: int = 0
    ORDER: int = 1
    SUM: int = 2


class EncodeDecodeFieldsMap(NamedTuple):
    python_only: List[str] = []
    encoded_only: List[str] = []
    python_to_encoded: Dict[str, str] = {}


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text

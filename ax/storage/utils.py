#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import enum
from collections import OrderedDict
from collections.abc import Mapping
from hashlib import md5

from ax.core.data import combine_dfs_favoring_recent, Data


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
    PREFERENCE_OBJECTIVE = "preference_objective"
    SCALARIZED_OBJECTIVE = "scalarized_objective"
    # Additional objective is not yet supported in Ax open-source.
    ADDITIONAL_OBJECTIVE = "additional_objective"
    OUTCOME_CONSTRAINT = "outcome_constraint"
    SCALARIZED_OUTCOME_CONSTRAINT = "scalarized_outcome_constraint"
    OBJECTIVE_THRESHOLD = "objective_threshold"
    TRACKING = "tracking"
    RISK_MEASURE = "risk_measure"  # DEPRECATED


# Keys for PreferenceOptimizationConfig properties stored in SQAMetric.properties
PREFERENCE_PROFILE_NAME = "preference_profile_name"
EXPECT_RELATIVIZED_OUTCOMES = "expect_relativized_outcomes"


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


def combine_datas_on_data_by_trial(
    data_by_trial: Mapping[int, dict[int, Data]],
) -> dict[int, OrderedDict[int, Data]]:
    """
    Load Ax Data from JSON.

    Old `_data_by_trial` is in the format `{trial_index: {timestamp: Data}}`.
    Current data is in the format `{trial_index: {0: Data}}`. This function
    converts `_data_by_trial` from the old format to the current format. Within
    each trial_index, it combines each fetch with the previous one,
    deduplicating in favor of the new data when there are multiple observations
    with the same "trial_index", "metric_name", and "arm_name", and, when
    present, "step."
    """
    combined_data_by_trial = {}
    for trial_index, trial_data in data_by_trial.items():
        if len(trial_data) == 0:
            continue
        sorted_datas = [data for _, data in sorted(trial_data.items())]
        df = sorted_datas.pop().full_df
        while len(sorted_datas) > 0:
            old_df = sorted_datas.pop().full_df
            df = combine_dfs_favoring_recent(last_df=old_df, new_df=df)
        combined_data_by_trial[trial_index] = OrderedDict([(0, Data(df=df))])
    return combined_data_by_trial

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Hash utilities for LILO (Language-in-the-Loop) data freshness tracking."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from ax.core.derived_metric import DerivedMetric
from ax.utils.common.constants import Keys

if TYPE_CHECKING:
    from ax.core.experiment import Experiment


def compute_lilo_input_hash(
    experiment: Experiment,
    input_metric_names: list[str],
) -> str:
    """Compute a hash of the experiment state relevant to LILO labeling.

    The hash captures two components:
    1. The experiment's LLM messages (user preferences that guide labeling).
    2. The observed metric data for ``input_metric_names`` across all trials.

    If any of these inputs change, the hash changes, indicating that existing
    LILO labels are stale and should be excluded from model fitting.

    Args:
        experiment: The experiment whose state to hash.
        input_metric_names: Names of the base metrics whose observed values
            are shown to the LLM for pairwise comparison.

    Returns:
        An SHA-256 hex digest string representing the current LILO input state.
    """
    parts: list[str] = []

    # Component 1: LLM messages (canonical serialization).
    for msg in experiment.llm_messages:
        parts.append(f"{msg.role}:{msg.content}")

    parts.append("---")  # Separator between components.

    # Component 2: Metric data for input_metric_names.
    data = experiment.data
    if not data.empty:
        df = data.df
        metric_df = df[df["metric_name"].isin(input_metric_names)]
        if not metric_df.empty:
            # Sort deterministically and serialize key columns.
            sorted_df = metric_df.sort_values(
                ["trial_index", "arm_name", "metric_name"]
            )
            for _, row in sorted_df.iterrows():
                parts.append(
                    f"{row['trial_index']}|{row['arm_name']}|"
                    f"{row['metric_name']}|{row['mean']}|{row['sem']}"
                )

    content = "\n".join(parts)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def get_current_lilo_hash(experiment: Experiment) -> str | None:
    """Compute the current LILO input hash, or ``None`` if not applicable.

    Looks up the pairwise preference metric on the experiment by name
    (``Keys.PAIRWISE_PREFERENCE_QUERY``), checks that it is a
    ``DerivedMetric`` (which provides ``input_metric_names``), and computes
    the hash.  In practice only ``LILOPairwiseMetric`` satisfies both
    conditions; we check ``DerivedMetric`` rather than ``LILOPairwiseMetric``
    directly because the latter lives in ``ax.fb`` and cannot be imported
    from this OSS module without creating a circular dependency.

    Returns:
        The SHA-256 hex digest of the current LILO input state, or ``None``
        if no suitable pairwise ``DerivedMetric`` is registered.
    """
    pairwise_metric_name = Keys.PAIRWISE_PREFERENCE_QUERY.value
    metric = experiment.metrics.get(pairwise_metric_name)
    # TODO: Replace `DerivedMetric` with `LILOPairwiseMetric` here.
    if metric is None or not isinstance(metric, DerivedMetric):
        return None
    return compute_lilo_input_hash(
        experiment=experiment,
        input_metric_names=metric.input_metric_names,
    )

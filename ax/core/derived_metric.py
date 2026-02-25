#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
DerivedMetric: A metric computed from other metrics.

``DerivedMetric`` is a base class for metrics whose values depend on other
metrics being fetched first.  The experiment's data-fetch loop uses
``isinstance(m, DerivedMetric)`` to guarantee that all base metric data is
attached to the cache before any derived metric's ``fetch_trial_data`` runs.

.. note:: **Transform compatibility.**
   Derived metrics are computed *before* any adapter transforms run.
   Transforms that modify metric values (e.g. ``Relativize``, ``Log``) will
   be applied to the already-computed derived value, **not** to its inputs
   individually.  This means a derived metric ``log(a) - log(b)`` followed
   by a ``Log`` transform would double-log the result.  Avoid using
   transforms that overlap with operations already baked into the derivation.
"""

from __future__ import annotations

from logging import Logger
from typing import Any

import pandas as pd
from ax.core.metric import Metric
from ax.exceptions.core import UserInputError
from ax.utils.common.logger import get_logger


logger: Logger = get_logger(__name__)


class DerivedMetric(Metric):
    """Base class for metrics that depend on other metrics.

    A ``DerivedMetric`` declares the names of metrics whose data must be
    available before this metric can be computed.  The experiment's two-phase
    fetch loop (see ``Experiment._lookup_or_fetch_trials_results``) separates
    derived metrics from base metrics and fetches base metrics first.

    Subclasses must override ``fetch_trial_data`` to define how the derived
    value is produced.

    Attributes:
        input_metric_names: Names of metrics that must be fetched first.
    """

    def __init__(
        self,
        name: str,
        input_metric_names: list[str],
        lower_is_better: bool | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        if not input_metric_names:
            raise UserInputError(
                f"DerivedMetric '{name}' must declare at least one input "
                f"metric in input_metric_names."
            )
        super().__init__(
            name=name,
            lower_is_better=lower_is_better,
            properties=properties,
        )
        self._input_metric_names = input_metric_names

    @property
    def input_metric_names(self) -> list[str]:
        """Names of metrics that this metric depends on."""
        return self._input_metric_names

    @staticmethod
    def _lookup_metric_values_for_arm(
        arm_df: pd.DataFrame,
        metric_name: str,
    ) -> pd.DataFrame:
        """Look up rows for *metric_name* by ``metric_name`` or
        ``metric_signature`` column."""
        return arm_df[
            (arm_df["metric_name"] == metric_name)
            | (arm_df["metric_signature"] == metric_name)
        ]

    @property
    def summary_dict(self) -> dict[str, Any]:
        """Fields of this metric's configuration that will appear
        in the ``Summary`` analysis table.
        """
        return {
            **super().summary_dict,
            "input_metric_names": self._input_metric_names,
        }

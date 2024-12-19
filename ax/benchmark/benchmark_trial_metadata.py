# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from ax.utils.testing.backend_simulator import BackendSimulator


@dataclass(kw_only=True, frozen=True)
class BenchmarkTrialMetadata:
    """
    Data pertaining to one trial evaluation.

    Args:
        df: A dict mapping each metric name to a Pandas DataFrame with columns
            ["metric_name", "arm_name", "mean", "sem", and "step"]. The "sem" is
            always present in this df even if noise levels are unobserved;
            ``BenchmarkMetric`` and ``BenchmarkMapMetric`` hide that data if it
            should not be observed, and ``BenchmarkMapMetric``s drop data from
            time periods that that are not observed based on the (simulated)
            trial progression.
        backend_simulator: Optionally, the backend simulator that is tracking
            the trial's status.
    """

    dfs: Mapping[str, pd.DataFrame]
    backend_simulator: BackendSimulator | None = None

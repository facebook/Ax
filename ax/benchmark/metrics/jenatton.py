# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from ax.benchmark.metrics.base import BenchmarkMetricBase, GroundTruthMetricMixin
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import MetricFetchE, MetricFetchResult
from ax.utils.common.result import Err, Ok
from ax.utils.common.typeutils import not_none


class JenattonMetric(BenchmarkMetricBase):
    """Jenatton metric for hierarchical search spaces."""

    has_ground_truth: bool = True

    def __init__(
        self,
        name: str = "jenatton",
        noise_std: float = 0.0,
        observe_noise_sd: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.noise_std = noise_std
        self.observe_noise_sd = observe_noise_sd
        self.lower_is_better = True

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MetricFetchResult:
        try:
            mean = [
                jenatton_test_function(**arm.parameters)  # pyre-ignore [6]
                for _, arm in trial.arms_by_name.items()
            ]
            if self.noise_std != 0:
                mean = [m + self.noise_std * np.random.randn() for m in mean]
            df = pd.DataFrame(
                {
                    "arm_name": [name for name, _ in trial.arms_by_name.items()],
                    "metric_name": self.name,
                    "mean": mean,
                    "sem": self.noise_std if self.observe_noise_sd else None,
                    "trial_index": trial.index,
                }
            )
            return Ok(value=Data(df=df))

        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )

    def make_ground_truth_metric(self) -> GroundTruthJenattonMetric:
        return GroundTruthJenattonMetric(original_metric=self)


class GroundTruthJenattonMetric(JenattonMetric, GroundTruthMetricMixin):
    def __init__(self, original_metric: JenattonMetric) -> None:
        """
        Args:
            original_metric: The original JenattonMetric to which this metric
                corresponds.
        """
        super().__init__(
            name=self.get_ground_truth_name(original_metric),
            noise_std=0.0,
            observe_noise_sd=False,
        )


def jenatton_test_function(
    x1: Optional[int] = None,
    x2: Optional[int] = None,
    x3: Optional[int] = None,
    x4: Optional[float] = None,
    x5: Optional[float] = None,
    x6: Optional[float] = None,
    x7: Optional[float] = None,
    r8: Optional[float] = None,
    r9: Optional[float] = None,
) -> float:
    """Jenatton test function for hierarchical search spaces.

    This function is taken from:

    R. Jenatton, C. Archambeau, J. González, and M. Seeger. Bayesian
    optimization with tree-structured dependencies. ICML 2017.
    """
    if x1 == 0:
        if x2 == 0:
            return not_none(x4) ** 2 + 0.1 + not_none(r8)
        else:
            return not_none(x5) ** 2 + 0.2 + not_none(r8)
    else:
        if x3 == 0:
            return not_none(x6) ** 2 + 0.3 + not_none(r9)
        else:
            return not_none(x7) ** 2 + 0.4 + not_none(r9)

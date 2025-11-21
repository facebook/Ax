#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.map_data import MAP_KEY, MapData
from ax.core.map_metric import MapMetric, MapMetricFetchResult
from ax.core.metric import MetricFetchE
from ax.utils.common.result import Err, Ok


class NoisyFunctionMapMetric(MapMetric):
    """A metric defined by a generic deterministic function, with normal noise
    with mean 0 and mean_sd scale added to the result.
    """

    def __init__(
        self,
        name: str,
        param_names: Iterable[str],
        noise_sd: float = 0.0,
        lower_is_better: bool | None = None,
    ) -> None:
        """
        Metric is computed by evaluating a deterministic function, implemented
        in f.

        f will expect an array x, which is constructed from the arm
        parameters by extracting the values of the parameter names given in
        param_names, in that order.

        Args:
            name: Name of the metric
            param_names: An ordered list of names of parameters to be passed
                to the deterministic function.
            noise_sd: Scale of normal noise added to the function result.
            lower_is_better: Flag for metrics which should be minimized.
        """
        self.param_names = param_names
        self.noise_sd = noise_sd
        super().__init__(name=name, lower_is_better=lower_is_better)

    @classmethod
    def is_available_while_running(cls) -> bool:
        return True

    @classmethod
    def overwrite_existing_data(cls) -> bool:
        return True

    def clone(self) -> NoisyFunctionMapMetric:
        return self.__class__(
            name=self._name,
            param_names=self.param_names,
            noise_sd=self.noise_sd,
            lower_is_better=self.lower_is_better,
        )

    def fetch_trial_data(
        self, trial: BaseTrial, noisy: bool = True, **kwargs: Any
    ) -> MapMetricFetchResult:
        try:
            res = [
                self.f(np.fromiter(arm.parameters.values(), dtype=float))
                for arm in trial.arms
            ]

            df = pd.DataFrame(
                {
                    "arm_name": [arm.name for arm in trial.arms],
                    "metric_name": self.name,
                    "sem": self.noise_sd if noisy else 0.0,
                    "trial_index": trial.index,
                    "mean": [
                        item["mean"] + self.noise_sd * np.random.randn()
                        if noisy
                        else 0.0
                        for item in res
                    ],
                    "metric_signature": self.signature,
                    MAP_KEY: [item[MAP_KEY] for item in res],
                }
            )
            return Ok(value=MapData(df=df))

        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )

    def f(self, x: npt.NDArray) -> Mapping[str, Any]:
        """The deterministic function that produces the metric outcomes."""
        raise NotImplementedError

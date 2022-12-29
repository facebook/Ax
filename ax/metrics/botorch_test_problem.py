# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.utils.common.result import Err, Ok


class BotorchTestProblemMetric(Metric):
    """A Metric for retriving information from a BotorchTestProblemRunner.
    A BotorchTestProblemRunner will attach the result of a call to
    BaseTestProblem.forward per Arm on a given trial, and this Metric will extract the
    proper value from the resulting tensor given its index.
    """

    def __init__(
        self, name: str, noise_sd: Optional[float] = None, index: Optional[int] = None
    ) -> None:
        super().__init__(name=name)
        self.noise_sd = noise_sd
        self.index = index

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MetricFetchResult:
        try:
            # run_metadata["Ys"] can be either a list of results or a single float
            mean = (
                [
                    trial.run_metadata["Ys"][name][self.index]
                    for name, arm in trial.arms_by_name.items()
                ]
                if self.index is not None
                else [
                    trial.run_metadata["Ys"][name]
                    for name, arm in trial.arms_by_name.items()
                ]
            )
            df = pd.DataFrame(
                {
                    "arm_name": [name for name, _ in trial.arms_by_name.items()],
                    "metric_name": self.name,
                    "mean": mean,
                    # If no noise_std is returned then Botorch evaluated the true
                    # function
                    "sem": self.noise_sd,
                    "trial_index": trial.index,
                }
            )

            return Ok(value=Data(df=df))

        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )

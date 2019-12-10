#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric


class NoisyFunctionMetric(Metric):
    """A metric defined by a generic deterministic function, with normal noise
    with mean 0 and mean_sd scale added to the result.
    """

    def __init__(
        self,
        name: str,
        param_names: List[str],
        noise_sd: float = 0.0,
        lower_is_better: Optional[bool] = None,
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

    def clone(self) -> "NoisyFunctionMetric":
        return self.__class__(
            name=self._name,
            param_names=self.param_names,
            noise_sd=self.noise_sd,
            lower_is_better=self.lower_is_better,
        )

    def fetch_trial_data(
        self, trial: BaseTrial, noisy: bool = True, **kwargs: Any
    ) -> Data:
        noise_sd = self.noise_sd if noisy else 0.0
        arm_names = []
        mean = []
        for name, arm in trial.arms_by_name.items():
            arm_names.append(name)
            x = np.array([arm.parameters[p] for p in self.param_names])
            mean.append(self.f(x) + np.random.randn() * noise_sd)
        df = pd.DataFrame(
            {
                "arm_name": arm_names,
                "metric_name": self.name,
                "mean": mean,
                "sem": noise_sd,
                "trial_index": trial.index,
                "n": 10000 / len(arm_names),
                "frac_nonnull": mean,
            }
        )
        return Data(df=df)

    def f(self, x: np.ndarray) -> float:
        """The deterministic function that produces the metric outcomes."""
        raise NotImplementedError

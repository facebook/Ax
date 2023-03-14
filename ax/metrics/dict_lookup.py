#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.utils.common.result import Err, Ok
from ax.utils.common.typeutils import not_none


class DictLookupMetric(Metric):
    """A metric defined by a dictionary mapping parameter values to the
    corresponding metric values.

    This provides an option to add normal noise with mean 0 and mean_sd scale
    to the given metric values.
    """

    def __init__(
        self,
        name: str,
        param_names: List[str],
        lookup_dict: Dict[Tuple[Union[str, float, int, bool], ...], float],
        noise_sd: Optional[float] = 0.0,
        lower_is_better: Optional[bool] = None,
    ) -> None:
        """Metric is computed via a dictionary look up using a tuple of
        parameter values, constructed based on the ordering of parameter
        names given in `param_names`.

        Args:
            name: Name of the metric.
            param_names: An ordered list of names of parameters to be used
                to construct the dictionary key.
            lookup_dict: A dictionary mapping a tuple of parameter values to
                the metric values.
            noise_sd: Scale of normal noise added to the function result. If
                None, interpret the function as noisy with unknown noise level.
            lower_is_better: Flag for metrics which should be minimized.
        """
        self.param_names = param_names
        self.lookup_dict = lookup_dict
        self.noise_sd = noise_sd
        super().__init__(name=name, lower_is_better=lower_is_better)

    @classmethod
    def is_available_while_running(cls) -> bool:
        return True

    def clone(self) -> DictLookupMetric:
        return self.__class__(
            name=self._name,
            param_names=self.param_names,
            lookup_dict=self.lookup_dict,
            noise_sd=self.noise_sd,
            lower_is_better=self.lower_is_better,
        )

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MetricFetchResult:
        try:
            noise_sd = self.noise_sd
            arm_names = []
            mean = []
            for name, arm in trial.arms_by_name.items():
                arm_names.append(name)
                lookup_key = tuple(
                    not_none(arm.parameters[p]) for p in self.param_names
                )
                try:
                    val = self.lookup_dict[lookup_key]
                except KeyError:
                    raise KeyError(
                        "Got a KeyError while attempting to retrieve the "
                        f"parameterization {arm.parameters} from the lookup dict. "
                        f"This parameterization corresponds to {lookup_key=}."
                    )
                if noise_sd:
                    val = val + noise_sd * np.random.randn()
                mean.append(val)
            # Indicate unknown noise level in data.
            if noise_sd is None:
                noise_sd = float("nan")
            df = pd.DataFrame(
                {
                    "arm_name": arm_names,
                    "metric_name": self.name,
                    "mean": mean,
                    "sem": noise_sd,
                    "trial_index": trial.index,
                }
            )

            return Ok(value=Data(df=df))

        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.types import TParameterization
from ax.utils.common.logger import get_logger

logger = get_logger(__name__)


class NoisyFunctionMapMetric(MapMetric):
    """A metric defined by a generic deterministic function, with normal noise
    with mean 0 and mean_sd scale added to the result.
    """

    def __init__(
        self,
        name: str,
        param_names: List[str],
        noise_sd: float = 0.0,
        lower_is_better: Optional[bool] = None,
        cache_evaluations: bool = True,
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
            cache_evaluations: Flag for whether previous evaluations should
                be cached. If so, those values are returned for previously
                evaluated parameters using the same realization of the
                observation noise.
        """
        self.param_names = param_names
        self.noise_sd = noise_sd
        self.cache = {}
        self.cache_evaluations = cache_evaluations
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
    ) -> MapData:
        noise_sd = self.noise_sd if noisy else 0.0
        arm_names = []
        mean = []
        # assume kwargs = {map_keys: [...], key=list(values) for key in map_keys}
        map_keys = kwargs.get("map_keys", [])
        map_keys_values = defaultdict(list)
        for name, arm in trial.arms_by_name.items():
            map_keys_dict_of_lists = {k: v for k, v in kwargs.items() if k in map_keys}
            map_keys_df = pd.DataFrame.from_dict(
                map_keys_dict_of_lists, orient="index"
            ).transpose()
            for _, row in map_keys_df.iterrows():
                x = self._merge_parameters_and_map_keys(
                    parameters=arm.parameters, map_key_series=row
                )
                # TODO(jej): Use hierarchical DF here for easier syntax?
                arm_names.append(name)
                mean.append(self._cached_f(x, noisy=noisy))
            for map_key, values in map_keys_dict_of_lists.items():
                map_keys_values[map_key].extend(values)
        df = pd.DataFrame(
            {
                "arm_name": arm_names,
                "metric_name": self.name,
                "mean": mean,
                "sem": noise_sd,
                "trial_index": trial.index,
                **map_keys_values,
            }
        )
        return MapData(df=df, map_keys=map_keys)

    def _merge_parameters_and_map_keys(
        self, parameters: TParameterization, map_key_series: pd.Series
    ) -> np.ndarray:
        params_with_overrides = deepcopy(parameters)
        params_with_overrides.update(dict(map_key_series))
        features = [
            params_with_overrides[p]
            for p in params_with_overrides
            if (p in self.param_names) or p in (map_key_series.keys())
        ]
        return np.array(features)

    def _cached_f(self, x: np.ndarray, noisy: bool) -> float:
        noise_sd = self.noise_sd if noisy else 0.0
        x_tuple = tuple(x)  # works since x is 1-d array
        if not self.cache_evaluations:
            return self.f(x) + np.random.randn() * noise_sd
        if x_tuple in self.cache:
            return self.cache[x_tuple]
        new_eval = self.f(x) + np.random.randn() * noise_sd
        self.cache[x_tuple] = new_eval
        return new_eval

    def f(self, x: np.ndarray) -> float:
        """The deterministic function that produces the metric outcomes."""
        raise NotImplementedError

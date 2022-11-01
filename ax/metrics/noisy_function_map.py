#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from logging import Logger

from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.map_data import MapData, MapKeyInfo
from ax.core.map_metric import MapMetric, MapMetricFetchResult
from ax.core.metric import MetricFetchE
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Err, Ok
from ax.utils.common.serialization import serialize_init_args
from ax.utils.common.typeutils import checked_cast

logger: Logger = get_logger(__name__)


class NoisyFunctionMapMetric(MapMetric):
    """A metric defined by a generic deterministic function, with normal noise
    with mean 0 and mean_sd scale added to the result.
    """

    def __init__(
        self,
        name: str,
        param_names: Iterable[str],
        # pyre-fixme[24]: Generic type `MapKeyInfo` expects 1 type parameter.
        map_key_infos: Iterable[MapKeyInfo],
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
        self.map_key_infos = map_key_infos
        self.noise_sd = noise_sd
        # pyre-fixme[4]: Attribute must be annotated.
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
            map_key_infos=self.map_key_infos,
            noise_sd=self.noise_sd,
            lower_is_better=self.lower_is_better,
            cache_evaluations=self.cache_evaluations,
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
                    "mean": [item["mean"] for item in res],
                    **{
                        mki.key: [item[mki.key] for item in res]
                        for mki in self.map_key_infos
                    },
                }
            )

            return Ok(value=MapData(df=df, map_key_infos=self.map_key_infos))

        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )

    def f(self, x: np.ndarray) -> Mapping[str, Any]:
        """The deterministic function that produces the metric outcomes."""
        raise NotImplementedError

    @classmethod
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def serialize_init_args(cls, obj: Any) -> Dict[str, Any]:
        nf_map_metric = checked_cast(NoisyFunctionMapMetric, obj)
        init_args = serialize_init_args(
            object=nf_map_metric, exclude_fields=["map_key_infos"]
        )
        init_args["map_key_infos"] = [
            serialize_init_args(object=mki) for mki in nf_map_metric.map_key_infos
        ]
        return init_args

    @classmethod
    def deserialize_init_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        args["map_key_infos"] = [MapKeyInfo(**mki) for mki in args["map_key_infos"]]
        return super().deserialize_init_args(args=args)

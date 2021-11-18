#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.exceptions.core import UserInputError
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.utils import get_data
from ax.utils.common import constants
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast


logger = get_logger(__name__)


@dataclass
class WinsorizationConfig:
    """Dataclass for storing Winsorization configuration parameters

    Attributes:
    lower_quantile_margin: Winsorization will increase any metric value below this
        quantile to this quantile's value.
    upper_quantile_margin: Winsorization will decrease any metric value above this
        quantile to this quantile's value. NOTE: this quantile will be inverted before
        any operations, e.g., a value of 0.2 will decrease values above the 80th
        percentile to the value of the 80th percentile.
    lower_boundary: If this value is lesser than the metric value corresponding to
        ``lower_quantile_margin``, set metric values below ``lower_boundary`` to
        ``lower_boundary`` and leave larger values unaffected.
    upper_boundary: If this value is greater than the metric value corresponding to
        ``upper_quantile_margin``, set metric values above ``upper_boundary`` to
        ``upper_boundary`` and leave smaller values unaffected.
    """

    lower_quantile_margin: float = 0.0
    upper_quantile_margin: float = 0.0
    lower_boundary: Optional[float] = None
    upper_boundary: Optional[float] = None


DEFAULT_WINSORIZATION_CONFIG_MINIMIZATION = WinsorizationConfig(
    lower_quantile_margin=constants.DEFAULT_WINSORIZATION_LIMITS_MINIMIZATION[0],
    upper_quantile_margin=constants.DEFAULT_WINSORIZATION_LIMITS_MINIMIZATION[1],
)
DEFAULT_WINSORIZATION_CONFIG_MAXIMIZATION = WinsorizationConfig(
    lower_quantile_margin=constants.DEFAULT_WINSORIZATION_LIMITS_MAXIMIZATION[0],
    upper_quantile_margin=constants.DEFAULT_WINSORIZATION_LIMITS_MAXIMIZATION[1],
)


class Winsorize(Transform):
    """Clip the mean values for each metric to lay within the limits provided in
    the config. The config can contain either or both of two keys:
    - ``"winsorization_config"``, corresponding to either a single
        ``WinsorizationConfig``, which, if provided will be used for all metrics; or
        a mapping ``Dict[str, WinsorizationConfig]`` between each metric name and its
        ``WinsorizationConfig``.
    - ``"optimization_config"``, which can be used to determine default winsorization
        settings if ``"winsorization_config"`` does not provide them for a given
        metric.
    For example,
    ``{"winsorization_config": WinsorizationConfig(lower_quantile_margin=0.3)}``
    will specify the same 30% winsorization from below for all metrics, whereas
    ```
    {
        "winsorization_config":
        {
            "metric_1": WinsorizationConfig(lower_quantile_margin=0.2),
            "metric_2": WinsorizationConfig(upper_quantile_margin=0.1),
        }
    }
    ```
    will winsorize 20% from below for metric_1 and 10% from above from metric_2.
    Additional metrics won't be winsorized.

    Additionally, you can pass in winsorization boundaries ``lower_boundary`` and
    ``upper_boundary``that specify a maximum allowable amount of winsorization. This
    can be used to ensure winsorization doesn't move values across the reference point.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        config: Optional[TConfig] = None,
    ) -> None:
        if len(observation_data) == 0:
            raise ValueError("Winsorize transform requires non-empty observation data.")
        metric_values = get_data(observation_data=observation_data)

        self.percentiles = {}
        for metric_name, vals in metric_values.items():
            metric_config = _maybe_get_winsorization_config_from_transform_config(
                transform_config=config,
                metric_name=metric_name,
            )

            if metric_config is None:
                self.percentiles[metric_name] = (-float("inf"), float("inf"))
            else:
                lower = metric_config.lower_quantile_margin
                upper = metric_config.upper_quantile_margin
                bnd_l = metric_config.lower_boundary
                bnd_u = metric_config.upper_boundary

                if lower >= 1 - upper:
                    raise ValueError(  # pragma: no cover
                        f"Lower bound: {lower} was greater than the inverse of the "
                        f"upper bound: {1 - upper} for metric {metric_name}. Decrease "
                        f"one or both of `lower_quantile_margin` and "
                        "`upper_quantile_margin`."
                    )

                pct_l = np.percentile(vals, lower * 100, interpolation="lower")
                pct_u = np.percentile(vals, (1 - upper) * 100, interpolation="higher")
                pct_l = min(pct_l, bnd_l if bnd_l is not None else float("inf"))
                pct_u = max(pct_u, bnd_u if bnd_u is not None else -float("inf"))
                self.percentiles[metric_name] = (pct_l, pct_u)

    def transform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        """Winsorize observation data in place."""
        for obsd in observation_data:
            for idx, metric_name in enumerate(obsd.metric_names):
                if metric_name not in self.percentiles:  # pragma: no cover
                    raise ValueError(f"Cannot winsorize unknown metric {metric_name}")
                # Clip on the winsorization bounds.
                obsd.means[idx] = max(obsd.means[idx], self.percentiles[metric_name][0])
                obsd.means[idx] = min(obsd.means[idx], self.percentiles[metric_name][1])
        return observation_data


def _maybe_get_winsorization_config_from_transform_config(
    metric_name: str,
    transform_config: Optional[TConfig] = None,
) -> Optional[WinsorizationConfig]:
    # Don't winsorize if `transform_config is None`.
    if transform_config is None:
        return None
    # Return `winsorization_config` if specified.
    if "winsorization_config" in transform_config:
        wconfig = transform_config["winsorization_config"]
        # If `winsorization_config` is a single `WinsorizationConfig`, use
        # it for all metrics.
        if isinstance(wconfig, WinsorizationConfig):
            return wconfig
        # If `winsorization_config` is a dict, use if metric_name is in keys,
        # and the corresponding value is a WinsorizationConfig.
        if isinstance(wconfig, dict) and metric_name in wconfig:
            metric_config = wconfig[metric_name]
            if not isinstance(metric_config, WinsorizationConfig):
                raise UserInputError(
                    "Expected winsorization config of type "
                    f"`WinsorizationConfig` but got {metric_config} of type "
                    f"{type(metric_config)} for metric {metric_name}."
                )
            return metric_config
    # If a WinsorizationConfig has not been specified for `metric_name` and
    # optimization_config is specified, use it to determine defaults.
    if "optimization_config" in transform_config:
        oconfig = transform_config["optimization_config"]
        if not isinstance(oconfig, OptimizationConfig):
            raise UserInputError(
                "Expected `optimization_config` of type `OptimizationConfig` but "
                f"got type `{type(oconfig)}."
            )
            return None
        oconfig = checked_cast(OptimizationConfig, oconfig)
        if oconfig.is_moo_problem:
            warnings.warn(
                "Winsorization defaults are currently not available for "
                f"multi-objective optimization problems. Not winsorizing {metric_name}."
            )
            return None
        return (
            DEFAULT_WINSORIZATION_CONFIG_MINIMIZATION
            if oconfig.objective.minimize
            else DEFAULT_WINSORIZATION_CONFIG_MAXIMIZATION
        )
    old_keys = ["winsorization_lower", "winsorization_upper", "percentile_bounds"]
    if any(old_key in transform_config for old_key in old_keys):
        DeprecationWarning(
            "Winsorization received an out-of-date `transform_config`, containing at "
            f"least one of the keys {old_keys}. Please update the config according to "
            "the docs of `ax.modelbridge.transforms.winsorize.Winsorize`."
        )
        return _get_winsorization_config_from_legacy_transform_config(
            metric_name=metric_name,
            transform_config=transform_config,
        )
    # If none of the above, don't winsorize.
    return None


def _get_winsorization_config_from_legacy_transform_config(
    metric_name: str,
    transform_config: TConfig,
) -> WinsorizationConfig:
    winsorization_config = WinsorizationConfig()
    if "winsorization_lower" in transform_config:
        winsorization_lower = transform_config["winsorization_lower"]
        if isinstance(winsorization_lower, dict):
            if metric_name in winsorization_lower:
                winsorization_config.lower_quantile_margin = winsorization_lower[
                    metric_name
                ]
        elif isinstance(winsorization_lower, (int, float)):
            winsorization_config.lower_quantile_margin = winsorization_lower
    if "winsorization_upper" in transform_config:
        winsorization_upper = transform_config["winsorization_upper"]
        if isinstance(winsorization_upper, dict):
            if metric_name in winsorization_upper:
                winsorization_config.upper_quantile_margin = winsorization_upper[
                    metric_name
                ]
        elif isinstance(winsorization_upper, (int, float)):
            winsorization_config.upper_quantile_margin = winsorization_upper
    if "percentile_bounds" in transform_config:
        percentile_bounds = transform_config["percentile_bounds"]
        output_percentile_bounds = (None, None)
        if isinstance(percentile_bounds, dict):
            if metric_name in percentile_bounds:
                output_percentile_bounds = percentile_bounds[metric_name]
        elif isinstance(percentile_bounds, tuple):
            output_percentile_bounds = percentile_bounds
        if len(output_percentile_bounds) != 2 or not all(
            isinstance(pb, (int, float)) or pb is None
            for pb in output_percentile_bounds
        ):
            raise ValueError(
                f"Expected percentile_bounds for metric {metric_name} to be "
                f"of the form (l, u), got {output_percentile_bounds}."
            )
        winsorization_config.lower_boundary = output_percentile_bounds[0]
        winsorization_config.upper_boundary = output_percentile_bounds[1]
    return winsorization_config

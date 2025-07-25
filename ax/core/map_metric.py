#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from ax.core.map_data import MapData, MapKeyInfo
from ax.core.metric import Metric, MetricFetchE
from ax.utils.common.result import Result

MapMetricFetchResult = Result[MapData, MetricFetchE]
DEFAULT_MAP_KEY = "step"


class MapMetric(Metric):
    """Base class for representing metrics that return `MapData`.

    The `fetch_trial_data` method is the essential method to override when
    subclassing, which specifies how to retrieve a Metric, for a given trial.

    A MapMetric must return a MapData object, which requires (at minimum) the following:
        https://ax.dev/api/_modules/ax/core/data.html#Data.required_columns

    Attributes:
        lower_is_better: Flag for metrics which should be minimized.
        properties: Properties specific to a particular metric.
    """

    data_constructor: type[MapData] = MapData
    map_key_info: MapKeyInfo[float] = MapKeyInfo(key=DEFAULT_MAP_KEY, default_value=0.0)

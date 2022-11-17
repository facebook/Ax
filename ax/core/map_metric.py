#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Type

from ax.core.map_data import MapData
from ax.core.metric import Metric, MetricFetchE
from ax.utils.common.result import Result

MapMetricFetchResult = Result[MapData, MetricFetchE]


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

    data_constructor: Type[MapData] = MapData

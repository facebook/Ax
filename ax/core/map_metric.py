#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict, Type

from ax.core.data import Data

from ax.core.map_data import MapData
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.utils.common.result import Ok, Result
from ax.utils.common.typeutils import checked_cast

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

    @classmethod
    def _wrap_experiment_data(cls, data: Data) -> Dict[int, MetricFetchResult]:
        return {
            trial_index: Ok(
                value=MapData(
                    df=data.true_df.loc[data.true_df["trial_index"] == trial_index],
                    map_key_infos=checked_cast(MapData, data).map_key_infos,
                )
            )
            for trial_index in data.true_df["trial_index"]
        }

    @classmethod
    def _wrap_trial_data_multi(cls, data: Data) -> Dict[str, MetricFetchResult]:
        return {
            metric_name: Ok(
                value=MapData(
                    df=data.true_df.loc[data.true_df["metric_name"] == metric_name],
                    map_key_infos=checked_cast(MapData, data).map_key_infos,
                )
            )
            for metric_name in data.true_df["metric_name"]
        }

    @classmethod
    def _wrap_experiment_data_multi(
        cls, data: Data
    ) -> Dict[int, Dict[str, MetricFetchResult]]:
        # pyre-fixme[7]
        return {
            trial_index: {
                metric_name: Ok(
                    value=MapData(
                        df=data.true_df.loc[
                            (data.true_df["trial_index"] == trial_index)
                            & (data.true_df["metric_name"] == metric_name)
                        ],
                        map_key_infos=checked_cast(MapData, data).map_key_infos,
                    )
                )
                for metric_name in data.true_df["metric_name"]
            }
            for trial_index in data.true_df["trial_index"]
        }

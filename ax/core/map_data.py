# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Sequence
from logging import Logger

import pandas as pd

# backward compatibility
from ax.core.data import Data, MAP_KEY  # noqa F401
from ax.utils.common.logger import get_logger

logger: Logger = get_logger(__name__)


class MapData(Data):
    """Class storing mapping-like results for an experiment.

    Data is stored at the trial_index-arm_name-metric_name-step level in a
    dataframe ``full_df``. A smaller DataFrame ``df`` may be constructed at the
    trial_index-arm_name-metric_name level, using only the most recent step for
    each trial/arm/metric.

    The data can be stored to an external store for future use by attaching it
    to an experiment using `experiment.attach_data()`.


    Attributes:
        full_df: DataFrame with underlying data. The required columns
            are "arm_name", "metric_name", "mean", "sem", and "step", the latter
            three of which must be numeric. This is close to the raw data input by the
            user as ``df``; by contrast, the property ``self.df`` is be a subset
            of the full data used for modeling. Constructing ``df`` can be
            expensive, so it is better to reference ``full_df`` than ``df`` for
            operations that do not require scanning the full data, such as
            accessing the columns of the DataFrame.
        _memo_df: Either ``None``, if ``self.df`` has never been accessed, or
            equivalent to ``self.df``.

    Properties:
        df: Potentially smaller representation of the data used for modeling,
            containing only the most recent ``step`` values
            for each trial-arm-metric. Because constructing ``df`` can be
            expensive, it is recommended to reference ``full_df`` for operations
            that do not require scanning the full data, such as accessing the
            columns of the DataFrame.
    """

    pass


def combine_datas_infer_type(data_list: Sequence[Data]) -> Data:
    """
    Combine muiltiple datas into one.

    If any of the datas is MapData, return MapData. Otherwise, return Data.
    Empty data is a MapData.
    """
    non_empty_datas = [d for d in data_list if not d.full_df.empty]
    if len(non_empty_datas) == 0:
        return MapData()

    has_map_data = any(isinstance(d, MapData) for d in non_empty_datas)
    df = pd.concat([d.full_df for d in non_empty_datas], axis=0, sort=has_map_data)
    if has_map_data:
        return MapData(df=df)
    return Data(df=df)


def data_from_df_infer_type(df: pd.DataFrame) -> Data:
    """
    Create a Data object from a DataFrame.

    If the DataFrame has a MAP_KEY column, return MapData. Otherwise, return Data.
    """
    if MAP_KEY in df.columns:
        return MapData(df=df)
    return Data(df=df)

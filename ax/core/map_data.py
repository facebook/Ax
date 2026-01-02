# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Sequence
from typing import Any

import pandas as pd

# backward compatibility
from ax.core.data import Data, MAP_KEY  # noqa F401
from ax.exceptions.core import AxError


class MapData:
    """MapData no longer exists. Use Data instead."""

    def __init__(self, *_: Any, **__: Any) -> None:
        raise AxError("MapData no longer exists. Use Data instead.")


def combine_datas_infer_type(data_list: Sequence[Data]) -> Data:
    """Combine muiltiple datas into one."""
    return Data.from_multiple_data(data=data_list)


def data_from_df_infer_type(df: pd.DataFrame) -> Data:
    """Create a Data object from a DataFrame."""
    return Data(df=df)

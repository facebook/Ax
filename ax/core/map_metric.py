#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Type

from ax.core.map_data import MapData
from ax.core.metric import Metric


if TYPE_CHECKING:  # pragma: no cover
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


class MapMetric(Metric):
    """Base class for representing metrics that return `MapData`.

    The `fetch_trial_data` method is the essential method to override when
    subclassing, which specifies how to retrieve a Metric, for a given trial.

    A MapMetric must return a MapData object, which requires (at minimum) the following:
        https://ax.dev/api/_modules/ax/core/abstract_data.html#AbstractDataFrameData.required_columns

    Attributes:
        lower_is_better: Flag for metrics which should be minimized.
        properties: Properties specific to a particular metric.
    """

    # pyre-fixme[15]: Inconsistent override of `Type[Data]` with `Type[MapData]`
    data_constructor: Type[MapData] = MapData

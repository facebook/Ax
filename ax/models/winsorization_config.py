#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional


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

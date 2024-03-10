#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from numbers import Real
from typing import List, Tuple

# type aliases
TRGB = Tuple[Real, ...]


def rgba(rgb_tuple: TRGB, alpha: float = 1) -> str:
    """Convert RGB tuple to an RGBA string."""
    return "rgba({},{},{},{alpha})".format(*rgb_tuple, alpha=alpha)


def plotly_color_scale(
    list_of_rgb_tuples: List[TRGB],
    reverse: bool = False,
    alpha: float = 1,
) -> List[Tuple[float, str]]:
    """Convert list of RGB tuples to list of tuples, where each tuple is
    break in [0, 1] and stringified RGBA color.
    """
    if reverse:
        list_of_rgb_tuples = list_of_rgb_tuples[::-1]
    return [
        (round(i / (len(list_of_rgb_tuples) - 1), 3), rgba(rgb))
        for i, rgb in enumerate(list_of_rgb_tuples)
    ]

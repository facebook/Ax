#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
from typing import List, Tuple


class COLORS(enum.Enum):
    STEELBLUE = (128, 177, 211)
    CORAL = (251, 128, 114)
    TEAL = (141, 211, 199)
    PINK = (188, 128, 189)
    LIGHT_PURPLE = (190, 186, 218)
    ORANGE = (253, 180, 98)


# colors to be used for plotting discrete series
DISCRETE_COLOR_SCALE = [
    COLORS.STEELBLUE.value,
    COLORS.CORAL.value,
    COLORS.TEAL.value,
    COLORS.PINK.value,
    COLORS.LIGHT_PURPLE.value,
    COLORS.ORANGE.value,
]

# 11-class PiYG from ColorBrewer (for contour plots)
GREEN_PINK_SCALE = [
    (142, 1, 82),
    (197, 27, 125),
    (222, 119, 174),
    (241, 182, 218),
    (253, 224, 239),
    (247, 247, 247),
    (230, 245, 208),
    (184, 225, 134),
    (127, 188, 65),
    (77, 146, 33),
    (39, 100, 25),
]
GREEN_SCALE = [
    (247, 252, 253),
    (229, 245, 249),
    (204, 236, 230),
    (153, 216, 201),
    (102, 194, 164),
    (65, 174, 118),
    (35, 139, 69),
    (0, 109, 44),
    (0, 68, 27),
]
BLUE_SCALE = [
    (255, 247, 251),
    (236, 231, 242),
    (208, 209, 230),
    (166, 189, 219),
    (116, 169, 207),
    (54, 144, 192),
    (5, 112, 176),
    (3, 78, 123),
]
# 24 Class Mixed Color Palette
# Source: https://graphicdesign.stackexchange.com/a/3815
MIXED_SCALE = [
    (2, 63, 165),
    (125, 135, 185),
    (190, 193, 212),
    (214, 188, 192),
    (187, 119, 132),
    (142, 6, 59),
    (74, 111, 227),
    (133, 149, 225),
    (181, 187, 227),
    (230, 175, 185),
    (224, 123, 145),
    (211, 63, 106),
    (17, 198, 56),
    (141, 213, 147),
    (198, 222, 199),
    (234, 211, 198),
    (240, 185, 141),
    (239, 151, 8),
    (15, 207, 192),
    (156, 222, 214),
    (213, 234, 231),
    (243, 225, 235),
    (246, 196, 225),
    (247, 156, 212),
]


def rgba(rgb_tuple: Tuple[float], alpha: float = 1) -> str:
    """Convert RGB tuple to an RGBA string."""
    return "rgba({},{},{},{alpha})".format(*rgb_tuple, alpha=alpha)


def plotly_color_scale(
    list_of_rgb_tuples: List[Tuple[float]], reverse: bool = False, alpha: float = 1
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

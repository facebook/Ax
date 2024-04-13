# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import enum

# Constants used for numerous plots
CI_OPACITY = 0.4
DECIMALS = 3
Z = 1.96


# color constants used for plotting
class COLORS(enum.Enum):
    STEELBLUE = (128, 177, 211)
    CORAL = (251, 128, 114)
    TEAL = (141, 211, 199)
    PINK = (188, 128, 189)
    LIGHT_PURPLE = (190, 186, 218)
    ORANGE = (253, 180, 98)

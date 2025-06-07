# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import plotly.express as px
from ax.analysis.plotly.utils import (
    AX_BLUE,
    CONSTRAINT_VIOLATION_RED,
    LIGHT_AX_BLUE,
    METRIC_CONTINUOUS_COLOR_SCALE,
)

DISCRETE_ARM_SCALE = px.colors.qualitative.Alphabet
CONSTRAINT_VIOLATION_COLOR = CONSTRAINT_VIOLATION_RED
CANDIDATE_AX_COLOR = LIGHT_AX_BLUE
AX_BLUE

# Use the same continuous sequential color scale for all plots. PRGn uses purples for
# low values and transitions to greens for high values.
POSITIVE_CHANGE_COLOR = METRIC_CONTINUOUS_COLOR_SCALE[8]  # lighter green
NEGATIVE_CHANGE_COLOR = METRIC_CONTINUOUS_COLOR_SCALE[2]  # lighter purple
METRIC_CONTINUOUS_COLOR_SCALE

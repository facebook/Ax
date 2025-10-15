# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-safe

import plotly.express as px

# Colors sampled from Botorch flame logo
BOTORCH_COLOR_SCALE = [
    "#f7931e",  # Botorch orange
    "#eb882d",
    "#df7d3c",
    "#d3724b",
    "#c7685a",
    "#bb5d69",
    "#af5278",
    "#a34887",
    "#973d96",
    "#8b32a5",
    "#7f28b5",  # Botorch purple
    "#792fbb",
    "#7436c1",
    "#6f3dc7",
    "#6a44cd",
    "#654bd4",
    "#6052da",
    "#5b59e0",
    "#5660e6",
    "#5167ec",
    "#4c6ef3",  # Botorch blue
]
AX_BLUE = "#5078f9"  # rgb code: rgb(80, 120, 249)
LIGHT_AX_BLUE = "#adc0fd"  # rgb(173, 192, 253)


# Use the same continuous sequential color scale for all plots. PRGn uses purples for
# low values and transitions to greens for high values.
METRIC_CONTINUOUS_COLOR_SCALE: list[str] = px.colors.diverging.Earth
COLOR_FOR_INCREASES: str = METRIC_CONTINUOUS_COLOR_SCALE[5]  # blue
COLOR_FOR_DECREASES: str = METRIC_CONTINUOUS_COLOR_SCALE[2]  # brown

DISCRETE_ARM_SCALE = px.colors.qualitative.Alphabet

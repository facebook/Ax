#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from logging import Logger
from typing import Dict, List, Optional, Tuple, Union

from ax.analysis.helpers.constants import DECIMALS, Z

from ax.core.generator_run import GeneratorRun

from ax.core.types import TParameterization
from ax.utils.common.logger import get_logger

from plotly import graph_objs as go


logger: Logger = get_logger(__name__)

# Typing alias
RawData = List[Dict[str, Union[str, float]]]

TNullableGeneratorRunsDict = Optional[Dict[str, GeneratorRun]]


def _format_dict(param_dict: TParameterization, name: str = "Parameterization") -> str:
    """Format a dictionary for labels.

    Args:
        param_dict: Dictionary to be formatted
        name: String name of the thing being formatted.

    Returns: stringified blob.
    """
    if len(param_dict) >= 10:
        blob = "{} has too many items to render on hover ({}).".format(
            name, len(param_dict)
        )
    else:
        blob = "<br><em>{}:</em><br>{}".format(
            name, "<br>".join("{}: {}".format(n, v) for n, v in param_dict.items())
        )
    return blob


def _format_CI(estimate: float, sd: float, zval: float = Z) -> str:
    """Format confidence intervals given estimate and standard deviation.

    Args:
        estimate: point estimate.
        sd: standard deviation of point estimate.
        zval: z-value associated with desired CI (e.g. 1.96 for 95% CIs)

    Returns: formatted confidence interval.
    """
    return "[{lb:.{digits}f}, {ub:.{digits}f}]".format(
        lb=estimate - zval * sd,
        ub=estimate + zval * sd,
        digits=DECIMALS,
    )


def resize_subtitles(figure: go.Figure, size: int) -> go.Figure:
    """Resize subtitles in a plotly figure
    args:
        figure: plotly figure to resize subtitles of
        size: font size to resize subtitles to
    """
    for ant in figure["layout"]["annotations"]:
        ant["font"].update(size=size)
    return figure


def arm_name_to_sort_key(arm_name: str) -> Tuple[str, int, int]:
    """Parses arm name into tuple suitable for reverse sorting by key

    Example:
        arm_names = ["0_0", "1_10", "1_2", "10_0", "control"]
        sorted(arm_names, key=arm_name_to_sort_key, reverse=True)
        ["control", "0_0", "1_2", "1_10", "10_0"]
    """
    try:
        trial_index, arm_index = arm_name.split("_")
        return ("", -int(trial_index), -int(arm_index))
    except (ValueError, IndexError):
        return (arm_name, 0, 0)

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any

from ax.core.optimization_config import OptimizationConfig
from ax.generators.winsorization_config import WinsorizationConfig
from botorch.acquisition import AcquisitionFunction

TConfig = dict[
    str,
    int
    | float
    | str
    | AcquisitionFunction
    | list[int]
    | list[str]
    | dict[int, Any]
    | dict[str, Any]
    | OptimizationConfig
    | WinsorizationConfig
    | None,
]

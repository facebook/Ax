#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Union

from ax.core.optimization_config import OptimizationConfig
from ax.models.winsorization_config import WinsorizationConfig
from botorch.acquisition import AcquisitionFunction

TConfig = Dict[
    str,
    Union[
        int,
        float,
        str,
        AcquisitionFunction,
        Dict[str, Any],
        OptimizationConfig,
        WinsorizationConfig,
        None,
    ],
]

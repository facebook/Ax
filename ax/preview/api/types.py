# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Mapping

TParameterValue = int | float | str | bool
TParameterization = Mapping[str, TParameterValue]

# Metric name => mean | (mean, sem)
TOutcome = Mapping[str, float | tuple[float, float]]

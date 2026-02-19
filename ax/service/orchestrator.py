#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Backward compatibility shim.

This module has been moved to ax.orchestration.orchestrator.
Please update imports to use: from ax.orchestration.orchestrator import Orchestrator
"""

import warnings

from ax.orchestration.orchestrator import (
    FailureRateExceededError,
    get_fitted_adapter,
    MessageOutput,
    OptimizationResult,
    Orchestrator,
    OrchestratorInternalError,
    OutputPriority,
    StatusQuoInfeasibleError,
)
from ax.orchestration.orchestrator_options import OrchestratorOptions

warnings.warn(
    "ax.service.orchestrator has been moved to ax.orchestration.orchestrator. "
    "Please update imports to use: "
    "from ax.orchestration.orchestrator import Orchestrator. "
    "This backward compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "FailureRateExceededError",
    "get_fitted_adapter",
    "MessageOutput",
    "OptimizationResult",
    "Orchestrator",
    "OrchestratorInternalError",
    "OrchestratorOptions",
    "OutputPriority",
    "StatusQuoInfeasibleError",
]

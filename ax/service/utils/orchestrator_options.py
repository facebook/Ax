# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Backward compatibility shim.

This module has been moved to ax.orchestration.orchestrator_options.
Please update imports to use:
    from ax.orchestration.orchestrator_options import OrchestratorOptions, TrialType
"""

import warnings

from ax.orchestration.orchestrator_options import OrchestratorOptions, TrialType

warnings.warn(
    "ax.service.utils.orchestrator_options has been moved to "
    "ax.orchestration.orchestrator_options. "
    "Please update imports to use: "
    "from ax.orchestration.orchestrator_options import OrchestratorOptions, TrialType. "
    "This backward compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["OrchestratorOptions", "TrialType"]

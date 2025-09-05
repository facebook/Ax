# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Scheduler was deprecated in Ax 1.1.0, so it should be removed in Ax
# Ax 1.2.0+.


class SchedulerOptions:
    raise DeprecationWarning(
        "Scheduler is deprecated following renaming of the Scheduler to "
        "Orchestrator. Please use Orchestrator and OrchestratorOptions instead; "
        "import with: `from ax.service.orchestrator import Orchestrator, "
        "OrchestratorOptions`"
    )

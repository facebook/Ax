# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


class Scheduler:
    raise DeprecationWarning(
        "Scheduler is deprecated following renaming of the Scheduler to "
        "Orchestrator. Please use Orchestrator instead; import with: "
        "`from ax.orchestration.orchestrator import Orchestrator`"
    )


class SchedulerInternalError(Exception):
    raise DeprecationWarning(
        "SchedulerInternalError is deprecated following renaming of the Scheduler to "
        "Orchestrator. Please use OrchestratorInternalError instead; import with: "
        "`from ax.orchestration.orchestrator import OrchestratorInternalError`"
    )

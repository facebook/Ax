# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


class SchedulerOptions:
    raise DeprecationWarning(
        "SchedulerOptions is deprecated following renaming of the Scheduler to "
        "Orchestrator. Please use OrchestratorOptions instead; import with: "
        "`from ax.orchestration.orchestrator_options import OrchestratorOptions`"
    )

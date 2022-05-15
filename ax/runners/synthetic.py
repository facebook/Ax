#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterable, Optional, Set

from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.runner import Runner


class SyntheticRunner(Runner):
    """Class for synthetic or dummy runner.

    Currently acts as a shell runner, only creating a name.
    """

    def __init__(self, dummy_metadata: Optional[str] = None) -> None:
        self.dummy_metadata = dummy_metadata

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        deployed_name = (
            trial.experiment.name + "_" + str(trial.index)
            if trial.experiment.has_name
            else str(trial.index)
        )
        metadata = {"name": deployed_name}

        # Add dummy metadata if needed for testing
        if self.dummy_metadata:
            metadata["dummy_metadata"] = self.dummy_metadata
        return metadata

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        return {TrialStatus.COMPLETED: {t.index for t in trials}}

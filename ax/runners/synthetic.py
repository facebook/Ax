#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Dict, Optional

from ax.core.base_trial import BaseTrial
from ax.core.runner import Runner


class SyntheticRunner(Runner):
    """Class for synthetic or dummy runner.

    Currently acts as a shell runner, only creating a name.
    """

    def __init__(self, dummy_metadata: Optional[str] = None):
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

#!/usr/bin/env python3

from typing import Any, Dict, Optional

from ae.lazarus.ae.core.base_trial import BaseTrial
from ae.lazarus.ae.core.runner import Runner
from ae.lazarus.ae.utils.common.equality import equality_typechecker


class SyntheticRunner(Runner):
    """Class for synthetic or dummy runner.

    Currently acts as a shell runner, only creating a name.
    """

    def __init__(self, dummy_metadata: Optional[str] = None):
        self.dummy_metadata = dummy_metadata

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        metadata = {"name": trial.experiment.name + "_" + str(trial.index)}

        # Add dummy metadata if needed for testing
        if self.dummy_metadata:
            metadata["dummy_metadata"] = self.dummy_metadata
        return metadata

    def staging_required(self) -> bool:
        return True

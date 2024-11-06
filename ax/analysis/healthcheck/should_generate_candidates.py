# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json

import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel

from ax.analysis.healthcheck.healthcheck_analysis import (
    HealthcheckAnalysis,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface


class ShouldGenerateCandidates(HealthcheckAnalysis):
    def __init__(
        self,
        should_generate: bool,
        reason: str,
        trial_index: int,
    ) -> None:
        self.should_generate = should_generate
        self.reason = reason
        self.trial_index = trial_index

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> HealthcheckAnalysisCard:
        status = (
            HealthcheckStatus.PASS
            if self.should_generate
            else HealthcheckStatus.WARNING
        )
        return HealthcheckAnalysisCard(
            name=self.name,
            title=f"Ready to Generate Candidates for Trial {self.trial_index}",
            blob=json.dumps(
                {
                    "status": status,
                }
            ),
            subtitle=self.reason,
            df=pd.DataFrame(
                {
                    "status": [status],
                    "reason": [self.reason],
                }
            ),
            level=AnalysisCardLevel.CRITICAL,
            attributes=self.attributes,
        )

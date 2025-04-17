# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Sequence

import pandas as pd
from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel

from ax.analysis.healthcheck.healthcheck_analysis import (
    HealthcheckAnalysis,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from pyre_extensions import override


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

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[HealthcheckAnalysisCard]:
        status = (
            HealthcheckStatus.PASS
            if self.should_generate
            else HealthcheckStatus.WARNING
        )
        return [
            self._create_healthcheck_analysis_card(
                title=f"Ready to Generate Candidates for Trial {self.trial_index}",
                subtitle=self.reason,
                df=pd.DataFrame(),
                level=AnalysisCardLevel.CRITICAL,
                status=status,
                category=AnalysisCardCategory.DIAGNOSTIC,
            ),
        ]

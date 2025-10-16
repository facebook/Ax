# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.healthcheck.healthcheck_analysis import (
    create_healthcheck_analysis_card,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import override


@final
class ShouldGenerateCandidates(Analysis):
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
    ) -> HealthcheckAnalysisCard:
        status = (
            HealthcheckStatus.PASS
            if self.should_generate
            else HealthcheckStatus.WARNING
        )
        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title=f"Ready to Generate Candidates for Trial {self.trial_index}",
            subtitle=self.reason,
            df=pd.DataFrame(),
            status=status,
        )

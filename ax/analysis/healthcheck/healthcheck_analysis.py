# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import json
from enum import IntEnum

from ax.analysis.analysis import Analysis, AnalysisCard
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface


class HealthcheckStatus(IntEnum):
    PASS = 0
    FAIL = 1
    WARNING = 2


class HealthcheckAnalysisCard(AnalysisCard):
    blob_annotation = "healthcheck"

    def get_status(self) -> HealthcheckStatus:
        return HealthcheckStatus(json.loads(self.blob)["status"])


class HealthcheckAnalysis(Analysis):
    """
    An analysis that performs a health check.
    """

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> HealthcheckAnalysisCard: ...

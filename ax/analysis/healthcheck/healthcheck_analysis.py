# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import json
import traceback
from enum import IntEnum
from typing import Sequence

import pandas as pd

from ax.analysis.analysis import (
    Analysis,
    AnalysisBlobAnnotation,
    AnalysisCard,
    AnalysisCardCategory,
    AnalysisCardLevel,
    AnalysisE,
)
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from pyre_extensions import override


class HealthcheckStatus(IntEnum):
    PASS = 0
    FAIL = 1
    WARNING = 2


class HealthcheckAnalysisCard(AnalysisCard):
    blob_annotation: AnalysisBlobAnnotation = AnalysisBlobAnnotation.HEALTHCHECK

    def get_status(self) -> HealthcheckStatus:
        return HealthcheckStatus(json.loads(self.blob)["status"])


class HealthcheckAnalysisE(AnalysisE):
    def error_card(self) -> list[AnalysisCard]:
        exception_stack_trace = "".join(
            traceback.format_exception(
                type(self.exception),
                self.exception,
                self.exception.__traceback__,
            )
        )
        return [
            HealthcheckAnalysisCard(
                name=self.analysis.name,
                title=f"{self.analysis.name} Error",
                subtitle=(
                    f"An error occurred while computing {self.analysis}:\n"
                    f"{exception_stack_trace}"
                ),
                attributes=self.analysis.attributes,
                blob=json.dumps({"status": HealthcheckStatus.FAIL}),
                df=pd.DataFrame(),
                level=AnalysisCardLevel.DEBUG,
                category=AnalysisCardCategory.ERROR,
            )
        ]


class HealthcheckAnalysis(Analysis):
    """
    An analysis that performs a health check.
    """

    """
    For HealthcheckAnalysis, generate a HealthcheckAnalysisCard with FAIL status.
    """
    exception_class: type[AnalysisE] = HealthcheckAnalysisE

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[HealthcheckAnalysisCard]: ...

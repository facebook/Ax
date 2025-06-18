# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from enum import IntEnum

import pandas as pd
from ax.adapter.base import Adapter

from ax.analysis.analysis import Analysis
from ax.analysis.analysis_card import AnalysisCard, AnalysisCardBase
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import override


class HealthcheckStatus(IntEnum):
    PASS = 0
    FAIL = 1
    WARNING = 2


class HealthcheckAnalysisCard(AnalysisCard):
    def get_status(self) -> HealthcheckStatus:
        return HealthcheckStatus(json.loads(self.blob)["status"])

    def is_passing(self) -> bool:
        return self.get_status() == HealthcheckStatus.PASS

    def get_aditional_attrs(self) -> dict[str, str | int | float | bool]:
        return json.loads(self.blob)


class HealthcheckAnalysis(Analysis):
    """
    An analysis that performs a health check.
    """

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardBase: ...

    def _create_healthcheck_analysis_card(
        self,
        title: str,
        subtitle: str,
        df: pd.DataFrame,
        status: HealthcheckStatus,
        **additional_attrs: str | int | float | bool,
    ) -> HealthcheckAnalysisCard:
        return HealthcheckAnalysisCard(
            name=self.__class__.__name__,
            title=title,
            subtitle=subtitle,
            df=df,
            blob=json.dumps(
                {
                    "status": status,
                    **additional_attrs,
                }
            ),
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.analysis.analysis import ErrorAnalysisCard
from ax.analysis.healthcheck.healthcheck_analysis import (
    create_healthcheck_analysis_card,
    HealthcheckStatus,
    sort_healthcheck_cards,
)
from ax.core.analysis_card import AnalysisCardBase
from ax.utils.common.testutils import TestCase


def _card(name: str, status: HealthcheckStatus) -> AnalysisCardBase:
    return create_healthcheck_analysis_card(
        name=name, title=name, subtitle=name, df=pd.DataFrame(), status=status
    )


def _error(name: str) -> AnalysisCardBase:
    return ErrorAnalysisCard(
        name=name, title=name, subtitle=name, df=pd.DataFrame(), blob=""
    )


class TestHealthcheckAnalysis(TestCase):
    def test_sort_ordering(self) -> None:
        cards: list[AnalysisCardBase] = [
            _card("RegularAnalysis", HealthcheckStatus.PASS),
            _card("WarningAnalysis", HealthcheckStatus.WARNING),
            _error("ErrorAnalysis"),
            _card("BaselineImprovementAnalysis", HealthcheckStatus.PASS),
            _card("FailAnalysis", HealthcheckStatus.FAIL),
        ]
        result = sort_healthcheck_cards(cards)

        self.assertEqual(
            [c.name for c in result],
            [
                "ErrorAnalysis",
                "FailAnalysis",
                "WarningAnalysis",
                "BaselineImprovementAnalysis",
                "RegularAnalysis",
            ],
        )

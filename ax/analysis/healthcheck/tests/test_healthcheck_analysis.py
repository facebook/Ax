# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.analysis.healthcheck.healthcheck_analysis import (
    create_healthcheck_analysis_card,
    HealthcheckStatus,
)
from ax.utils.common.testutils import TestCase


class TestHealthcheckAnalysisCard(TestCase):
    def test_is_user_facing(self) -> None:
        # Only PASS status should be hidden; all others are user-facing
        for status in HealthcheckStatus:
            with self.subTest(status=status.name):
                card = create_healthcheck_analysis_card(
                    name="TestAnalysis",
                    title="Test Healthcheck",
                    subtitle="Test subtitle",
                    df=pd.DataFrame(),
                    status=status,
                )
                expected = status != HealthcheckStatus.PASS
                self.assertEqual(card.is_user_facing(), expected)

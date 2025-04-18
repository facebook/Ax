# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import cast, Sequence

from ax.analysis.analysis import AnalysisBlobAnnotation
from ax.analysis.healthcheck.healthcheck_analysis import (
    HealthcheckAnalysis,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from ax.utils.common.testutils import TestCase

ERROR_MESSAGE = "Dummy analysis failed!"


class TestHealtheckException(TestCase):
    class DummyAnalysis(HealthcheckAnalysis):
        def compute(
            self,
            experiment: Experiment | None = None,
            generation_strategy: GenerationStrategy | None = None,
            adapter: Adapter | None = None,
        ) -> Sequence[HealthcheckAnalysisCard]:
            raise ValueError(ERROR_MESSAGE)

    def test_error_analysis_card_on_exception(self) -> None:
        analysis = self.DummyAnalysis()
        with self.assertLogs("ax.analysis.analysis", "ERROR") as logs:
            analysis_cards = analysis.compute_result().unwrap_or_else(
                lambda e: e.error_card()
            )
        # Check that the error message is logged
        self.assertEqual(len(logs.output), 1)
        self.assertIn(ERROR_MESSAGE, logs.output[0])

        # Check that a failed healtheck analysis card is produced
        self.assertEqual(len(analysis_cards), 1)
        analysis_card = analysis_cards[0]
        self.assertEqual(
            analysis_card.blob_annotation, AnalysisBlobAnnotation.HEALTHCHECK
        )
        self.assertEqual(type(analysis_card), HealthcheckAnalysisCard)
        self.assertEqual(
            (cast(HealthcheckAnalysisCard, analysis_card)).get_status(),
            HealthcheckStatus.FAIL,
        )

        # Check that error message is in the analysis card
        self.assertEqual(analysis_card.name, "DummyAnalysis")
        self.assertEqual(analysis_card.title, "DummyAnalysis Failure")
        self.assertIn(ERROR_MESSAGE, analysis_card.subtitle)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckAnalysisCard
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.utils.common.testutils import TestCase

ERROR_MESSAGE = "Dummy analysis failed!"


class TestHealtheckException(TestCase):
    class DummyAnalysis(Analysis):
        def compute(
            self,
            experiment: Experiment | None = None,
            generation_strategy: GenerationStrategy | None = None,
            adapter: Adapter | None = None,
        ) -> HealthcheckAnalysisCard:
            raise ValueError(ERROR_MESSAGE)

    def test_error_analysis_card_on_exception(self) -> None:
        analysis = self.DummyAnalysis()
        with self.assertLogs("ax.analysis.analysis", "ERROR") as logs:
            analysis_cards = analysis.compute_or_error_card().flatten()
        # Check that the error message is logged
        self.assertEqual(len(logs.output), 2)
        self.assertIn(ERROR_MESSAGE, logs.output[1])

        # Check that a failed healtheck analysis card is produced
        self.assertEqual(len(analysis_cards), 1)
        analysis_card = analysis_cards[0]

        # Check that error message is in the analysis card
        self.assertEqual(analysis_card.name, "DummyAnalysis")
        self.assertEqual(analysis_card.title, "DummyAnalysis Error")

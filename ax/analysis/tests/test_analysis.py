# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.analysis import AnalysisE, error_card_from_analysis_e
from ax.analysis.plotly.parallel_coordinates import ParallelCoordinatesPlot
from ax.utils.common.testutils import TestCase


class AnalysisTest(TestCase):
    def test_error_card_from_analysis_e(self) -> None:
        for exception, expected_subtitle in (
            (
                ValueError("something went wrong"),
                "ValueError: something went wrong",
            ),
            (
                ValueError(),
                "ValueError encountered while computing ParallelCoordinatesPlot.",
            ),
        ):
            with self.subTest(exception=exception):
                analysis_e = AnalysisE(
                    message="test",
                    exception=exception,
                    analysis=ParallelCoordinatesPlot(),
                )

                card = error_card_from_analysis_e(analysis_e)

                self.assertEqual(card.name, "ParallelCoordinatesPlot")
                self.assertEqual(card.title, "ParallelCoordinatesPlot Error")
                self.assertEqual(card.subtitle, expected_subtitle)
                self.assertIn("ValueError", card.blob)

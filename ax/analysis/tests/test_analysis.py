#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.analysis import (
    AnalysisE,
    error_card_from_analysis_e,
    NOT_APPLICABLE_STATE_SUBTITLE,
)
from ax.analysis.plotly.parallel_coordinates import ParallelCoordinatesPlot
from ax.core.analysis_card import ErrorAnalysisCard, NotApplicableStateAnalysisCard
from ax.exceptions.analysis import AnalysisNotApplicableStateError
from ax.utils.common.testutils import TestCase


class AnalysisTest(TestCase):
    def test_error_card_from_analysis_e(self) -> None:
        for (
            exception,
            expected_card_type,
            expected_title,
            expected_subtitle,
            expected_blob,
        ) in (
            (
                ValueError("something went wrong"),
                ErrorAnalysisCard,
                "ParallelCoordinatesPlot Error",
                "ValueError: something went wrong",
                "ValueError",
            ),
            (
                ValueError(),
                ErrorAnalysisCard,
                "ParallelCoordinatesPlot Error",
                "ValueError encountered while computing ParallelCoordinatesPlot.",
                "ValueError",
            ),
            (
                AnalysisNotApplicableStateError("Experiment has no data."),
                NotApplicableStateAnalysisCard,
                "ParallelCoordinatesPlot -- Not Available Yet",
                NOT_APPLICABLE_STATE_SUBTITLE,
                "Experiment has no data.",
            ),
        ):
            with self.subTest(exception=exception):
                analysis_e = AnalysisE(
                    message="test",
                    exception=exception,
                    analysis=ParallelCoordinatesPlot(),
                )

                card = error_card_from_analysis_e(analysis_e)

                self.assertIsInstance(card, expected_card_type)
                self.assertEqual(card.name, "ParallelCoordinatesPlot")
                self.assertEqual(card.title, expected_title)
                self.assertEqual(card.subtitle, expected_subtitle)
                self.assertIn(expected_blob, card.blob)

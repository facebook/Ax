# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel
from ax.analysis.plotly.surface.contour import ContourPlot
from ax.exceptions.core import UserInputError
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize


class TestContourPlot(TestCase):
    @mock_botorch_optimize
    def setUp(self) -> None:
        super().setUp()
        self.client = AxClient()
        self.client.create_experiment(
            is_test=True,
            name="foo",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [-1.0, 1.0],
                },
                {
                    "name": "y",
                    "type": "range",
                    "bounds": [-1.0, 1.0],
                },
            ],
            objectives={"bar": ObjectiveProperties(minimize=True)},
        )

        for _ in range(10):
            parameterization, trial_index = self.client.get_next_trial()
            self.client.complete_trial(
                trial_index=trial_index,
                raw_data={
                    "bar": parameterization["x"] ** 2 + parameterization["y"] ** 2
                },
            )

    def test_compute(self) -> None:
        analysis = ContourPlot(
            x_parameter_name="x", y_parameter_name="y", metric_name="bar"
        )

        # Test that it fails if no Experiment is provided
        with self.assertRaisesRegex(UserInputError, "requires an Experiment"):
            analysis.compute()
        # Test that it fails if no GenerationStrategy is provided
        with self.assertRaisesRegex(UserInputError, "requires a GenerationStrategy"):
            analysis.compute(experiment=self.client.experiment)

        card = analysis.compute(
            experiment=self.client.experiment,
            generation_strategy=self.client.generation_strategy,
        )
        self.assertEqual(
            card.name,
            "ContourPlot",
        )
        self.assertEqual(card.title, "x, y vs. bar")
        self.assertEqual(
            card.subtitle,
            "2D contour of the surrogate model's predicted outcomes for bar",
        )
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        self.assertEqual(card.category, AnalysisCardCategory.INSIGHT)
        self.assertEqual(
            {*card.df.columns},
            {
                "x",
                "y",
                "bar_mean",
            },
        )
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, "plotly")

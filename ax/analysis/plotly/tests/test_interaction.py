# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.plotly.interaction import InteractionPlot
from ax.exceptions.core import UserInputError
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.common.testutils import TestCase

from ax.utils.testing.mock import mock_botorch_optimize


class TestInteractionPlot(TestCase):
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

        for _ in range(2):
            parameterization, trial_index = self.client.get_next_trial()
            self.client.complete_trial(
                trial_index=trial_index,
                raw_data={
                    "bar": parameterization["x"] ** 2 + parameterization["y"] ** 2
                },
            )

    @TestCase.ax_long_test(
        reason="This test requires fitting an OAK model, which can be time intensive"
    )
    @mock_botorch_optimize
    def test_compute(self) -> None:
        analysis = InteractionPlot(metric_name="bar")

        # Test that it fails if no Experiment is provided
        with self.assertRaisesRegex(UserInputError, "requires an Experiment"):
            analysis.compute()

        card = analysis.compute(
            experiment=self.client.experiment,
            generation_strategy=self.client.generation_strategy,
        )
        self.assertEqual(
            card.name,
            "InteractionPlot",
        )
        self.assertEqual(card.title, "Interaction Analysis for bar")
        self.assertEqual(
            card.subtitle,
            "Understand an Experiment's data as one- or two-dimensional additive "
            "components with sparsity. Important components are visualized through "
            "slice or contour plots",
        )
        self.assertEqual(card.level, AnalysisCardLevel.MID)
        self.assertEqual(
            {*card.df.columns},
            {"feature", "sensitivity"},
        )
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, "plotly")

        fig = card.get_figure()

        # Ensure there is at least one of each type of plot in the figure (we cannot
        # check for the exact number of subplots because we added by trace and there
        # may be many traces per subplot).
        trace_names = [trace.__class__.__name__ for trace in fig.data]
        self.assertIn("Bar", trace_names)
        self.assertTrue(
            any("Scatter" in name for name in trace_names)
            or any("Contour" in name for name in trace_names)
        )

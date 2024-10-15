# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.plotly.cross_validation import CrossValidationPlot
from ax.exceptions.core import UserInputError
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import fast_botorch_optimize


class TestCrossValidationPlot(TestCase):
    @fast_botorch_optimize
    def test_compute(self) -> None:
        client = AxClient()
        client.create_experiment(
            is_test=True,
            name="foo",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [-1.0, 1.0],
                }
            ],
            objectives={"bar": ObjectiveProperties(minimize=True)},
        )

        for _ in range(10):
            parameterization, trial_index = client.get_next_trial()
            client.complete_trial(
                trial_index=trial_index, raw_data={"bar": parameterization["x"] ** 2}
            )

        analysis = CrossValidationPlot(metric_name="bar")

        # Test that it fails if no GenerationStrategy is provided
        with self.assertRaisesRegex(UserInputError, "requires a GenerationStrategy"):
            analysis.compute()

        card = analysis.compute(generation_strategy=client.generation_strategy)
        self.assertEqual(
            card.name,
            "CrossValidationPlot",
        )
        self.assertEqual(card.title, "Cross Validation for bar")
        self.assertEqual(
            card.subtitle,
            "Out-of-sample predictions using leave-one-out CV",
        )
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        self.assertEqual(
            {*card.df.columns},
            {"arm_name", "observed", "observed_sem", "predicted", "predicted_sem"},
        )
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, "plotly")

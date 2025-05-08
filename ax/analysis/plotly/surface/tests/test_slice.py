# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.analysis import (
    AnalysisBlobAnnotation,
    AnalysisCardCategory,
    AnalysisCardLevel,
)
from ax.analysis.plotly.surface.slice import compute_slice_adhoc, SlicePlot
from ax.core.trial import Trial
from ax.exceptions.core import UserInputError
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize

from pyre_extensions import assert_is_instance, none_throws


class TestSlicePlot(TestCase):
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
                }
            ],
            objectives={"bar": ObjectiveProperties(minimize=True)},
        )

        for _ in range(10):
            parameterization, trial_index = self.client.get_next_trial()
            self.client.complete_trial(
                trial_index=trial_index, raw_data={"bar": parameterization["x"] ** 2}
            )

    def test_compute(self) -> None:
        analysis = SlicePlot(parameter_name="x", metric_name="bar")

        # Test that it fails if no Experiment is provided
        with self.assertRaisesRegex(UserInputError, "requires an Experiment"):
            analysis.compute()
        # Test that it fails if no GenerationStrategy is provided
        with self.assertRaisesRegex(
            UserInputError, "Must provide either a GenerationStrategy or an Adapter"
        ):
            analysis.compute(experiment=self.client.experiment)

        (card,) = analysis.compute(
            experiment=self.client.experiment,
            generation_strategy=self.client.generation_strategy,
        )
        self.assertEqual(
            card.name,
            "SlicePlot",
        )
        self.assertEqual(card.title, "x vs. bar")
        self.assertEqual(
            card.subtitle,
            (
                "The slice plot provides a one-dimensional view of predicted "
                "outcomes for bar as a function of a single parameter, "
                "while keeping all other parameters fixed at their status_quo "
                "value (or mean value if status_quo is unavailable). "
                "This visualization helps in understanding the sensitivity and "
                "impact of changes in the selected parameter on the predicted "
                "metric outcomes."
            ),
        )
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        self.assertEqual(card.category, AnalysisCardCategory.INSIGHT)
        self.assertEqual(
            {*card.df.columns},
            {"x", "bar_mean", "bar_sem", "sampled"},
        )
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, AnalysisBlobAnnotation.PLOTLY)

        # Assert that any row where sampled is True has a value of x that is
        # sampled in at least one trial.
        x_values_sampled = {
            none_throws(assert_is_instance(trial, Trial).arm).parameters["x"]
            for trial in self.client.experiment.trials.values()
        }
        self.assertTrue(
            card.df.apply(
                lambda row: row["x"] in x_values_sampled if row["sampled"] else True,
                axis=1,
            ).all()
        )

    def test_compute_adhoc(self) -> None:
        (card,) = compute_slice_adhoc(
            parameter_name="x",
            metric_name="bar",
            experiment=self.client.experiment,
            generation_strategy=self.client.generation_strategy,
        )
        self.assertEqual(
            card.name,
            "SlicePlot",
        )
        self.assertEqual(card.title, "x vs. bar")
        self.assertEqual(
            card.subtitle,
            (
                "The slice plot provides a one-dimensional view of predicted "
                "outcomes for bar as a function of a single parameter, "
                "while keeping all other parameters fixed at their status_quo "
                "value (or mean value if status_quo is unavailable). "
                "This visualization helps in understanding the sensitivity and "
                "impact of changes in the selected parameter on the predicted "
                "metric outcomes."
            ),
        )
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        self.assertEqual(card.category, AnalysisCardCategory.INSIGHT)
        self.assertEqual(
            {*card.df.columns},
            {"x", "bar_mean", "bar_sem", "sampled"},
        )
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, AnalysisBlobAnnotation.PLOTLY)

        # Assert that any row where sampled is True has a value of x that is
        # sampled in at least one trial.
        x_values_sampled = {
            none_throws(assert_is_instance(trial, Trial).arm).parameters["x"]
            for trial in self.client.experiment.trials.values()
        }
        self.assertTrue(
            card.df.apply(
                lambda row: row["x"] in x_values_sampled if row["sampled"] else True,
                axis=1,
            ).all()
        )

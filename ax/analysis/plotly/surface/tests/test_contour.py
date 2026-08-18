# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
from ax.analysis.plotly.surface.contour import compute_contour_adhoc, ContourPlot
from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.core.trial import Trial
from ax.exceptions.core import UserInputError
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize
from plotly import graph_objects as go
from pyre_extensions import assert_is_instance, none_throws


class TestContourPlot(TestCase):
    @mock_botorch_optimize
    def setUp(self) -> None:
        super().setUp()

        # There were some flaky test failures on the github side. Fix the random seed
        # to reduce the flakiness.
        self.client = AxClient(random_seed=42)
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
                {
                    "name": "z",
                    "type": "choice",
                    "values": [1, 2, 3, 4],
                    "value_type": "int",
                    "is_ordered": True,
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
        self.expected_subtitle_contains = [
            "The contour plot visualizes the predicted outcomes "
            "for bar across a two-dimensional parameter space, "
            "with other parameters held fixed at their best trial value",
        ]
        self.expected_title = "bar (Mean) vs. x, y"
        self.expected_name = "ContourPlot"

    def _sampled_points_from_card(
        self, fig: go.Figure
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Extract the (x, y) coordinates of the "Sampled" scatter trace."""
        scatters = [trace for trace in fig.data if isinstance(trace, go.Scatter)]
        self.assertEqual(len(scatters), 1)
        return tuple(scatters[0].x), tuple(scatters[0].y)

    def test_compute(self) -> None:
        analysis = ContourPlot(
            x_parameter_name="x", y_parameter_name="y", metric_name="bar"
        )

        # Test that it fails if no Experiment is provided
        with self.assertRaisesRegex(UserInputError, "requires an Experiment"):
            analysis.compute()
        # Test that it fails if no GenerationStrategy is provided
        with self.assertRaisesRegex(
            UserInputError, "Must provide either a GenerationStrategy or an Adapter"
        ):
            analysis.compute(experiment=self.client.experiment)

        card = analysis.compute(
            experiment=self.client.experiment,
            generation_strategy=self.client.generation_strategy,
        )
        self.assertEqual(
            card.name,
            self.expected_name,
        )
        self.assertEqual(card.title, self.expected_title)
        for expected_text in self.expected_subtitle_contains:
            self.assertIn(expected_text, card.subtitle)
        self.assertIsNotNone(card.blob)
        # The prediction grid lives in the Plotly blob, not the DataFrame.
        self.assertTrue(card.df.empty)

        # Assert that every point marked as sampled in the figure has x and y values
        # that were sampled in at least one trial.
        x_values_sampled = {
            none_throws(assert_is_instance(trial, Trial).arm).parameters["x"]
            for trial in self.client.experiment.trials.values()
        }
        y_values_sampled = {
            none_throws(assert_is_instance(trial, Trial).arm).parameters["y"]
            for trial in self.client.experiment.trials.values()
        }
        xs, ys = self._sampled_points_from_card(fig=card.get_figure())
        for x, y in zip(xs, ys, strict=True):
            self.assertIn(x, x_values_sampled)
            self.assertIn(y, y_values_sampled)

        # Less-than-or-equal to because we may have removed some duplicates
        self.assertLessEqual(len(xs), len(self.client.experiment.trials))

    def test_compute_adhoc(self) -> None:
        card = compute_contour_adhoc(
            x_parameter_name="x",
            y_parameter_name="y",
            metric_name="bar",
            experiment=self.client.experiment,
            generation_strategy=self.client.generation_strategy,
        )
        self.assertEqual(
            card.name,
            self.expected_name,
        )
        self.assertEqual(card.title, self.expected_title)
        for expected_text in self.expected_subtitle_contains:
            self.assertIn(expected_text, card.subtitle)
        self.assertIsNotNone(card.blob)
        self.assertTrue(card.df.empty)

        # Assert that every point marked as sampled in the figure has x and y values
        # that were sampled in at least one trial.
        x_values_sampled = {
            none_throws(assert_is_instance(trial, Trial).arm).parameters["x"]
            for trial in self.client.experiment.trials.values()
        }
        y_values_sampled = {
            none_throws(assert_is_instance(trial, Trial).arm).parameters["y"]
            for trial in self.client.experiment.trials.values()
        }
        xs, ys = self._sampled_points_from_card(fig=card.get_figure())
        for x, y in zip(xs, ys, strict=True):
            self.assertIn(x, x_values_sampled)
            self.assertIn(y, y_values_sampled)

        # Less-than-or-equal to because we may have removed some duplicates
        self.assertLessEqual(len(xs), len(self.client.experiment.trials))

    def test_trial_status_filtering(self) -> None:
        # Add an abandoned trial whose arm sits at a parameterization no completed
        # trial uses, so its exclusion is observable in the figure's sampled points.
        abandoned_trial = self.client.experiment.new_trial(
            generator_run=GeneratorRun(
                arms=[Arm(parameters={"x": 0.5, "y": 0.5, "z": 1})]
            )
        )
        abandoned_trial.mark_abandoned()

        analysis = ContourPlot(
            x_parameter_name="x", y_parameter_name="y", metric_name="bar"
        )
        card = analysis.compute(
            experiment=self.client.experiment,
            generation_strategy=self.client.generation_strategy,
        )
        xs, ys = self._sampled_points_from_card(fig=card.get_figure())
        self.assertNotIn((0.5, 0.5), set(zip(xs, ys, strict=True)))

    def test_display_sem(self) -> None:
        """Test that display='sem' shows standard error contour."""
        analysis = ContourPlot(
            x_parameter_name="x",
            y_parameter_name="y",
            metric_name="bar",
            display="sem",
        )
        card = analysis.compute(
            experiment=self.client.experiment,
            generation_strategy=self.client.generation_strategy,
        )

        # Title should indicate Standard Error
        self.assertEqual(card.title, "bar (Standard Error) vs. x, y")
        self.assertEqual(card.name, "ContourPlot")

        # The contoured surface should be the sems, not the means. SEMs are
        # non-negative, and the two surfaces should differ.
        sem_z = np.asarray(card.get_figure().data[0].z, dtype=float)
        self.assertTrue((sem_z >= 0).all())

        mean_card = ContourPlot(
            x_parameter_name="x",
            y_parameter_name="y",
            metric_name="bar",
            display="mean",
        ).compute(
            experiment=self.client.experiment,
            generation_strategy=self.client.generation_strategy,
        )
        mean_z = np.asarray(mean_card.get_figure().data[0].z, dtype=float)
        self.assertEqual(sem_z.shape, mean_z.shape)
        self.assertFalse(np.allclose(sem_z, mean_z))

    def test_invalid_display_value(self) -> None:
        """Test that invalid display value raises UserInputError at compute time."""
        analysis = ContourPlot(
            x_parameter_name="x",
            y_parameter_name="y",
            metric_name="bar",
            display="invalid",
        )
        with self.assertRaisesRegex(UserInputError, "display must be 'mean' or 'sem'"):
            analysis.compute(
                experiment=self.client.experiment,
                generation_strategy=self.client.generation_strategy,
            )

    def test_compute_with_choice_parameter(self) -> None:
        """Test contour plot with ordered ChoiceParameter on one axis."""
        analysis = ContourPlot(
            x_parameter_name="x", y_parameter_name="z", metric_name="bar"
        )
        card = analysis.compute(
            experiment=self.client.experiment,
            generation_strategy=self.client.generation_strategy,
        )

        # Assert: Verify the contour plot was created successfully
        self.assertEqual(card.name, "ContourPlot")
        self.assertEqual(card.title, "bar (Mean) vs. x, z")

        # Assert: the y-axis (the choice parameter) only contains its discrete values
        contour = card.get_figure().data[0]
        self.assertEqual(contour.y, (1, 2, 3, 4))
        self.assertGreater(len(contour.x), 0)

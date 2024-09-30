# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.plotly.scatter import _prepare_data, ScatterPlot
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment_with_multi_objective,
    get_experiment_with_observations,
)


class TestParallelCoordinatesPlot(TestCase):
    def test_compute(self) -> None:
        analysis = ScatterPlot(
            x_metric_name="branin_a",
            y_metric_name="branin_b",
            show_pareto_frontier=True,
        )
        experiment = get_branin_experiment_with_multi_objective(
            with_completed_trial=True
        )

        with self.assertRaisesRegex(UserInputError, "requires an Experiment"):
            analysis.compute()

        card = analysis.compute(experiment=experiment)
        self.assertEqual(
            card.name,
            (
                "ScatterPlot(x_metric_name=branin_a, y_metric_name=branin_b, "
                "show_pareto_frontier=True)"
            ),
        )
        self.assertEqual(card.title, "Observed branin_a vs. branin_b")
        self.assertEqual(
            card.subtitle,
            "Compare arms by their observed metric values",
        )
        self.assertEqual(card.level, AnalysisCardLevel.HIGH)
        self.assertEqual(
            {*card.df.columns},
            {"arm_name", "trial_index", "branin_a", "branin_b", "is_optimal"},
        )
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, "plotly")

    def test_prepare_data(self) -> None:
        observations = [[float(i), float(i + 1)] for i in range(10)]
        experiment = get_experiment_with_observations(
            observations=observations,
        )

        data = _prepare_data(
            experiment=experiment, x_metric_name="m1", y_metric_name="m2"
        )

        # Ensure that the data is in the correct shape
        self.assertEqual(len(data), len(observations))
        self.assertEqual(
            {*data.columns},
            {
                "trial_index",
                "arm_name",
                "m1",
                "m2",
                "is_optimal",
            },
        )

        # Check data is correct
        for i in range(len(observations)):
            row = data.iloc[i]
            self.assertEqual(row["trial_index"], i)
            self.assertEqual(row["arm_name"], f"{i}_0")
            self.assertEqual(row["m1"], observations[i][0])
            self.assertEqual(row["m2"], observations[i][1])

            # Ensure that the optimal point is labeled correctly
            if i == len(observations) - 1:
                self.assertTrue(row["is_optimal"])
            else:
                self.assertFalse(row["is_optimal"])

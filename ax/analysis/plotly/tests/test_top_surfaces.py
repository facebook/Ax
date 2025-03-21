# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.plotly.top_surfaces import TopSurfacesAnalysis
from ax.api.client import Client
from ax.api.configs import ExperimentConfig, ParameterType, RangeParameterConfig
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize
from pyre_extensions import assert_is_instance


class TestTopSurfacesAnalysis(TestCase):
    @mock_botorch_optimize
    def test_compute(self) -> None:
        client = Client()
        client.configure_experiment(
            experiment_config=ExperimentConfig(
                name="foo",
                parameters=[
                    RangeParameterConfig(
                        name="x1",
                        parameter_type=ParameterType.FLOAT,
                        bounds=(0, 1),
                    ),
                    RangeParameterConfig(
                        name="x2",
                        parameter_type=ParameterType.FLOAT,
                        bounds=(0, 1),
                    ),
                ],
            )
        )
        client.configure_optimization(objective="bar")

        for _ in range(6):
            for trial_index, parameterization in client.get_next_trials().items():
                client.complete_trial(
                    trial_index=trial_index,
                    raw_data={
                        "bar": assert_is_instance(parameterization["x1"], float)
                        - 2 * assert_is_instance(parameterization["x2"], float)
                    },
                )

        analysis = TopSurfacesAnalysis(metric_name="bar", order="first")

        with self.assertRaisesRegex(
            UserInputError, "requires either a TorchAdapter or a GenerationStrategy"
        ):
            analysis.compute()

        cards = analysis.compute(
            experiment=client._experiment,
            generation_strategy=client._generation_strategy,
        )

        self.assertEqual(len(cards), 3)
        for card in cards:
            self.assertEqual(
                card.name,
                "TopSurfacesAnalysis",
            )

        # First card should be the sensitivity analysis.
        self.assertEqual(cards[0].title, "Sensitivity Analysis for bar")

        # Other cards should be slices.
        self.assertIn("vs. bar", cards[1].title)
        self.assertIn("vs. bar", cards[2].title)

        second = TopSurfacesAnalysis(metric_name="bar", order="second")

        with_contours = second.compute(
            experiment=client._experiment,
            generation_strategy=client._generation_strategy,
        )

        self.assertEqual(len(with_contours), 4)
        for card in with_contours:
            self.assertEqual(
                card.name,
                "TopSurfacesAnalysis",
            )

        # First card should be the sensitivity analysis.
        self.assertEqual(with_contours[0].title, "Sensitivity Analysis for bar")

        # Other cards should be slices or contours.
        self.assertIn("vs. bar", with_contours[1].title)
        self.assertIn("vs. bar", with_contours[2].title)
        self.assertIn("vs. bar", with_contours[3].title)

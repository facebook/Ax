# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel
from ax.analysis.plotly.sensitivity import SensitivityAnalysisPlot
from ax.api.client import Client
from ax.api.configs import ExperimentConfig, ParameterType, RangeParameterConfig
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize
from pyre_extensions import assert_is_instance


class TestSensitivityAnalysisPlot(TestCase):
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

        analysis = SensitivityAnalysisPlot(metric_names=["bar"], order="first")

        with self.assertRaisesRegex(
            UserInputError, "Must provide either a GenerationStrategy or an Adapter"
        ):
            analysis.compute()

        (card,) = analysis.compute(generation_strategy=client._generation_strategy)
        self.assertEqual(
            card.name,
            "SensitivityAnalysisPlot",
        )
        self.assertEqual(card.title, "Sensitivity Analysis for bar")
        self.assertEqual(
            card.subtitle,
            "Understand how each parameter affects bar according to a first-order "
            "sensitivity analysis.",
        )
        self.assertEqual(card.level, AnalysisCardLevel.MID)
        self.assertEqual(card.category, AnalysisCardCategory.INSIGHT)
        self.assertEqual(
            {*card.df.columns},
            {"parameter_name", "sensitivity"},
        )
        self.assertEqual(len(card.df), 2)
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, "plotly")

        second_order = SensitivityAnalysisPlot(metric_names=["bar"], order="second")
        (card,) = second_order.compute(generation_strategy=client._generation_strategy)
        self.assertEqual(len(card.df), 3)  # 2 first order + 1 second order

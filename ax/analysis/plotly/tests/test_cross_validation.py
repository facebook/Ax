# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel
from ax.analysis.plotly.cross_validation import CrossValidationPlot
from ax.core.trial import Trial
from ax.exceptions.core import UserInputError
from ax.modelbridge.registry import Generators
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize
from pyre_extensions import assert_is_instance, none_throws


class TestCrossValidationPlot(TestCase):
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
        analysis = CrossValidationPlot(metric_name="bar")

        # Test that it fails if no GenerationStrategy is provided
        with self.assertRaisesRegex(UserInputError, "requires a GenerationStrategy"):
            analysis.compute()

        card = analysis.compute(generation_strategy=self.client.generation_strategy)
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
        self.assertEqual(card.category, AnalysisCardCategory.INSIGHT)
        self.assertEqual(
            {*card.df.columns},
            {"arm_name", "observed", "observed_95_ci", "predicted", "predicted_95_ci"},
        )
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, "plotly")
        # Assert that all arms are in the cross validation df
        # because trial index is not specified
        for t in self.client.experiment.trials.values():
            # Skip the last trial because the model was used to generate it
            # and therefore hasn't observed it
            if t.index == max(self.client.experiment.trials.keys()):
                continue
            arm_name = none_throws(assert_is_instance(t, Trial).arm).name
            self.assertIn(
                arm_name,
                card.df["arm_name"].unique(),
            )

    def test_it_can_only_contain_observation_prior_to_the_trial_index(self) -> None:
        analysis = CrossValidationPlot(metric_name="bar", trial_index=7)
        with self.assertRaisesRegex(
            UserInputError,
            "CrossValidationPlot was specified to be for the generation of trial 7",
        ):
            analysis.compute(generation_strategy=self.client.generation_strategy)

    def test_it_can_specify_trial_index_correctly(self) -> None:
        analysis = CrossValidationPlot(metric_name="bar", trial_index=9)
        card = analysis.compute(generation_strategy=self.client.generation_strategy)
        for t in self.client.experiment.trials.values():
            # Skip the last trial because the model was used to generate it
            # and therefore hasn't observed it
            if t.index == max(self.client.experiment.trials.keys()):
                continue
            arm_name = none_throws(assert_is_instance(t, Trial).arm).name
            self.assertIn(
                arm_name,
                card.df["arm_name"].unique(),
            )

    @mock_botorch_optimize
    def test_compute_adhoc(self) -> None:
        metric_mapping = {"bar": "spunky"}
        data = self.client.experiment.lookup_data()
        adapter = Generators.BOTORCH_MODULAR(
            experiment=self.client.experiment, data=data
        )
        analysis = CrossValidationPlot()._compute_adhoc(
            adapter=adapter, data=data, metric_name_mapping=metric_mapping
        )
        self.assertEqual(len(analysis), 1)
        card = analysis[0]
        self.assertEqual(card.name, "CrossValidationPlot")
        # validate that the metric name replacement occured
        self.assertEqual(card.title, "Cross Validation for spunky")

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
from ax.analysis.plotly.cross_validation import (
    cross_validation_adhoc_compute,
    CrossValidationPlot,
)
from ax.core.trial import Trial
from ax.exceptions.core import UserInputError
from ax.modelbridge.registry import Generators
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_offline_experiments, get_online_experiments
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_default_generation_strategy_at_MBM_node
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
        with self.assertRaisesRegex(
            UserInputError, "Must provide either a GenerationStrategy or an Adapter"
        ):
            analysis.compute()

        (card,) = analysis.compute(generation_strategy=self.client.generation_strategy)
        self.assertEqual(
            card.name,
            "CrossValidationPlot",
        )
        self.assertEqual(card.title, "Cross Validation for bar")
        self.assertEqual(
            card.subtitle,
            (
                "The cross-validation plot displays the model fit for each "
                "metric in the experiment. It employs a leave-one-out "
                "approach, where the model is trained on all data except one "
                "sample, which is used for validation. The plot shows the "
                "predicted outcome for the validation set on the y-axis against "
                "its actual value on the x-axis. Points that align closely with "
                "the dotted diagonal line indicate a strong model fit, signifying "
                "accurate predictions. Additionally, the plot includes 95% "
                "confidence intervals that provide insight into the noise in "
                "observations and the uncertainty in model predictions. A "
                "horizontal, flat line of predictions indicates that the model "
                "has not picked up on sufficient signal in the data, and instead "
                "is just predicting the mean."
            ),
        )
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        self.assertEqual(card.category, AnalysisCardCategory.INSIGHT)
        self.assertEqual(
            {*card.df.columns},
            {"arm_name", "observed", "observed_95_ci", "predicted", "predicted_95_ci"},
        )
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, AnalysisBlobAnnotation.PLOTLY)
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

    def test_raises_if_no_metric_name_and_no_exp(self) -> None:
        analysis = CrossValidationPlot()
        with self.subTest("raises if no metric name and no experiment"):
            with self.assertRaisesRegex(
                UserInputError, "attempting to infer metric name"
            ):
                analysis.compute(generation_strategy=self.client.generation_strategy)
        with self.subTest("infer from experiment"):
            (card,) = analysis.compute(
                generation_strategy=self.client.generation_strategy,
                experiment=self.client.experiment,
            )
            # validates that metric name was successfully inferred from exp
            self.assertEqual(card.title, "Cross Validation for bar")

    def test_it_can_specify_trial_index_correctly(self) -> None:
        analysis = CrossValidationPlot(metric_name="bar", trial_index=9)
        (card,) = analysis.compute(generation_strategy=self.client.generation_strategy)
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
        analysis = cross_validation_adhoc_compute(
            adapter=adapter, data=data, metric_name_mapping=metric_mapping
        )
        self.assertEqual(len(analysis), 1)
        card = analysis[0]
        self.assertEqual(card.name, "CrossValidationPlot")
        # validate that the metric name replacement occurred
        self.assertEqual(card.title, "Cross Validation for spunky")

    @mock_botorch_optimize
    def test_online(self) -> None:
        # Test CrossValidationPlot can be computed for a variety of experiments which
        # resemble those we see in an online setting.

        for experiment in get_online_experiments():
            for untransform in [True, False]:
                for refined_metric_name in [None, "foo"]:
                    generation_strategy = get_default_generation_strategy_at_MBM_node(
                        experiment=experiment
                    )

                    # Pick an arbitrary metric from the experiment's optimization config
                    metric_name = none_throws(
                        experiment.optimization_config
                    ).objective.metric_names[0]

                    analysis = CrossValidationPlot(
                        metric_name=metric_name,
                        untransform=untransform,
                        refined_metric_name=refined_metric_name,
                    )

                    _ = analysis.compute(
                        experiment=experiment, generation_strategy=generation_strategy
                    )

    @mock_botorch_optimize
    def test_offline(self) -> None:
        # Test CrossValidationPlot can be computed for a variety of experiments which
        # resemble those we see in an online setting.

        for experiment in get_offline_experiments():
            for untransform in [True, False]:
                for refined_metric_name in [None, "foo"]:
                    generation_strategy = get_default_generation_strategy_at_MBM_node(
                        experiment=experiment
                    )

                    # Pick an arbitrary metric from the experiment's optimization config
                    metric_name = none_throws(
                        experiment.optimization_config
                    ).objective.metric_names[0]

                    analysis = CrossValidationPlot(
                        metric_name=metric_name,
                        untransform=untransform,
                        refined_metric_name=refined_metric_name,
                    )

                    _ = analysis.compute(
                        experiment=experiment, generation_strategy=generation_strategy
                    )

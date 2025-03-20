# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel
from ax.analysis.plotly.scatter import _prepare_data, scatter_plot, ScatterPlot
from ax.exceptions.core import DataRequiredError, UserInputError
from ax.modelbridge.registry import Generators
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment_with_multi_objective,
    get_experiment_with_observations,
)
from ax.utils.testing.mock import mock_botorch_optimize


class TestScatterPlot(TestCase):
    def setUp(self) -> None:
        self.x_metric_name = "branin_a"
        self.y_metric_name = "branin_b"
        self.experiment = get_branin_experiment_with_multi_objective()
        self.sobol_adapter = Generators.SOBOL(search_space=self.experiment.search_space)
        self.exp_w_trial_complete = get_branin_experiment_with_multi_objective(
            with_completed_trial=True
        )

    def test_compute(self) -> None:
        analysis = ScatterPlot(
            x_metric_name=self.x_metric_name,
            y_metric_name=self.y_metric_name,
            show_pareto_frontier=True,
        )
        with self.assertRaisesRegex(UserInputError, "requires an Experiment"):
            analysis.compute()

        card = analysis.compute(experiment=self.exp_w_trial_complete)
        self.assertEqual(card.name, "ScatterPlot")
        self.assertEqual(
            card.title, f"Observed {self.x_metric_name} vs. {self.y_metric_name}"
        )
        self.assertEqual(
            card.subtitle,
            (
                "The scatter plot displays individual data points "
                "representing either observed or predicted values "
                "from the model for two selected metrics. Each point on the plot "
                "corresponds to an arm in a Trial. This visualization is particularly "
                "useful for understanding the trade-off between two metrics in the "
                "experiment."
            ),
        )
        self.assertEqual(card.level, AnalysisCardLevel.HIGH)
        self.assertEqual(card.category, AnalysisCardCategory.INSIGHT)
        self.assertEqual(
            {*card.df.columns},
            {
                "arm_name",
                "trial_index",
                self.x_metric_name,
                self.y_metric_name,
                f"{self.x_metric_name}_sem",
                f"{self.y_metric_name}_sem",
                "is_optimal",
            },
        )
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, "plotly")

    @mock_botorch_optimize
    def test_adhoc_scatter(self) -> None:
        adapter = Generators.BOTORCH_MODULAR(experiment=self.exp_w_trial_complete)
        with self.subTest("No custom points provided"):
            adhoc_cards = scatter_plot(
                adapter=adapter,
                experiment=self.exp_w_trial_complete,
                x_metric_name=self.x_metric_name,
                y_metric_name=self.y_metric_name,
            )
            self.assertEqual(len(adhoc_cards), 1)
            adhoc_card = adhoc_cards[0]
            self.assertEqual(
                adhoc_card.title,
                f"Observed {self.x_metric_name} vs. {self.y_metric_name}",
            )
            self.assertIsNotNone(adhoc_card.blob)
            self.assertEqual(adhoc_card.blob_annotation, "plotly")
        with self.subTest("Custom points provided"):
            gr = adapter.gen(n=3)
            adhoc_cards = scatter_plot(
                adapter=adapter,
                experiment=self.exp_w_trial_complete,
                x_metric_name=self.x_metric_name,
                y_metric_name=self.y_metric_name,
                arms_to_predict_with_adapter=gr.arms,
            )
            self.assertEqual(len(adhoc_cards), 1)
            adhoc_card = adhoc_cards[0]
            self.assertEqual(
                adhoc_card.title,
                f"Observed {self.x_metric_name} vs. {self.y_metric_name}",
            )
            # candidate points are given an index of -1, check that exists
            self.assertTrue((adhoc_card.df["trial_index"] == -1).any())
        with self.subTest("human readable metric names provided"):
            metric_name_mapping = {
                self.x_metric_name: "spunky",
                self.y_metric_name: "sneaky",
            }
            adhoc_cards = scatter_plot(
                adapter=adapter,
                experiment=self.exp_w_trial_complete,
                x_metric_name=self.x_metric_name,
                y_metric_name=self.y_metric_name,
                metric_name_mapping=metric_name_mapping,
            )
            self.assertEqual(len(adhoc_cards), 1)
            adhoc_card = adhoc_cards[0]
            self.assertEqual(
                adhoc_card.title,
                "Observed spunky vs. sneaky",
            )

    def test_prepare_data(self) -> None:
        observations = [[float(i), float(i + 1)] for i in range(10)]
        experiment = get_experiment_with_observations(
            observations=observations, with_sem=True
        )
        # Mark one trial as failed to ensure that it gets filtered out.
        experiment.trials[9].mark_failed(unsafe=True)

        with self.subTest("Test no trial filter applied"):
            data = _prepare_data(
                experiment=experiment, x_metric_name="m1", y_metric_name="m2"
            )

            # Ensure that the data is in the correct shape and only completed trials
            # have rows in the dataframe.
            self.assertEqual(
                len(data),
                len(
                    [
                        trial
                        for trial in experiment.trials.values()
                        if trial.status.is_completed
                    ]
                ),
            )
            self.assertEqual(
                {*data.columns},
                {
                    "trial_index",
                    "arm_name",
                    "m1",
                    "m2",
                    "m1_sem",
                    "m2_sem",
                    "is_optimal",
                },
            )

            # Check data is correct, ignoring the last trial since it was failed.
            for i in range(len(observations) - 1):
                row = data.iloc[i]
                self.assertEqual(row["trial_index"], i)
                self.assertEqual(row["arm_name"], f"{i}_0")
                self.assertEqual(row["m1"], observations[i][0])
                self.assertEqual(row["m2"], observations[i][1])

                # Ensure that the optimal point is labeled correctly
                if i == len(observations) - 2:
                    self.assertTrue(row["is_optimal"])
                else:
                    self.assertFalse(row["is_optimal"])

        with self.subTest("Test trial filter applied"):
            data = _prepare_data(
                experiment=experiment,
                x_metric_name="m1",
                y_metric_name="m2",
                trial_index=3,
            )
            self.assertEqual(len(data), 1)
            self.assertEqual(data["trial_index"].item(), 3)

    def test_it_only_has_observations_with_data_for_both_metrics(self) -> None:
        # GIVEN an experiment with multiple trials and metrics
        t0 = self.experiment.new_batch_trial(
            generator_run=self.sobol_adapter.gen(3)
        ).mark_completed(unsafe=True)
        t1 = self.experiment.new_batch_trial(
            generator_run=self.sobol_adapter.gen(3)
        ).mark_completed(unsafe=True)
        t2 = self.experiment.new_batch_trial(
            generator_run=self.sobol_adapter.gen(3)
        ).mark_completed(unsafe=True)

        # AND given some trials have data for one metric and not the other
        t0.fetch_data(
            metrics=[self.experiment.metrics[self.x_metric_name]],
        )
        t1.fetch_data(
            metrics=[
                self.experiment.metrics[self.x_metric_name],
                self.experiment.metrics[self.y_metric_name],
            ],
        )
        t2.fetch_data(
            metrics=[self.experiment.metrics[self.y_metric_name]],
        )

        # WHEN we call `compute`
        analysis = ScatterPlot(
            x_metric_name=self.x_metric_name,
            y_metric_name=self.y_metric_name,
            show_pareto_frontier=True,
        )
        card = analysis.compute(experiment=self.experiment)

        # THEN it only has observations with data for both metrics
        self.assertEqual(
            card.df["trial_index"].unique(),
            [t1.index],
        )

    def test_it_must_have_some_observations_with_data_for_both_metrics(self) -> None:
        # GIVEN an experiment with multiple trials and metrics
        t0 = self.experiment.new_batch_trial(
            generator_run=self.sobol_adapter.gen(3)
        ).mark_completed(unsafe=True)
        t1 = self.experiment.new_batch_trial(
            generator_run=self.sobol_adapter.gen(3)
        ).mark_completed(unsafe=True)

        # AND given some trials have data for one metric and not the other
        t0.fetch_data(
            metrics=[self.experiment.metrics[self.x_metric_name]],
        )
        t1.fetch_data(
            metrics=[self.experiment.metrics[self.y_metric_name]],
        )

        # WHEN we call `compute`
        analysis = ScatterPlot(
            x_metric_name=self.x_metric_name,
            y_metric_name=self.y_metric_name,
            show_pareto_frontier=True,
        )

        # THEN it raises an error
        with self.assertRaisesRegex(
            DataRequiredError,
            "No observations have data for both branin_a and branin_b.",
        ):
            analysis.compute(experiment=self.experiment)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from unittest.mock import patch

import torch

from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.plotly.arm_effects.predicted_effects import PredictedEffectsPlot
from ax.analysis.plotly.arm_effects.utils import get_predictions_by_arm
from ax.core.observation import ObservationFeatures
from ax.core.trial import Trial
from ax.exceptions.core import UserInputError
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.prediction_utils import predict_at_point
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_metric,
    get_branin_outcome_constraint,
)
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_sobol_MBM_MTGP_gs
from botorch.utils.probability.utils import compute_log_prob_feas_from_bounds
from pyre_extensions import assert_is_instance, none_throws


class TestPredictedEffectsPlot(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.generation_strategy = get_sobol_MBM_MTGP_gs()

    def test_compute_for_requires_an_exp(self) -> None:
        analysis = PredictedEffectsPlot(metric_name="branin")

        with self.assertRaisesRegex(UserInputError, "requires an Experiment"):
            analysis.compute()

    def test_compute_for_requires_a_gs(self) -> None:
        analysis = PredictedEffectsPlot(metric_name="branin")
        experiment = get_branin_experiment()
        with self.assertRaisesRegex(UserInputError, "requires a GenerationStrategy"):
            analysis.compute(experiment=experiment)

    def test_compute_for_requires_trials(self) -> None:
        analysis = PredictedEffectsPlot(metric_name="branin")
        experiment = get_branin_experiment()
        generation_strategy = choose_generation_strategy(
            search_space=experiment.search_space,
            experiment=experiment,
        )
        with self.assertRaisesRegex(UserInputError, "it has no trials"):
            analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )

    def test_compute_for_requires_a_model_that_predicts(self) -> None:
        analysis = PredictedEffectsPlot(metric_name="branin")
        experiment = get_branin_experiment(with_batch=True, with_completed_batch=True)
        generation_strategy = choose_generation_strategy(
            search_space=experiment.search_space,
            experiment=experiment,
        )
        with self.assertRaisesRegex(
            UserInputError, "where the current model supports prediction"
        ):
            analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )

    @mock_botorch_optimize
    def test_compute(self) -> None:
        # GIVEN an experiment with metrics and batch trials
        experiment = get_branin_experiment(with_status_quo=True)
        none_throws(experiment.optimization_config).outcome_constraints = [
            get_branin_outcome_constraint(name="constraint_branin")
        ]
        experiment.add_tracking_metric(get_branin_metric(name="tracking_branin"))
        generation_strategy = self.generation_strategy
        experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        ).set_status_quo_with_weight(
            status_quo=experiment.status_quo, weight=1.0
        ).mark_completed(unsafe=True)
        experiment.fetch_data()
        experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        ).set_status_quo_with_weight(status_quo=experiment.status_quo, weight=1.0)
        experiment.fetch_data()
        # Ensure the current model is Botorch
        self.assertEqual(none_throws(generation_strategy.model)._model_key, "BoTorch")
        for metric in experiment.metrics:
            with self.subTest(metric=metric):
                # WHEN we compute the analysis for a metric
                analysis = PredictedEffectsPlot(metric_name=metric)
                card = analysis.compute(
                    experiment=experiment, generation_strategy=generation_strategy
                )
                # THEN it makes a card with the right name, title, and subtitle
                self.assertEqual(card.name, "PredictedEffectsPlot")
                self.assertEqual(card.title, f"Predicted Effects for {metric}")
                self.assertEqual(
                    card.subtitle,
                    "View a candidate trial and its arms' predicted metric values",
                )
                # AND THEN it has an appropriate level based on whether we're
                # optimizing for the metric
                self.assertEqual(
                    card.level,
                    (
                        AnalysisCardLevel.HIGH
                        if metric == "branin"
                        else (
                            AnalysisCardLevel.HIGH - 1
                            if metric == "constraint_branin"
                            else AnalysisCardLevel.HIGH - 2
                        )
                    ),
                )
                # AND THEN it has the right rows and columns in the dataframe
                self.assertEqual(
                    {*card.df.columns},
                    {
                        "arm_name",
                        "source",
                        "parameters",
                        "mean",
                        "sem",
                        "error_margin",
                        "constraints_violated",
                        "overall_probability_constraints_violated",
                    },
                )
                self.assertIsNotNone(card.blob)
                self.assertEqual(card.blob_annotation, "plotly")
                for trial in experiment.trials.values():
                    for arm in trial.arms:
                        self.assertIn(arm.name, card.df["arm_name"].unique())

    @mock_botorch_optimize
    def test_it_does_not_plot_abandoned_arms(self) -> None:
        # GIVEN an experiment with candidate and abandoned arms
        # in the completed and candidate trials
        experiment = get_branin_experiment(with_status_quo=True)
        generation_strategy = self.generation_strategy
        completed_trial = experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        ).set_status_quo_with_weight(status_quo=experiment.status_quo, weight=1.0)
        completed_trial.mark_arm_abandoned(
            arm_name="0_0", reason="This arm is bad, I'm abandoning it"
        )
        completed_trial.mark_completed(unsafe=True)
        experiment.fetch_data()
        candidate_trial = experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        ).set_status_quo_with_weight(status_quo=experiment.status_quo, weight=1.0)
        candidate_trial.mark_arm_abandoned(
            arm_name="1_0", reason="This arm is bad, I'm abandoning it"
        )
        # WHEN we compute the analysis
        analysis = PredictedEffectsPlot(
            metric_name=none_throws(
                experiment.optimization_config
            ).objective.metric.name
        )
        card = analysis.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )
        # THEN it has the right arms
        plotted_arm_names = set(card.df["arm_name"].unique())
        for trial in experiment.trials.values():
            for arm in trial.arms:
                if arm.name in ("0_0", "1_0"):
                    self.assertNotIn(arm.name, plotted_arm_names)
                else:
                    self.assertIn(arm.name, plotted_arm_names)

    @mock_botorch_optimize
    def test_compute_multitask(self) -> None:
        # GIVEN an experiment with candidates generated with a multitask model
        experiment = get_branin_experiment(with_status_quo=True)
        generation_strategy = self.generation_strategy
        experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        ).set_status_quo_with_weight(
            status_quo=experiment.status_quo, weight=1
        ).mark_completed(unsafe=True)
        experiment.fetch_data()
        experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        ).set_status_quo_with_weight(
            status_quo=experiment.status_quo, weight=1
        ).mark_completed(unsafe=True)
        experiment.fetch_data()
        # leave as a candidate
        experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment,
                n=10,
                fixed_features=ObservationFeatures(parameters={}, trial_index=1),
            )
        ).set_status_quo_with_weight(status_quo=experiment.status_quo, weight=1)
        experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment,
                n=10,
                fixed_features=ObservationFeatures(parameters={}, trial_index=1),
            )
        ).set_status_quo_with_weight(status_quo=experiment.status_quo, weight=1)
        self.assertEqual(none_throws(generation_strategy.model)._model_key, "ST_MTGP")
        # WHEN we compute the analysis
        analysis = PredictedEffectsPlot(metric_name="branin")
        with patch(
            f"{get_predictions_by_arm.__module__}.predict_at_point",
            wraps=predict_at_point,
        ) as predict_at_point_spy:
            card = analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )
        # THEN it has the right rows for arms with data, as well as the latest trial
        arms_with_data = set(experiment.lookup_data().df["arm_name"].unique())
        max_trial_index = max(experiment.trials.keys())
        for trial in experiment.trials.values():
            if trial.status.expecting_data or trial.index == max_trial_index:
                for arm in trial.arms:
                    self.assertIn(arm.name, card.df["arm_name"].unique())
            else:
                # arms from other candidate trials are only in the df if they
                # are repeated in the target trial
                for arm in trial.arms:
                    self.assertTrue(
                        arm.name not in card.df["arm_name"].unique()
                        # it's repeated in another trial
                        or arm.name in arms_with_data
                        or arm.name in experiment.trials[max_trial_index].arms_by_name,
                        arm.name,
                    )
        # AND THEN it always predicts for the target trial
        self.assertEqual(
            len(
                {
                    call[1]["obsf"].trial_index
                    for call in predict_at_point_spy.call_args_list
                }
            ),
            1,
        )

    @mock_botorch_optimize
    def test_it_does_not_plot_abandoned_trials(self) -> None:
        # GIVEN an experiment with candidate and abandoned trials
        experiment = get_branin_experiment()
        generation_strategy = self.generation_strategy
        experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        ).mark_completed(unsafe=True)
        experiment.fetch_data()
        # candidate trial
        experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        )
        experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        ).mark_abandoned()
        arms_with_data = set(experiment.lookup_data().df["arm_name"].unique())
        # WHEN we compute the analysis
        analysis = PredictedEffectsPlot(metric_name="branin")
        card = analysis.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )
        # THEN it has the right rows for arms with data, as well as the latest
        # non abandoned trial (with index 1)
        for arm in experiment.trials[0].arms + experiment.trials[1].arms:
            self.assertIn(arm.name, card.df["arm_name"].unique())

        # AND THEN it does not have the arms from the abandoned trial (index 2)
        for arm in experiment.trials[2].arms:
            self.assertTrue(
                arm.name not in card.df["arm_name"].unique()
                # it's repeated in another trial
                or arm.name in arms_with_data
                or arm.name in experiment.trials[1].arms_by_name,
                arm.name,
            )

    @mock_botorch_optimize
    def test_it_works_for_non_batch_experiments(self) -> None:
        # GIVEN an experiment with the default generation strategy
        experiment = get_branin_experiment(with_batch=False)
        generation_strategy = choose_generation_strategy(
            search_space=experiment.search_space,
            experiment=experiment,
        )
        # AND GIVEN we generate all Sobol trials and one GPEI trial
        sobol_key = Models.SOBOL.value
        last_model_key = sobol_key
        while last_model_key == sobol_key:
            trial = experiment.new_trial(
                generator_run=generation_strategy._gen_with_multiple_nodes(
                    experiment=experiment,
                    n=1,
                )[0]
            )
            last_model_key = none_throws(trial.generator_run)._model_key
            if last_model_key == sobol_key:
                trial.mark_running(no_runner_required=True)
                trial.mark_completed()
                trial.fetch_data()

        # WHEN we compute the analysis
        analysis = PredictedEffectsPlot(metric_name="branin")
        card = analysis.compute(
            experiment=experiment,
            generation_strategy=generation_strategy,
        )
        # THEN it has all arms represented in the dataframe
        for trial in experiment.trials.values():
            self.assertIn(
                none_throws(assert_is_instance(trial, Trial).arm).name,
                card.df["arm_name"].unique(),
            )

    @mock_botorch_optimize
    def test_constraints(self) -> None:
        # GIVEN an experiment with metrics and batch trials
        experiment = get_branin_experiment(with_status_quo=True)
        none_throws(experiment.optimization_config).outcome_constraints = [
            get_branin_outcome_constraint(name="constraint_branin_1"),
            get_branin_outcome_constraint(name="constraint_branin_2"),
        ]
        generation_strategy = self.generation_strategy
        trial = experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            ),
        )
        trial.set_status_quo_with_weight(status_quo=experiment.status_quo, weight=1.0)
        trial.mark_completed(unsafe=True)
        experiment.fetch_data()
        trial = experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            ),
        )
        trial.set_status_quo_with_weight(status_quo=experiment.status_quo, weight=1.0)
        # WHEN we compute the analysis and constraints are violated
        analysis = PredictedEffectsPlot(metric_name="branin")
        with self.subTest("violated"):
            with patch(
                f"{compute_log_prob_feas_from_bounds.__module__}.log_ndtr",
                side_effect=lambda t: torch.as_tensor([[0.25]] * t.size()[0]).log(),
            ):
                card = analysis.compute(
                    experiment=experiment, generation_strategy=generation_strategy
                )
            # THEN it marks that constraints are violated for the non-SQ arms
            non_sq_df = card.df[card.df["arm_name"] != "status_quo"]
            sq_row = card.df[card.df["arm_name"] == "status_quo"]
            self.assertTrue(
                all(non_sq_df["constraints_violated"] != "No constraints violated"),
                non_sq_df["constraints_violated"],
            )
            self.assertTrue(
                all(
                    non_sq_df["constraints_violated"]
                    == (
                        "<br />  constraint_branin_1: 75.0% chance violated"
                        "<br />  constraint_branin_2: 75.0% chance violated"
                    )
                ),
                str(non_sq_df["constraints_violated"][0]),
            )
            # AND THEN it marks that constraints are not violated for the SQ
            self.assertEqual(
                sq_row["overall_probability_constraints_violated"].iloc[0], 0
            )
            self.assertEqual(
                sq_row["constraints_violated"].iloc[0], "No constraints violated"
            )

        # WHEN we compute the analysis and constraints are violated
        with self.subTest("not violated"):
            with patch(
                f"{compute_log_prob_feas_from_bounds.__module__}.log_ndtr",
                side_effect=lambda t: torch.as_tensor([[1]] * t.size()[0]).log(),
            ):
                card = analysis.compute(
                    experiment=experiment, generation_strategy=generation_strategy
                )
            # THEN it marks that constraints are not violated
            self.assertTrue(
                all(card.df["constraints_violated"] == "No constraints violated"),
                str(card.df["constraints_violated"]),
            )

        # AND THEN it has not modified the constraints
        opt_config = none_throws(experiment.optimization_config)
        self.assertTrue(opt_config.outcome_constraints[0].relative)
        self.assertTrue(opt_config.outcome_constraints[1].relative)

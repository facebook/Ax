# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from unittest.mock import patch

import torch

from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.plotly.arm_effects.insample_effects import InSampleEffectsPlot
from ax.analysis.plotly.arm_effects.utils import get_predictions_by_arm
from ax.exceptions.core import DataRequiredError, UserInputError
from ax.modelbridge.prediction_utils import predict_at_point
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_metric,
    get_branin_outcome_constraint,
)
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_sobol_MBM_MTGP_gs
from botorch.utils.probability.utils import compute_log_prob_feas_from_bounds
from pyre_extensions import none_throws


class TestInsampleEffectsPlot(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.generation_strategy = get_sobol_MBM_MTGP_gs()

    def test_compute_for_requires_an_exp(self) -> None:
        analysis = InSampleEffectsPlot(
            metric_name="branin", trial_index=0, use_modeled_effects=True
        )

        with self.assertRaisesRegex(UserInputError, "requires an Experiment"):
            analysis.compute()

    @mock_botorch_optimize
    def test_compute_uses_gs_model_if_possible(self) -> None:
        # GIVEN an experiment and GS with a Botorch model
        experiment = get_branin_experiment(with_status_quo=True)
        generation_strategy = self.generation_strategy
        experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        ).set_status_quo_with_weight(
            status_quo=experiment.status_quo, weight=1.0
        ).mark_completed(unsafe=True)
        experiment.fetch_data()
        generation_strategy._gen_with_multiple_nodes(experiment=experiment, n=10)
        # Ensure the current model is Botorch
        self.assertEqual(none_throws(generation_strategy.model)._model_key, "BoTorch")
        # WHEN we compute the analysis
        analysis = InSampleEffectsPlot(
            metric_name="branin", trial_index=0, use_modeled_effects=True
        )
        with patch(
            f"{get_predictions_by_arm.__module__}.predict_at_point",
            wraps=predict_at_point,
        ) as predict_at_point_spy:
            card = analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )
        # THEN it uses the model from the GS
        models_used_for_prediction = [
            call[1]["model"]._model_key for call in predict_at_point_spy.call_args_list
        ]
        self.assertTrue(
            [all(m == "BoTorch" for m in models_used_for_prediction)],
            models_used_for_prediction,
        )
        # AND THEN it has predictions for all arms
        trial = experiment.trials[0]
        self.assertEqual(
            len(card.df),
            len(trial.arms),
        )
        for arm in trial.arms:
            self.assertIn(arm.name, card.df["arm_name"].unique())

    def test_compute_modeled_can_use_ebts_for_gs_with_non_predictive_model(
        self,
    ) -> None:
        # GIVEN an experiment and GS with a non Botorch model
        experiment = get_branin_experiment()
        generation_strategy = self.generation_strategy
        generation_strategy.experiment = experiment
        experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        ).mark_completed(unsafe=True)
        experiment.fetch_data()
        # Ensure the current model is not Botorch
        self.assertEqual(none_throws(generation_strategy.model)._model_key, "Sobol")
        # WHEN we compute the analysis
        analysis = InSampleEffectsPlot(
            metric_name="branin", trial_index=0, use_modeled_effects=True
        )
        with patch(
            f"{get_predictions_by_arm.__module__}.predict_at_point",
            wraps=predict_at_point,
        ) as predict_at_point_spy:
            card = analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )
        # THEN it uses the empirical bayes model
        models_used_for_prediction = [
            call[1]["model"]._model_key for call in predict_at_point_spy.call_args_list
        ]
        self.assertTrue(
            [all(m == "EB" for m in models_used_for_prediction)],
            models_used_for_prediction,
        )
        # AND THEN it has predictions for all arms
        trial = experiment.trials[0]
        self.assertEqual(
            len(card.df),
            len(trial.arms),
        )
        for arm in trial.arms:
            self.assertIn(arm.name, card.df["arm_name"].unique())

        # AND THEN the card is labeled correctly
        self.assertEqual(card.name, "ModeledEffectsPlot")
        self.assertEqual(card.title, "Modeled Effects for branin on trial 0")
        self.assertEqual(
            card.subtitle, "View a trial and its arms' modeled metric values"
        )
        # +2 because it's on objective, +1 because it's modeled
        self.assertEqual(card.level, AnalysisCardLevel.MID + 3)

    def test_compute_modeled_can_use_ebts_for_no_gs(self) -> None:
        # GIVEN an experiment with a trial with data
        experiment = get_branin_experiment()
        generation_strategy = self.generation_strategy
        generation_strategy.experiment = experiment
        experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        ).mark_completed(unsafe=True)
        experiment.fetch_data()
        # WHEN we compute the analysis
        analysis = InSampleEffectsPlot(
            metric_name="branin", trial_index=0, use_modeled_effects=True
        )
        with patch(
            f"{get_predictions_by_arm.__module__}.predict_at_point",
            wraps=predict_at_point,
        ) as predict_at_point_spy:
            card = analysis.compute(experiment=experiment, generation_strategy=None)
        # THEN it uses the empirical bayes model
        models_used_for_prediction = [
            call[1]["model"]._model_key for call in predict_at_point_spy.call_args_list
        ]
        self.assertTrue(
            [all(m == "EB" for m in models_used_for_prediction)],
            models_used_for_prediction,
        )
        # AND THEN it has predictions for all arms
        trial = experiment.trials[0]
        self.assertEqual(
            len(card.df),
            len(trial.arms),
        )
        for arm in trial.arms:
            self.assertIn(arm.name, card.df["arm_name"].unique())

        # AND THEN the card is labeled correctly
        self.assertEqual(card.name, "ModeledEffectsPlot")
        self.assertEqual(card.title, "Modeled Effects for branin on trial 0")
        self.assertEqual(
            card.subtitle, "View a trial and its arms' modeled metric values"
        )
        # +2 because it's on objective, +1 because it's modeled
        self.assertEqual(card.level, AnalysisCardLevel.MID + 3)

    def test_compute_unmodeled_uses_thompson(self) -> None:
        # GIVEN an experiment with a trial with data
        experiment = get_branin_experiment()
        generation_strategy = self.generation_strategy
        generation_strategy.experiment = experiment
        trial = experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        )
        trial.mark_arm_abandoned(
            arm_name="0_3",
            reason=(
                "We need to make sure it doesn't try to predict for abandoned arms "
                "because the Thompson model doesn't support out-of-sample prediction."
            ),
        )
        trial.mark_completed(unsafe=True)
        experiment.fetch_data()
        # WHEN we compute the analysis
        analysis = InSampleEffectsPlot(
            metric_name="branin", trial_index=0, use_modeled_effects=False
        )
        with patch(
            f"{get_predictions_by_arm.__module__}.predict_at_point",
            wraps=predict_at_point,
        ) as predict_at_point_spy:
            card = analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )
        # THEN it uses the thompson model
        models_used_for_prediction = [
            call[1]["model"]._model_key for call in predict_at_point_spy.call_args_list
        ]
        self.assertTrue(
            [all(m == "Thompson" for m in models_used_for_prediction)],
            models_used_for_prediction,
        )
        # AND THEN it has predictions for all arms
        trial = experiment.trials[0]
        data_df = experiment.lookup_data(trial_indices=[trial.index]).df
        self.assertEqual(
            len(card.df),
            # -1 because the abandoned arm is not in card.df
            len(trial.arms) - 1,
        )
        for arm in trial.arms:
            # arm 0_3 is abandoned so it's not in card.df
            if arm.name == "0_3":
                continue
            self.assertIn(arm.name, card.df["arm_name"].unique())
            self.assertAlmostEqual(
                card.df.loc[card.df["arm_name"] == arm.name, "mean"].item(),
                data_df.loc[data_df["arm_name"] == arm.name, "mean"].item(),
            )
            self.assertAlmostEqual(
                card.df.loc[card.df["arm_name"] == arm.name, "sem"].item(),
                data_df.loc[data_df["arm_name"] == arm.name, "sem"].item(),
            )

        # AND THEN the card is labeled correctly
        self.assertEqual(card.name, "ObservedEffectsPlot")
        self.assertEqual(card.title, "Observed Effects for branin on trial 0")
        self.assertEqual(
            card.subtitle, "View a trial and its arms' observed metric values"
        )
        # +2 because it's on objective
        self.assertEqual(card.level, AnalysisCardLevel.MID + 2)

    def test_compute_requires_data_for_the_metric_on_the_trial_without_a_model(
        self,
    ) -> None:
        # GIVEN an experiment with a trial with no data
        experiment = get_branin_experiment()
        generation_strategy = self.generation_strategy
        generation_strategy.experiment = experiment
        experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        ).mark_completed(unsafe=True)
        self.assertTrue(experiment.lookup_data().df.empty)
        # WHEN we compute the analysis
        analysis = InSampleEffectsPlot(
            metric_name="branin",
            trial_index=0,
            use_modeled_effects=False,
        )
        with self.assertRaisesRegex(
            DataRequiredError,
            "Cannot plot effects for 'branin' on trial 0 because it has no data.",
        ):
            analysis.compute(experiment=experiment, generation_strategy=None)
            # THEN it raises an error

    @mock_botorch_optimize
    def test_compute_requires_data_for_the_metric_on_the_trial_with_a_model(
        self,
    ) -> None:
        # GIVEN an experiment and GS with a Botorch model
        experiment = get_branin_experiment(with_status_quo=True)
        generation_strategy = self.generation_strategy
        generation_strategy.experiment = experiment
        experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            )
        ).set_status_quo_with_weight(
            status_quo=experiment.status_quo, weight=1.0
        ).mark_completed(unsafe=True)
        experiment.fetch_data()
        # AND GIVEN the experiment has a trial with no data
        empty_trial = experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            ),
        )
        # Ensure the current model is Botorch
        self.assertEqual(none_throws(generation_strategy.model)._model_key, "BoTorch")
        self.assertTrue(
            experiment.lookup_data(trial_indices=[empty_trial.index]).df.empty
        )
        # WHEN we compute the analysis
        analysis = InSampleEffectsPlot(
            metric_name="branin",
            trial_index=empty_trial.index,
            use_modeled_effects=True,
        )
        with self.assertRaisesRegex(
            DataRequiredError,
            (
                f"Cannot plot effects for 'branin' on trial {empty_trial.index} "
                "because it has no data."
            ),
        ):
            analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )
            # THEN it raises an error

    @mock_botorch_optimize
    def test_constraints(self) -> None:
        # GIVEN an experiment with metrics and batch trials
        experiment = get_branin_experiment(with_status_quo=True)
        none_throws(experiment.optimization_config).outcome_constraints = [
            get_branin_outcome_constraint(name="constraint_branin_1"),
            get_branin_outcome_constraint(name="constraint_branin_2"),
        ]
        generation_strategy = self.generation_strategy
        generation_strategy.experiment = experiment
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
        analysis = InSampleEffectsPlot(
            metric_name="branin", trial_index=0, use_modeled_effects=True
        )
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

    def test_level(self) -> None:
        # GIVEN an experiment with metrics and batch trials
        experiment = get_branin_experiment(with_status_quo=True)
        none_throws(experiment.optimization_config).outcome_constraints = [
            get_branin_outcome_constraint(name="constraint_branin"),
        ]
        experiment.add_tracking_metric(get_branin_metric(name="tracking_branin"))
        generation_strategy = self.generation_strategy
        generation_strategy.experiment = experiment
        trial = experiment.new_batch_trial(
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment, n=10
            ),
        ).set_status_quo_with_weight(status_quo=experiment.status_quo, weight=1.0)
        trial.mark_completed(unsafe=True)
        experiment.fetch_data()

        metric_to_level = {
            "branin": AnalysisCardLevel.MID + 2,
            "constraint_branin": AnalysisCardLevel.MID + 1,
            "tracking_branin": AnalysisCardLevel.MID,
        }

        for metric, level in metric_to_level.items():
            with self.subTest("objective is high"):
                # WHEN we compute the analysis for an objective
                analysis = InSampleEffectsPlot(
                    # trial_index and use_modeled_effects don't affect the level
                    metric_name=metric,
                    trial_index=0,
                    use_modeled_effects=False,
                )
                card = analysis.compute(experiment=experiment)
                # THEN the card has the correct level
                self.assertEqual(card.level, level)

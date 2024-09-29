# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.plotly.predicted_effects import PredictedEffectsPlot
from ax.core.base_trial import TrialStatus
from ax.core.observation import ObservationFeatures
from ax.core.trial import Trial
from ax.exceptions.core import UserInputError
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_node import GenerationNode
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.registry import Models
from ax.modelbridge.transition_criterion import MaxTrials
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_metric,
    get_branin_outcome_constraint,
)
from ax.utils.testing.mock import fast_botorch_optimize
from pyre_extensions import none_throws


class TestParallelCoordinatesPlot(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.generation_strategy = GenerationStrategy(
            nodes=[
                GenerationNode(
                    node_name="Sobol",
                    model_specs=[ModelSpec(model_enum=Models.SOBOL)],
                    transition_criteria=[
                        MaxTrials(
                            threshold=1,
                            transition_to="GPEI",
                        )
                    ],
                ),
                GenerationNode(
                    node_name="GPEI",
                    model_specs=[
                        ModelSpec(
                            model_enum=Models.BOTORCH_MODULAR,
                        ),
                    ],
                    transition_criteria=[
                        MaxTrials(
                            threshold=1,
                            transition_to="MTGP",
                            only_in_statuses=[
                                TrialStatus.RUNNING,
                                TrialStatus.COMPLETED,
                                TrialStatus.EARLY_STOPPED,
                            ],
                        )
                    ],
                ),
                GenerationNode(
                    node_name="MTGP",
                    model_specs=[
                        ModelSpec(
                            model_enum=Models.ST_MTGP,
                        ),
                    ],
                ),
            ],
        )

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

    @fast_botorch_optimize
    def test_compute(self) -> None:
        # GIVEN an experiment with metrics and batch trials
        experiment = get_branin_experiment(with_status_quo=True)
        none_throws(experiment.optimization_config).outcome_constraints = [
            get_branin_outcome_constraint(name="constraint_branin")
        ]
        experiment.add_tracking_metric(get_branin_metric(name="tracking_branin"))
        generation_strategy = self.generation_strategy
        experiment.new_batch_trial(
            generator_run=generation_strategy.gen(experiment=experiment, n=10)
        ).set_status_quo_with_weight(
            status_quo=experiment.status_quo, weight=1.0
        ).mark_completed(
            unsafe=True
        )
        experiment.fetch_data()
        experiment.new_batch_trial(
            generator_run=generation_strategy.gen(experiment=experiment, n=10)
        ).set_status_quo_with_weight(status_quo=experiment.status_quo, weight=1.0)
        experiment.fetch_data()
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
                            AnalysisCardLevel.MID
                            if metric == "constraint_branin"
                            else AnalysisCardLevel.LOW
                        )
                    ),
                )
                # AND THEN it has the right rows and columns in the dataframe
                self.assertEqual(
                    {*card.df.columns},
                    {"arm_name", "source", "x1", "x2", "mean", "error_margin"},
                )
                self.assertIsNotNone(card.blob)
                self.assertEqual(card.blob_annotation, "plotly")
                for trial in experiment.trials.values():
                    for arm in trial.arms:
                        self.assertIn(arm.name, card.df["arm_name"].unique())

    @fast_botorch_optimize
    def test_compute_multitask(self) -> None:
        # GIVEN an experiment with candidates generated with a multitask model
        experiment = get_branin_experiment()
        generation_strategy = self.generation_strategy
        experiment.new_batch_trial(
            generator_run=generation_strategy.gen(experiment=experiment, n=10)
        ).mark_completed(unsafe=True)
        experiment.fetch_data()
        experiment.new_batch_trial(
            generator_run=generation_strategy.gen(experiment=experiment, n=10)
        ).mark_completed(unsafe=True)
        experiment.fetch_data()
        # leave as a candidate
        experiment.new_batch_trial(
            generator_run=generation_strategy.gen(
                experiment=experiment,
                n=10,
                fixed_features=ObservationFeatures(parameters={}, trial_index=1),
            )
        )
        experiment.new_batch_trial(
            generator_run=generation_strategy.gen(
                experiment=experiment,
                n=10,
                fixed_features=ObservationFeatures(parameters={}, trial_index=1),
            )
        )
        self.assertEqual(none_throws(generation_strategy.model)._model_key, "ST_MTGP")
        # WHEN we compute the analysis
        analysis = PredictedEffectsPlot(metric_name="branin")
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

    @fast_botorch_optimize
    def test_it_does_not_plot_abandoned_trials(self) -> None:
        # GIVEN an experiment with candidate and abandoned trials
        experiment = get_branin_experiment()
        generation_strategy = self.generation_strategy
        experiment.new_batch_trial(
            generator_run=generation_strategy.gen(experiment=experiment, n=10)
        ).mark_completed(unsafe=True)
        experiment.fetch_data()
        # candidate trial
        experiment.new_batch_trial(
            generator_run=generation_strategy.gen(experiment=experiment, n=10)
        )
        experiment.new_batch_trial(
            generator_run=generation_strategy.gen(experiment=experiment, n=10)
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

    @fast_botorch_optimize
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
                generator_run=generation_strategy.gen(
                    experiment=experiment, n=1, pending_observation=True
                )
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
                none_throws(checked_cast(Trial, trial).arm).name,
                card.df["arm_name"].unique(),
            )

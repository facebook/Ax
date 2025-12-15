# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.adapter.registry import Generators
from ax.analysis.results import ArmEffectsPair, ResultsAnalysis
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.core.analysis_card import AnalysisCardGroup, ErrorAnalysisCard
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.optimization_config import Objective, OptimizationConfig
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import (
    GenerationNode,
    GenerationStrategy,
)
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_optimization_config,
    get_data,
    get_experiment_with_scalarized_objective_and_outcome_constraint,
    get_offline_experiments,
    get_online_experiments,
)
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_default_generation_strategy_at_MBM_node
from pyre_extensions import assert_is_instance, none_throws


class TestResultsAnalysis(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.client = Client()
        self.client.configure_experiment(
            name="test_experiment",
            parameters=[
                RangeParameterConfig(
                    name="x1",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
                RangeParameterConfig(
                    name="x2",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
            ],
        )

    def test_validate_applicable_state(self) -> None:
        analysis = ResultsAnalysis()

        with self.subTest("requires_experiment"):
            error_message = analysis.validate_applicable_state()
            self.assertIsNotNone(error_message)
            self.assertIn("Requires an Experiment", none_throws(error_message))

        with self.subTest("requires_trials"):
            experiment = get_branin_experiment()
            error_message = analysis.validate_applicable_state(experiment=experiment)
            self.assertIsNotNone(error_message)
            self.assertIn("has no trials", none_throws(error_message))

        with self.subTest("requires_data"):
            experiment = get_branin_experiment()
            experiment.new_trial()
            error_message = analysis.validate_applicable_state(experiment=experiment)
            self.assertIsNotNone(error_message)
            self.assertIn("has no data", none_throws(error_message))

    @mock_botorch_optimize
    def test_compute_with_single_objective_no_constraints(self) -> None:
        # Setup: Create experiment with single objective and no constraints
        client = self.client
        client.configure_optimization(objective="foo")

        # Generate and complete a trial
        client.get_next_trials(max_trials=1)
        client.complete_trial(trial_index=0, raw_data={"foo": 1.0})

        # Create generation strategy
        generation_strategy = get_default_generation_strategy_at_MBM_node(
            experiment=client._experiment
        )

        # Execute: Compute ResultsAnalysis
        analysis = ResultsAnalysis()
        card_group = analysis.compute(
            experiment=client._experiment,
            generation_strategy=generation_strategy,
        )

        # Assert: Should produce valid card group
        self.assertIsNotNone(card_group)
        self.assertEqual(card_group.title, "Results Analysis")
        self.assertGreater(len(card_group.children), 0)

        # Assert: Should have arm effects pair
        child_names = [child.name for child in card_group.children]
        self.assertTrue(
            any("ArmEffects" in name for name in child_names),
            "Should have arm effects in children",
        )

        # Assert: Should have best trials
        self.assertTrue(
            any("BestTrials" in name for name in child_names),
            "Should have best trials in children",
        )

        # Assert: No error cards should be present
        for card in card_group.flatten():
            self.assertNotIsInstance(card, ErrorAnalysisCard)

    @mock_botorch_optimize
    def test_compute_with_multiple_objectives(self) -> None:
        # Setup: Create experiment with multiple objectives
        client = Client()
        client.configure_experiment(
            name="multi_objective_experiment",
            parameters=[
                RangeParameterConfig(
                    name="x1",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
            ],
        )
        client.configure_optimization(objective="foo, bar")

        # Generate and complete a trial
        client.get_next_trials(max_trials=1)
        client.complete_trial(trial_index=0, raw_data={"foo": 1.0, "bar": 2.0})

        # Create generation strategy
        generation_strategy = get_default_generation_strategy_at_MBM_node(
            experiment=client._experiment
        )

        # Execute: Compute ResultsAnalysis
        analysis = ResultsAnalysis()
        card_group = analysis.compute(
            experiment=client._experiment,
            generation_strategy=generation_strategy,
        )

        # Assert: Should have objective scatter plots for multiple objectives
        child_names = [child.name for child in card_group.children]
        self.assertTrue(
            any("Objective Scatter" in name for name in child_names),
            "Should have objective scatter plots for multiple objectives",
        )

        # Assert: No error cards should be present
        for card in card_group.flatten():
            self.assertNotIsInstance(card, ErrorAnalysisCard)

    @mock_botorch_optimize
    def test_compute_with_constraints(self) -> None:
        # Setup: Create experiment with objective and constraints
        client = Client()
        client.configure_experiment(
            name="constrained_experiment",
            parameters=[
                RangeParameterConfig(
                    name="x1",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
            ],
        )
        client.configure_optimization(
            objective="foo",
            outcome_constraints=["bar >= 0.5"],
        )

        # Generate and complete a trial
        client.get_next_trials(max_trials=1)
        client.complete_trial(trial_index=0, raw_data={"foo": 1.0, "bar": 0.6})

        # Create generation strategy
        generation_strategy = get_default_generation_strategy_at_MBM_node(
            experiment=client._experiment
        )

        # Execute: Compute ResultsAnalysis
        analysis = ResultsAnalysis()
        card_group = analysis.compute(
            experiment=client._experiment,
            generation_strategy=generation_strategy,
        )

        # Assert: Should have constraint scatter plots
        child_names = [child.name for child in card_group.children]
        self.assertTrue(
            any("Constraint Scatter" in name for name in child_names),
            "Should have constraint scatter plots when constraints are present",
        )

        # Assert: No error cards should be present
        for card in card_group.flatten():
            self.assertNotIsInstance(card, ErrorAnalysisCard)

    @mock_botorch_optimize
    def test_compute_with_scalarized_constraints(self) -> None:
        # Setup: Create experiment with scalarized outcome constraints
        experiment = get_experiment_with_scalarized_objective_and_outcome_constraint()
        trial = experiment.new_batch_trial(
            generator_run=GeneratorRun(
                arms=[
                    Arm(parameters={"w": 5.1, "x": 5, "y": "foo", "z": True, "d": 11.2})
                ]
            ),
            should_add_status_quo_arm=True,
        )
        trial.mark_running(no_runner_required=True)

        # Attach data
        data = []
        for m_name in experiment.metrics.keys():
            data.append(
                get_data(
                    metric_name=m_name,
                    trial_index=trial.index,
                    num_non_sq_arms=1,
                    include_sq=True,
                )
            )
        experiment.attach_data(Data.from_multiple_data(data))
        trial.mark_completed()

        # Create generation strategy
        generation_strategy = get_default_generation_strategy_at_MBM_node(
            experiment=experiment
        )

        # Execute: Compute ResultsAnalysis
        analysis = ResultsAnalysis()
        card_group = analysis.compute(
            experiment=experiment,
            generation_strategy=generation_strategy,
        )

        # Assert: Should not error with scalarized constraints
        self.assertIsNotNone(card_group)
        self.assertEqual(card_group.title, "Results Analysis")
        self.assertGreater(len(card_group.children), 0)

        # Assert: No error cards should be present
        for card in card_group.flatten():
            self.assertNotIsInstance(card, ErrorAnalysisCard)

    @mock_botorch_optimize
    def test_compute_with_status_quo_relativizes(self) -> None:
        # Setup: Create experiment with status quo to enable relativization
        experiment = get_branin_experiment()
        experiment.status_quo = Arm(parameters={"x1": 0.5, "x2": 0.5})

        # Create a batch trial (required for relativization)
        trial = experiment.new_batch_trial()
        trial.add_arms_and_weights(
            arms=[
                Arm(parameters={"x1": 0.3, "x2": 0.7}),
                experiment.status_quo,
            ]
        )
        trial.mark_running(no_runner_required=True)

        # Attach data
        data = get_data(
            metric_name="branin",
            trial_index=trial.index,
            num_non_sq_arms=1,
            include_sq=True,
        )
        experiment.attach_data(data)
        trial.mark_completed()

        # Create generation strategy
        generation_strategy = get_default_generation_strategy_at_MBM_node(
            experiment=experiment
        )

        # Execute: Compute ResultsAnalysis
        analysis = ResultsAnalysis()
        card_group = analysis.compute(
            experiment=experiment,
            generation_strategy=generation_strategy,
        )

        # Assert: Analysis should complete without errors
        self.assertIsNotNone(card_group)
        for card in card_group.flatten():
            self.assertNotIsInstance(card, ErrorAnalysisCard)

    def test_compute_without_optimization_config(self) -> None:
        # Setup: Create experiment without optimization config
        experiment = get_branin_experiment()
        metrics = [
            m.clone()
            for m in none_throws(experiment.optimization_config).metrics.values()
        ]
        experiment._optimization_config = None
        experiment.add_tracking_metrics(metrics)

        trial = experiment.new_trial()
        trial.add_arm(
            Arm(parameters={"x1": 0.3, "x2": 0.7}),
        )
        trial.mark_running(no_runner_required=True)

        # Attach data
        data = get_data(metric_name="branin", trial_index=trial.index)
        experiment.attach_data(data)
        trial.mark_completed()

        generation_strategy = get_default_generation_strategy_at_MBM_node(
            experiment=experiment
        )

        # Execute: Compute ResultsAnalysis (without generation strategy since no opt
        # config)
        analysis = ResultsAnalysis()
        card_group = analysis.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )

        # Assert: Should complete without errors even without optimization config
        self.assertIsNotNone(card_group)
        self.assertGreater(len(card_group.children), 0)

    def test_compute_with_bandit_experiment(self) -> None:
        # Setup: Create a bandit experiment
        experiment = Experiment(
            name="bandit_test",
            search_space=SearchSpace(
                parameters=[
                    ChoiceParameter(
                        name="x1",
                        parameter_type=ParameterType.FLOAT,
                        values=[0.0, 0.5, 1.0],
                    ),
                ]
            ),
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric(name="foo"), minimize=True)
            ),
        )

        # Create multi-arm trial
        trial = experiment.new_batch_trial()
        trial.add_arms_and_weights(
            arms=[
                Arm(parameters={"x1": 0.0}),
                Arm(parameters={"x1": 0.5}),
                Arm(parameters={"x1": 1.0}),
            ]
        )
        trial.mark_running(no_runner_required=True)

        # Attach data
        data_rows = []
        for arm in trial.arms:
            data_rows.append(
                {
                    "trial_index": trial.index,
                    "arm_name": arm.name,
                    "metric_name": "foo",
                    "metric_signature": "foo",
                    "mean": float(arm.parameters["x1"]),
                    "sem": 0.1,
                }
            )
        experiment.attach_data(Data(df=pd.DataFrame(data_rows)))
        trial.mark_completed()

        # Create bandit generation strategy

        factorial_node = GenerationNode(
            name="FACTORIAL",
            generator_specs=[GeneratorSpec(generator_enum=Generators.FACTORIAL)],
            transition_criteria=[
                MinTrials(
                    threshold=1, transition_to="EMPIRICAL_BAYES_THOMPSON_SAMPLING"
                )
            ],
        )

        eb_ts_node = GenerationNode(
            name="EMPIRICAL_BAYES_THOMPSON_SAMPLING",
            generator_specs=[
                GeneratorSpec(generator_enum=Generators.EMPIRICAL_BAYES_THOMPSON)
            ],
            transition_criteria=None,
        )

        bandit_gs = GenerationStrategy(
            name=Keys.FACTORIAL_PLUS_EMPIRICAL_BAYES_THOMPSON_SAMPLING,
            nodes=[factorial_node, eb_ts_node],
        )
        bandit_gs._curr = bandit_gs._nodes[1]

        # Execute: Compute ResultsAnalysis
        analysis = ResultsAnalysis()
        card_group = analysis.compute(
            experiment=experiment,
            generation_strategy=bandit_gs,
        )

        # Assert: Should include BanditRollout card
        child_names = [child.name for child in card_group.flatten()]
        self.assertIn("BanditRollout", child_names)

    @mock_botorch_optimize
    def test_online_experiments(self) -> None:
        # Test ResultsAnalysis can be computed for a variety of experiments which
        # resemble those we see in an online setting (with status
        # quo and relativization).
        analysis = ResultsAnalysis()

        for experiment in get_online_experiments():
            generation_strategy = get_default_generation_strategy_at_MBM_node(
                experiment=experiment
            )
            card_group = analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )

            # Assert: No error cards should be present
            total_errors = sum(
                isinstance(card, ErrorAnalysisCard) for card in card_group.flatten()
            )
            self.assertEqual(total_errors, 0)

            # Assert: Should contain expected analysis types
            self.assertIsNotNone(card_group)
            self.assertGreater(len(card_group.children), 0)

    @mock_botorch_optimize
    def test_offline_experiments(self) -> None:
        # Test ResultsAnalysis can be computed for a variety of experiments which
        # resemble those we see in an offline setting (without
        # status quo, no relativization).
        analysis = ResultsAnalysis()

        for experiment in get_offline_experiments():
            generation_strategy = get_default_generation_strategy_at_MBM_node(
                experiment=experiment
            )
            card_group = analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )

            # Assert: No error cards should be present
            for card in card_group.flatten():
                self.assertNotIsInstance(card, ErrorAnalysisCard)

            # Assert: Should contain expected analysis types
            self.assertIsNotNone(card_group)
            self.assertGreater(len(card_group.children), 0)


class TestArmEffectsPair(TestCase):
    @mock_botorch_optimize
    def test_compute(self) -> None:
        # Setup: Create experiment with data and optimization config
        experiment = get_branin_experiment()
        experiment.optimization_config = get_branin_optimization_config()

        trial = experiment.new_batch_trial()
        trial.add_arm(Arm(parameters={"x1": 0.5, "x2": 0.5}))
        trial.mark_running(no_runner_required=True)

        data = get_data(metric_name="branin", trial_index=trial.index)
        experiment.attach_data(data)
        trial.mark_completed()

        generation_strategy = get_default_generation_strategy_at_MBM_node(
            experiment=experiment
        )

        with self.subTest("valid_experiment"):
            analysis = ArmEffectsPair(metric_names=["branin"])
            card_group = analysis.compute(
                experiment=experiment,
                generation_strategy=generation_strategy,
            )

            self.assertIsNotNone(card_group)
            self.assertGreater(len(card_group.children), 0)

            # Each child should be a pair (predicted and raw)
            for child in card_group.children:
                self.assertEqual(
                    len(assert_is_instance(child, AnalysisCardGroup).children),
                    2,
                    "Each pair should have 2 children",
                )

        with self.subTest("requires_experiment"):
            analysis = ArmEffectsPair(metric_names=["test_metric"])
            with self.assertRaisesRegex(UserInputError, "requires an Experiment"):
                analysis.compute()

    @mock_botorch_optimize
    def test_compute_with_status_quo(self) -> None:
        # Setup: Create experiment with status quo for relativization
        experiment = get_branin_experiment()
        experiment.status_quo = Arm(parameters={"x1": 0.5, "x2": 0.5})

        trial = experiment.new_batch_trial()
        trial.add_arms_and_weights(
            arms=[
                Arm(parameters={"x1": 0.3, "x2": 0.7}),
                experiment.status_quo,
            ]
        )
        trial.mark_running(no_runner_required=True)

        data = get_data(
            metric_name="branin",
            trial_index=trial.index,
            num_non_sq_arms=1,
            include_sq=True,
        )
        experiment.attach_data(data)
        trial.mark_completed()

        generation_strategy = get_default_generation_strategy_at_MBM_node(
            experiment=experiment
        )

        with self.subTest("relativization"):
            analysis = ArmEffectsPair(metric_names=["branin"], relativize=True)
            card_group = analysis.compute(
                experiment=experiment,
                generation_strategy=generation_strategy,
            )

            self.assertIsNotNone(card_group)
            for card in card_group.flatten():
                self.assertNotIsInstance(card, ErrorAnalysisCard)

        with self.subTest("trial_index_filter"):
            # Add a second trial
            trial2 = experiment.new_batch_trial()
            trial2.add_arm(Arm(parameters={"x1": 0.6, "x2": 0.4}))
            trial2.mark_running(no_runner_required=True)
            data2 = get_data(metric_name="branin", trial_index=trial2.index)
            experiment.attach_data(data2)
            trial2.mark_completed()

            analysis = ArmEffectsPair(metric_names=["branin"], trial_index=0)
            card_group = analysis.compute(
                experiment=experiment,
                generation_strategy=generation_strategy,
            )

            self.assertIsNotNone(card_group)
            self.assertGreater(len(card_group.children), 0)

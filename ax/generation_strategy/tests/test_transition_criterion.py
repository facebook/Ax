# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from logging import Logger
from unittest.mock import MagicMock

import pandas as pd
from ax.adapter.registry import Generators
from ax.core.arm import Arm
from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
from ax.core.data import Data
from ax.core.derived_metric import DerivedMetric
from ax.core.experiment import Experiment
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import DataRequiredError, UserInputError
from ax.exceptions.generation_strategy import MaxParallelismReachedException
from ax.generation_strategy.generation_strategy import (
    GenerationNode,
    GenerationStep,
    GenerationStrategy,
)
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import (
    AutoTransitionAfterGen,
    AuxiliaryExperimentCheck,
    FreshLILOLabelCheck,
    IsSingleObjective,
    MaxGenerationParallelism,
    MaxTrialsAwaitingData,
    MinTrials,
)
from ax.utils.common.constants import Keys
from ax.utils.common.hash_utils import compute_lilo_input_hash
from ax.utils.common.logger import get_logger
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_experiment,
    get_branin_multi_objective_optimization_config,
    get_experiment,
)

logger: Logger = get_logger(__name__)


def _mock_node(trials_from_node: set[int]) -> MagicMock:
    """Create a mock GenerationNode with a specified trials_from_node set."""
    node = MagicMock()
    node.trials_from_node = trials_from_node
    return node


class TestTransitionCriterion(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sobol_generator_spec = GeneratorSpec(
            generator_enum=Generators.SOBOL,
            generator_kwargs={"init_position": 3},
            generator_gen_kwargs={"some_gen_kwarg": "some_value"},
        )
        self.branin_experiment = get_branin_experiment()

    def test_aux_experiment_check(self) -> None:
        # Test incorrect instantiation
        with self.assertRaisesRegex(UserInputError, r"cannot have both .* None"):
            AuxiliaryExperimentCheck(
                transition_to="some_node",
                auxiliary_experiment_purposes_to_include=None,
                auxiliary_experiment_purposes_to_exclude=None,
            )

    def test_aux_experiment_check_in_gs(self) -> None:
        experiment = self.branin_experiment
        gs = GenerationStrategy(
            name="test",
            nodes=[
                GenerationNode(
                    name="sobol_1",
                    generator_specs=[self.sobol_generator_spec],
                    transition_criteria=[
                        AuxiliaryExperimentCheck(
                            transition_to="sobol_2",
                            auxiliary_experiment_purposes_to_include=[
                                AuxiliaryExperimentPurpose.PE_EXPERIMENT
                            ],
                        )
                    ],
                ),
                GenerationNode(
                    name="sobol_2",
                    generator_specs=[self.sobol_generator_spec],
                    transition_criteria=[
                        AuxiliaryExperimentCheck(
                            transition_to="sobol_1",
                            auxiliary_experiment_purposes_to_exclude=[
                                AuxiliaryExperimentPurpose.PE_EXPERIMENT
                            ],
                        )
                    ],
                ),
            ],
        )
        gs._experiment = experiment
        aux_exp = AuxiliaryExperiment(experiment=get_experiment())
        # Initial check
        self.assertEqual(gs.current_node_name, "sobol_1")

        # Do not transition because no aux experiment
        grs = gs.gen(experiment=experiment, n=5)[0]
        self.assertEqual(gs.current_node_name, "sobol_1")
        self.assertEqual(len(grs), 1)
        self.assertEqual(len(grs[0].arms), 5)

        # Transition because auxiliary_experiment_purposes_to_include is met
        experiment.auxiliary_experiments_by_purpose = {
            AuxiliaryExperimentPurpose.PE_EXPERIMENT: [aux_exp],
        }
        grs = gs.gen(experiment=experiment, n=5)[0]
        self.assertEqual(gs.current_node_name, "sobol_2")
        self.assertEqual(len(grs), 1)
        self.assertEqual(len(grs[0].arms), 5)

        # Not having the aux exp purpose at all should be the same and remain in sobol_1
        experiment.auxiliary_experiments_by_purpose = {}
        grs = gs.gen(experiment=experiment, n=5)[0]
        self.assertEqual(gs.current_node_name, "sobol_1")
        self.assertEqual(len(grs), 1)
        self.assertEqual(len(grs[0].arms), 5)

        # Having multiple aux exp should be fine and we move back to sobol_2
        experiment.auxiliary_experiments_by_purpose = {
            AuxiliaryExperimentPurpose.PE_EXPERIMENT: [aux_exp, aux_exp],
        }
        grs = gs.gen(experiment=experiment, n=5)[0]
        self.assertEqual(gs.current_node_name, "sobol_2")
        self.assertEqual(len(grs), 1)
        self.assertEqual(len(grs[0].arms), 5)

        # Empty the aux exp list is the same as not having the aux exp purpose
        # and should move back to sobol_1
        experiment.auxiliary_experiments_by_purpose = {
            AuxiliaryExperimentPurpose.PE_EXPERIMENT: [],
        }
        grs = gs.gen(experiment=experiment, n=5)[0]
        self.assertEqual(gs.current_node_name, "sobol_1")
        self.assertEqual(len(grs), 1)
        self.assertEqual(len(grs[0].arms), 5)

    def test_default_step_criterion_setup(self) -> None:
        """This test ensures that the default completion criterion for GenerationSteps
        is set as expected.

        The default completion criterion is to create two TransitionCriterion, one
        of type `MaximumTrialsInStatus` and one of type `MinTrials`.
        These are constructed via the inputs of `num_trials`, `enforce_num_trials`,
        and `minimum_trials_observed` on the GenerationStep.
        """
        experiment = get_experiment()
        gs = GenerationStrategy(
            name="SOBOL+MBM::default",
            steps=[
                GenerationStep(
                    generator=Generators.SOBOL,
                    num_trials=3,
                ),
                GenerationStep(
                    generator=Generators.BOTORCH_MODULAR,
                    num_trials=4,
                    max_parallelism=1,
                    min_trials_observed=2,
                    enforce_num_trials=False,
                ),
                GenerationStep(
                    generator=Generators.BOTORCH_MODULAR,
                    num_trials=-1,
                ),
            ],
        )
        gs.experiment = experiment

        step_0_expected_transition_criteria = [
            MinTrials(
                threshold=3,
                transition_to="GenerationStep_1_BoTorch",
                only_in_statuses=None,
                not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
            ),
        ]
        step_1_expected_transition_criteria = [
            MinTrials(
                threshold=4,
                transition_to="GenerationStep_2_BoTorch",
                only_in_statuses=None,
                not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
            ),
            MinTrials(
                only_in_statuses=[TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED],
                threshold=2,
                transition_to="GenerationStep_2_BoTorch",
                count_only_trials_with_data=True,
            ),
        ]
        step_1_expected_pausing_criteria = [
            MaxGenerationParallelism(
                threshold=1,
                only_in_statuses=[TrialStatus.RUNNING],
            ),
        ]
        step_2_expected_transition_criteria = []
        self.assertEqual(
            gs._nodes[0].transition_criteria, step_0_expected_transition_criteria
        )
        self.assertEqual(
            gs._nodes[1].transition_criteria, step_1_expected_transition_criteria
        )
        self.assertEqual(
            gs._nodes[1].pausing_criteria, step_1_expected_pausing_criteria
        )
        self.assertEqual(
            gs._nodes[2].transition_criteria, step_2_expected_transition_criteria
        )

    def test_min_trials_is_met(self) -> None:
        experiment = self.branin_experiment
        gs = GenerationStrategy(
            name="SOBOL::default",
            steps=[
                GenerationStep(
                    generator=Generators.SOBOL,
                    num_trials=4,
                    min_trials_observed=2,
                    enforce_num_trials=True,
                ),
                GenerationStep(
                    Generators.SOBOL,
                    num_trials=-1,
                    max_parallelism=1,
                ),
            ],
        )
        gs.experiment = experiment

        # Need to add trials to test the transition criteria `is_met` method
        for _i in range(4):
            experiment.new_trial(
                generator_run=gs.gen_single_trial(experiment=experiment)
            )
        node_0_trials = gs._nodes[0].trials_from_node
        node_1_trials = gs._nodes[1].trials_from_node

        self.assertEqual(len(node_0_trials), 4)
        self.assertEqual(len(node_1_trials), 0)

        # MinTrials is met should not pass yet, because no trials
        # are marked completed
        self.assertFalse(
            gs._nodes[0]
            .transition_criteria[1]
            .is_met(experiment=experiment, curr_node=gs._nodes[0])
        )

        # Should pass after two trials are marked completed AND have data
        for idx, trial in experiment.trials.items():
            trial.mark_running(no_runner_required=True).mark_completed()
            if idx == 1:
                break
        # With count_only_trials_with_data=True (now the default for
        # min_trials_observed), this should still be False without data.
        self.assertFalse(
            gs._nodes[0]
            .transition_criteria[1]
            .is_met(experiment=experiment, curr_node=gs._nodes[0])
        )
        # Attach data for both completed trials
        experiment.attach_data(
            get_branin_data(
                trials=[experiment.trials[0], experiment.trials[1]],
                metrics=["branin"],
            )
        )
        self.assertTrue(
            gs._nodes[0]
            .transition_criteria[1]
            .is_met(experiment=experiment, curr_node=gs._nodes[0])
        )

        # Check mixed status MinTrials
        min_criterion = MinTrials(
            threshold=3,
            transition_to="next_node",  # placeholder for testing, transition not used
            only_in_statuses=[TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED],
        )
        self.assertFalse(
            min_criterion.is_met(experiment=experiment, curr_node=gs._nodes[0])
        )
        for idx, trial in experiment.trials.items():
            if idx == 2:
                trial._status = TrialStatus.EARLY_STOPPED
        self.assertTrue(
            min_criterion.is_met(experiment=experiment, curr_node=gs._nodes[0])
        )

    def test_min_trials_count_only_with_data(self) -> None:
        """Test that count_only_trials_with_data excludes COMPLETED trials
        that are missing required optimization config metrics."""
        experiment = self.branin_experiment
        gs = GenerationStrategy(
            name="SOBOL::default",
            steps=[
                GenerationStep(
                    generator=Generators.SOBOL,
                    num_trials=4,
                    min_trials_observed=2,
                    enforce_num_trials=True,
                ),
                GenerationStep(
                    Generators.SOBOL,
                    num_trials=-1,
                    max_parallelism=1,
                ),
            ],
        )
        gs.experiment = experiment

        for _i in range(4):
            experiment.new_trial(
                generator_run=gs.gen_single_trial(experiment=experiment)
            )

        # Create a MinTrials criterion with count_only_trials_with_data=True
        min_criterion = MinTrials(
            threshold=2,
            transition_to="GenerationStep_1",
            only_in_statuses=[TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED],
            count_only_trials_with_data=True,
        )

        # Mark all 4 trials as completed
        for trial in experiment.trials.values():
            trial.mark_running(no_runner_required=True).mark_completed()

        # Even though 4 trials are COMPLETED, none have data, so the
        # criterion should not be met.
        self.assertFalse(
            min_criterion.is_met(experiment=experiment, curr_node=gs._nodes[0])
        )

        # Attach data for "branin" (the opt config metric) to 1 trial only
        experiment.attach_data(
            get_branin_data(trials=[experiment.trials[0]], metrics=["branin"])
        )
        # Still not met — only 1 trial has data, need 2
        self.assertFalse(
            min_criterion.is_met(experiment=experiment, curr_node=gs._nodes[0])
        )

        # Attach data for a NON-opt-config metric to trial 1 (missing "branin")
        experiment.attach_data(
            Data(
                df=pd.DataFrame(
                    [
                        {
                            "trial_index": 1,
                            "arm_name": experiment.trials[1].arm.name,
                            "metric_name": "not_branin",
                            "mean": 1.0,
                            "sem": 0.0,
                            "metric_signature": "not_branin",
                        }
                    ]
                )
            )
        )
        # Still not met — trial 1 has data but not for "branin"
        self.assertFalse(
            min_criterion.is_met(experiment=experiment, curr_node=gs._nodes[0])
        )

        # Attach "branin" data to trial 1 too
        experiment.attach_data(
            get_branin_data(trials=[experiment.trials[1]], metrics=["branin"])
        )
        # Now 2 trials have "branin" data — criterion should be met
        self.assertTrue(
            min_criterion.is_met(experiment=experiment, curr_node=gs._nodes[0])
        )

    def test_auto_transition(self) -> None:
        """Very simple test to validate AutoTransitionAfterGen"""
        experiment = self.branin_experiment
        gs = GenerationStrategy(
            name="test",
            nodes=[
                GenerationNode(
                    name="sobol_1",
                    generator_specs=[self.sobol_generator_spec],
                    transition_criteria=[
                        AutoTransitionAfterGen(transition_to="sobol_2")
                    ],
                ),
                GenerationNode(
                    name="sobol_2", generator_specs=[self.sobol_generator_spec]
                ),
            ],
        )
        gs.experiment = experiment
        self.assertEqual(gs.current_node_name, "sobol_1")
        gs.gen(experiment=experiment)
        gs.gen(experiment=experiment)
        self.assertEqual(gs.current_node_name, "sobol_2")

    def test_auto_with_should_skip_node(self) -> None:
        experiment = self.branin_experiment
        gs = GenerationStrategy(
            name="test",
            nodes=[
                GenerationNode(
                    name="sobol_1",
                    generator_specs=[self.sobol_generator_spec],
                    transition_criteria=[
                        AutoTransitionAfterGen(transition_to="sobol_2")
                    ],
                ),
                GenerationNode(
                    name="sobol_2", generator_specs=[self.sobol_generator_spec]
                ),
            ],
        )
        gs._nodes[0]._should_skip = True
        self.assertTrue(
            gs._nodes[0]
            .transition_criteria[0]
            .is_met(experiment=experiment, curr_node=gs._nodes[0])
        )

    def test_is_single_objective_does_not_transition(self) -> None:
        exp = self.branin_experiment
        exp.optimization_config = get_branin_multi_objective_optimization_config()
        gs = GenerationStrategy(
            name="test",
            nodes=[
                GenerationNode(
                    name="sobol_1",
                    generator_specs=[self.sobol_generator_spec],
                    transition_criteria=[IsSingleObjective(transition_to="sobol_2")],
                ),
                GenerationNode(
                    name="sobol_2", generator_specs=[self.sobol_generator_spec]
                ),
            ],
        )
        self.assertEqual(gs.current_node_name, "sobol_1")
        # Should not transition because this is a MOO experiment
        gr = gs.gen_single_trial(experiment=exp)
        gr2 = gs.gen_single_trial(experiment=exp)
        self.assertEqual(gr._generation_node_name, "sobol_1")
        self.assertEqual(gr2._generation_node_name, "sobol_1")
        self.assertEqual(gs.current_node_name, "sobol_1")

    def test_is_single_objective_transitions(self) -> None:
        exp = self.branin_experiment
        gs = GenerationStrategy(
            name="test",
            nodes=[
                GenerationNode(
                    name="sobol_1",
                    generator_specs=[self.sobol_generator_spec],
                    transition_criteria=[
                        IsSingleObjective(transition_to="sobol_2"),
                        AutoTransitionAfterGen(
                            transition_to="sobol_2", continue_trial_generation=False
                        ),
                    ],
                ),
                GenerationNode(
                    name="sobol_2", generator_specs=[self.sobol_generator_spec]
                ),
            ],
        )
        self.assertEqual(gs.current_node_name, "sobol_1")
        gr = gs.gen_single_trial(experiment=exp)
        gr2 = gs.gen_single_trial(experiment=exp)
        # First generation should use sobol_1, then transition to sobol_2
        self.assertEqual(gr._generation_node_name, "sobol_1")
        self.assertEqual(gr2._generation_node_name, "sobol_2")
        self.assertEqual(gs.current_node_name, "sobol_2")

    def test_trials_from_node_empty(self) -> None:
        """Tests MinTrials defaults to experiment
        level trials when trials_from_node is None.
        """
        experiment = get_experiment()
        gs = GenerationStrategy(
            name="SOBOL::default",
            steps=[
                GenerationStep(
                    generator=Generators.SOBOL,
                    num_trials=4,
                    min_trials_observed=2,
                    enforce_num_trials=True,
                ),
            ],
        )
        gs.experiment = experiment
        max_criterion_with_status = MinTrials(
            threshold=2,
            transition_to="next_node",
            only_in_statuses=[TrialStatus.COMPLETED],
        )
        max_criterion = MinTrials(threshold=2, transition_to="next_node")
        self.assertFalse(
            max_criterion.is_met(experiment=experiment, curr_node=gs._nodes[0])
        )

        for _i in range(3):
            experiment.new_trial(gs.gen_single_trial(experiment=experiment))
        self.assertTrue(
            max_criterion.is_met(experiment=experiment, curr_node=gs._nodes[0])
        )

        # Before marking trial status it should be false, until trials are completed
        self.assertFalse(
            max_criterion_with_status.is_met(
                experiment=experiment, curr_node=gs._nodes[0]
            )
        )
        for idx, trial in experiment.trials.items():
            trial._status = TrialStatus.COMPLETED
            if idx == 1:
                break
        self.assertTrue(
            max_criterion_with_status.is_met(
                experiment=experiment, curr_node=gs._nodes[0]
            )
        )

    def test_repr(self) -> None:
        self.maxDiff = None
        min_trials_criterion = MinTrials(
            threshold=5,
            transition_to="GenerationStep_1",
            only_in_statuses=[TrialStatus.COMPLETED],
            not_in_statuses=[TrialStatus.FAILED],
        )
        self.assertEqual(
            str(min_trials_criterion),
            "MinTrials({'threshold': 5, "
            + "'transition_to': 'GenerationStep_1', "
            + "'only_in_statuses': [<enum 'TrialStatus'>.COMPLETED], "
            + "'not_in_statuses': [<enum 'TrialStatus'>.FAILED], "
            + "'use_all_trials_in_exp': False, "
            + "'continue_trial_generation': False, "
            + "'count_only_trials_with_data': False})",
        )
        minimum_trials_in_status_criterion = MinTrials(
            threshold=0,
            transition_to="GenerationStep_2",
            only_in_statuses=[TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED],
            not_in_statuses=[TrialStatus.FAILED],
        )
        self.assertEqual(
            str(minimum_trials_in_status_criterion),
            "MinTrials({'threshold': 0, "
            + "'transition_to': 'GenerationStep_2', "
            + "'only_in_statuses': "
            + "[<enum 'TrialStatus'>.COMPLETED, <enum 'TrialStatus'>.EARLY_STOPPED], "
            + "'not_in_statuses': [<enum 'TrialStatus'>.FAILED], "
            + "'use_all_trials_in_exp': False, "
            + "'continue_trial_generation': False, "
            + "'count_only_trials_with_data': False})",
        )
        max_parallelism = MaxGenerationParallelism(
            only_in_statuses=[TrialStatus.EARLY_STOPPED],
            threshold=3,
            not_in_statuses=[TrialStatus.FAILED],
        )
        self.assertEqual(
            str(max_parallelism),
            "MaxGenerationParallelism({'threshold': 3, "
            + "'only_in_statuses': "
            + "[<enum 'TrialStatus'>.EARLY_STOPPED], "
            + "'not_in_statuses': [<enum 'TrialStatus'>.FAILED], "
            + "'use_all_trials_in_exp': False, "
            + "'count_only_trials_with_data': False})",
        )
        auto_transition = AutoTransitionAfterGen(transition_to="GenerationStep_2")
        self.assertEqual(
            str(auto_transition),
            "AutoTransitionAfterGen({'transition_to': 'GenerationStep_2', "
            + "'continue_trial_generation': True})",
        )


class TestPausingCriterion(TestCase):
    """Tests for PausingCriterion classes."""

    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_branin_experiment()

    def test_max_trials_awaiting_data(self) -> None:
        with self.subTest("default_not_in_statuses"):
            criterion = MaxTrialsAwaitingData(threshold=10)
            self.assertEqual(
                criterion.not_in_statuses,
                [TrialStatus.FAILED, TrialStatus.ABANDONED],
            )

        with self.subTest("block_continued_generation_error"):
            criterion = MaxTrialsAwaitingData(threshold=3)
            with self.assertRaises(DataRequiredError):
                criterion.block_continued_generation_error(
                    node_name="test", experiment=self.experiment, trials_from_node=set()
                )

    def test_max_generation_parallelism_block_error(self) -> None:
        criterion = MaxGenerationParallelism(
            threshold=2, only_in_statuses=[TrialStatus.RUNNING]
        )
        with self.assertRaises(MaxParallelismReachedException):
            criterion.block_continued_generation_error(
                node_name="test",
                experiment=self.experiment,
                trials_from_node={0, 1, 2},
            )

    def test_fresh_lilo_label_check(self) -> None:
        """Verify FreshLILOLabelCheck counts only hash-fresh trials."""
        exp = get_branin_experiment()

        # Register a DerivedMetric with pairwise name.
        pairwise_metric = DerivedMetric(
            name=Keys.PAIRWISE_PREFERENCE_QUERY.value,
            input_metric_names=["branin"],
        )
        exp.add_tracking_metric(pairwise_metric)

        criterion = FreshLILOLabelCheck(
            threshold=2,
            transition_to="next_node",
            only_in_statuses=[TrialStatus.COMPLETED],
        )

        # Helper to create and complete a trial with data.
        def _add_trial(idx: int, exp: Experiment = exp) -> None:
            trial = exp.new_batch_trial()
            trial.add_arm(
                Arm(name=f"{idx}_0", parameters={"x1": float(idx), "x2": 0.0})
            )
            trial.mark_running(no_runner_required=True)
            trial.mark_completed()
            exp.attach_data(
                Data(
                    df=pd.DataFrame(
                        [
                            {
                                "trial_index": idx,
                                "arm_name": f"{idx}_0",
                                "metric_name": "branin",
                                "metric_signature": "branin",
                                "mean": float(idx),
                                "sem": 0.1,
                            }
                        ]
                    )
                )
            )

        # Create 3 trials, stamp first 2 with current hash.
        for i in range(3):
            _add_trial(i)

        current_hash = compute_lilo_input_hash(exp, ["branin"])
        trials_from_node = {0, 1, 2}

        with self.subTest("no_hashes_none_count"):
            # No hash stamps → no trials counted (only LILO trials with
            # a matching hash contribute).
            count = criterion.num_contributing_to_threshold(exp, trials_from_node)
            self.assertEqual(count, 0)

        # Stamp trials 0 and 1 with the current hash.
        exp.trials[0]._properties[Keys.LILO_INPUT_HASH] = current_hash
        exp.trials[1]._properties[Keys.LILO_INPUT_HASH] = current_hash

        with self.subTest("fresh_hashes_count"):
            count = criterion.num_contributing_to_threshold(exp, trials_from_node)
            # Trials 0, 1 (fresh hash). Trial 2 (no hash → excluded).
            self.assertEqual(count, 2)

        # Make trial 1 stale.
        exp.trials[1]._properties[Keys.LILO_INPUT_HASH] = "stale_hash"

        with self.subTest("stale_hash_excluded"):
            count = criterion.num_contributing_to_threshold(exp, trials_from_node)
            # Trial 0 (fresh). Trial 1 (stale) and trial 2 (no hash) excluded.
            self.assertEqual(count, 1)
            self.assertFalse(criterion.is_met(exp, _mock_node(trials_from_node)))

        # Make trial 0 stale too.
        exp.trials[0]._properties[Keys.LILO_INPUT_HASH] = "another_stale"

        with self.subTest("not_enough_fresh"):
            count = criterion.num_contributing_to_threshold(exp, trials_from_node)
            # All stamped trials are stale, trial 2 has no hash → 0.
            self.assertEqual(count, 0)
            self.assertFalse(criterion.is_met(exp, _mock_node(trials_from_node)))

        with self.subTest("data_change_invalidates"):
            # Add new data — changes the current hash, making ALL stamped
            # trials stale.
            _add_trial(3)
            trials_from_node.add(3)
            count = criterion.num_contributing_to_threshold(exp, trials_from_node)
            # Trials 0, 1 stale. Trials 2, 3 have no hash → excluded.
            self.assertEqual(count, 0)

    def test_fresh_lilo_label_check_require_sufficient(self) -> None:
        """Verify require_sufficient flag controls is_met direction."""
        exp = get_branin_experiment()

        pairwise_metric = DerivedMetric(
            name=Keys.PAIRWISE_PREFERENCE_QUERY.value,
            input_metric_names=["branin"],
        )
        exp.add_tracking_metric(pairwise_metric)

        # Create 2 completed trials with data.
        for i in range(2):
            trial = exp.new_batch_trial()
            trial.add_arm(Arm(name=f"{i}_0", parameters={"x1": float(i), "x2": 0.0}))
            trial.mark_running(no_runner_required=True)
            trial.mark_completed()
            exp.attach_data(
                Data(
                    df=pd.DataFrame(
                        [
                            {
                                "trial_index": i,
                                "arm_name": f"{i}_0",
                                "metric_name": "branin",
                                "metric_signature": "branin",
                                "mean": float(i),
                                "sem": 0.1,
                            }
                        ]
                    )
                )
            )

        current_hash = compute_lilo_input_hash(exp, ["branin"])
        # Stamp both trials as fresh.
        exp.trials[0]._properties[Keys.LILO_INPUT_HASH] = current_hash
        exp.trials[1]._properties[Keys.LILO_INPUT_HASH] = current_hash
        trials_from_node = {0, 1}

        sufficient = FreshLILOLabelCheck(
            threshold=2,
            transition_to="MBG",
            require_sufficient=True,
            only_in_statuses=[TrialStatus.COMPLETED],
        )
        insufficient = FreshLILOLabelCheck(
            threshold=2,
            transition_to="LILO",
            require_sufficient=False,
            only_in_statuses=[TrialStatus.COMPLETED],
        )

        with self.subTest("sufficient_met_when_enough_fresh"):
            # 2 fresh >= threshold 2 → require_sufficient=True is met.
            self.assertTrue(sufficient.is_met(exp, _mock_node(trials_from_node)))

        with self.subTest("insufficient_not_met_when_enough_fresh"):
            # 2 fresh >= threshold 2 → require_sufficient=False is NOT met.
            self.assertFalse(insufficient.is_met(exp, _mock_node(trials_from_node)))

        # Make trial 0 stale → only 1 fresh trial.
        exp.trials[0]._properties[Keys.LILO_INPUT_HASH] = "stale"

        with self.subTest("sufficient_not_met_when_stale"):
            # 1 fresh < threshold 2 → require_sufficient=True is NOT met.
            self.assertFalse(sufficient.is_met(exp, _mock_node(trials_from_node)))

        with self.subTest("insufficient_met_when_stale"):
            # 1 fresh < threshold 2 → require_sufficient=False IS met.
            self.assertTrue(insufficient.is_met(exp, _mock_node(trials_from_node)))

    def test_fresh_lilo_label_check_non_lilo_fallback(self) -> None:
        """Non-LILO experiment: require_sufficient=True always met,
        require_sufficient=False never met."""
        exp = get_branin_experiment()
        # No pairwise DerivedMetric registered — non-LILO experiment.
        trials_from_node: set[int] = set()

        sufficient = FreshLILOLabelCheck(
            threshold=32,
            transition_to="MBG",
            require_sufficient=True,
        )
        insufficient = FreshLILOLabelCheck(
            threshold=32,
            transition_to="LILO",
            require_sufficient=False,
        )

        with self.subTest("non_lilo_sufficient_always_met"):
            self.assertTrue(sufficient.is_met(exp, _mock_node(trials_from_node)))

        with self.subTest("non_lilo_insufficient_never_met"):
            self.assertFalse(insufficient.is_met(exp, _mock_node(trials_from_node)))

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from logging import Logger

from ax.adapter.registry import Generators
from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
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
    IsSingleObjective,
    MaxGenerationParallelism,
    MaxTrialsAwaitingData,
    MinTrials,
)
from ax.utils.common.logger import get_logger
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_multi_objective_optimization_config,
    get_experiment,
)

logger: Logger = get_logger(__name__)


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

        # Should pass after two trials are marked completed
        for idx, trial in experiment.trials.items():
            trial.mark_running(no_runner_required=True).mark_completed()
            if idx == 1:
                break
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

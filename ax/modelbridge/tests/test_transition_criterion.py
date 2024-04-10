# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from logging import Logger
from unittest.mock import patch

import pandas as pd
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.modelbridge.transition_criterion import (
    MaxGenerationParallelism,
    MaxTrials,
    MinimumPreferenceOccurances,
    MinimumTrialsInStatus,
    MinTrials,
    TransitionCriterion,
)
from ax.utils.common.logger import get_logger
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment, get_experiment

logger: Logger = get_logger(__name__)


class TestTransitionCriterion(TestCase):
    def test_minimum_preference_criterion(self) -> None:
        """Tests the minimum preference criterion subcalss of TransitionCriterion."""
        criterion = MinimumPreferenceOccurances(metric_name="m1", threshold=3)
        experiment = get_experiment()
        generation_strategy = GenerationStrategy(
            name="SOBOL::default",
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=-1,
                    completion_criteria=[criterion],
                ),
                GenerationStep(
                    model=Models.GPEI,
                    num_trials=-1,
                    max_parallelism=1,
                ),
            ],
        )
        generation_strategy.experiment = experiment

        # Has not seen enough of each preference
        self.assertFalse(
            generation_strategy._maybe_move_to_next_step(
                raise_data_required_error=False
            )
        )

        data = Data(
            df=pd.DataFrame(
                {
                    "trial_index": range(6),
                    "arm_name": [f"{i}_0" for i in range(6)],
                    "metric_name": ["m1" for _ in range(6)],
                    "mean": [0, 0, 0, 1, 1, 1],
                    "sem": [0 for _ in range(6)],
                }
            )
        )
        with patch.object(experiment, "fetch_data", return_value=data):
            # We have seen three "yes" and three "no"
            self.assertTrue(
                generation_strategy._maybe_move_to_next_step(
                    raise_data_required_error=False
                )
            )
            self.assertEqual(
                generation_strategy._curr.model_spec_to_gen_from.model_enum, Models.GPEI
            )

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
            name="SOBOL+GPEI::default",
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=3,
                ),
                GenerationStep(
                    model=Models.GPEI,
                    num_trials=4,
                    max_parallelism=1,
                    min_trials_observed=2,
                    enforce_num_trials=False,
                ),
                GenerationStep(
                    model=Models.GPEI,
                    num_trials=-1,
                ),
            ],
        )
        gs.experiment = experiment

        step_0_expected_transition_criteria = [
            MaxTrials(
                threshold=3,
                block_gen_if_met=True,
                transition_to="GenerationStep_1",
                only_in_statuses=None,
                not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
            ),
        ]
        step_1_expected_transition_criteria = [
            MaxTrials(
                threshold=4,
                block_gen_if_met=False,
                transition_to="GenerationStep_2",
                only_in_statuses=None,
                not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
            ),
            MinTrials(
                only_in_statuses=[TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED],
                threshold=2,
                transition_to="GenerationStep_2",
            ),
            MaxGenerationParallelism(
                threshold=1,
                only_in_statuses=[TrialStatus.RUNNING],
                block_gen_if_met=True,
                block_transition_if_unmet=False,
            ),
        ]
        step_2_expected_transition_criteria = []
        self.assertEqual(
            gs._steps[0].transition_criteria, step_0_expected_transition_criteria
        )
        self.assertEqual(
            gs._steps[1].transition_criteria, step_1_expected_transition_criteria
        )
        self.assertEqual(
            gs._steps[2].transition_criteria, step_2_expected_transition_criteria
        )

    def test_min_trials_is_met(self) -> None:
        """Test that the is_met method in  MinTrials works"""
        experiment = get_branin_experiment()
        gs = GenerationStrategy(
            name="SOBOL::default",
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=4,
                    min_trials_observed=2,
                    enforce_num_trials=True,
                ),
                GenerationStep(
                    Models.SOBOL,
                    num_trials=-1,
                    max_parallelism=1,
                ),
            ],
        )
        gs.experiment = experiment

        # Need to add trials to test the transition criteria `is_met` method
        for _i in range(4):
            experiment.new_trial(gs.gen(experiment=experiment))

        # TODO: @mgarrard More comphrensive test of trials_from_node
        node_0_trials = gs._steps[0].trials_from_node
        node_1_trials = gs._steps[1].trials_from_node

        self.assertEqual(len(node_0_trials), 4)
        self.assertEqual(len(node_1_trials), 0)

        # MinTrials is met should not pass yet, becasue no trials
        # are marked completed
        self.assertFalse(
            gs._steps[0]
            .transition_criteria[1]
            .is_met(experiment, gs._steps[0].trials_from_node)
        )

        # Should pass after two trials are marked completed
        for idx, trial in experiment.trials.items():
            trial.mark_running(no_runner_required=True).mark_completed()
            if idx == 1:
                break
        self.assertTrue(
            gs._steps[0]
            .transition_criteria[1]
            .is_met(experiment, gs._steps[0].trials_from_node)
        )

        # Check mixed status MinTrials
        min_criterion = MinTrials(
            threshold=3,
            only_in_statuses=[TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED],
        )
        self.assertFalse(
            min_criterion.is_met(experiment, gs._steps[0].trials_from_node)
        )
        for idx, trial in experiment.trials.items():
            if idx == 2:
                trial._status = TrialStatus.EARLY_STOPPED
        self.assertTrue(min_criterion.is_met(experiment, gs._steps[0].trials_from_node))

    def test_max_trials_is_met(self) -> None:
        """Test that the is_met method in MaxTrials works"""
        experiment = get_branin_experiment()
        gs = GenerationStrategy(
            name="SOBOL::default",
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=4,
                    min_trials_observed=0,
                    enforce_num_trials=True,
                ),
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=4,
                    min_trials_observed=0,
                    enforce_num_trials=False,
                ),
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=-1,
                    max_parallelism=1,
                ),
            ],
        )
        gs.experiment = experiment

        # No trials yet, first step should fail
        self.assertFalse(
            gs._steps[0]
            .transition_criteria[0]
            .is_met(
                experiment=experiment,
                trials_from_node=gs._steps[0].trials_from_node,
            )
        )

        # After adding trials, should pass
        for _i in range(4):
            experiment.new_trial(gs.gen(experiment=experiment))
        self.assertTrue(
            gs._steps[0]
            .transition_criteria[0]
            .is_met(
                experiment=experiment,
                trials_from_node=gs._steps[0].trials_from_node,
            )
        )

        # Check not in statuses and only in statuses
        max_criterion_not_in_statuses = MaxTrials(
            threshold=2,
            block_gen_if_met=True,
            not_in_statuses=[TrialStatus.COMPLETED],
        )
        max_criterion_only_statuses = MaxTrials(
            threshold=2,
            block_gen_if_met=True,
            only_in_statuses=[TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED],
        )
        # experiment currently has 4 trials, but none of them are completed
        self.assertTrue(
            max_criterion_not_in_statuses.is_met(
                experiment, trials_from_node=gs._steps[0].trials_from_node
            )
        )
        self.assertFalse(
            max_criterion_only_statuses.is_met(
                experiment, trials_from_node=gs._steps[0].trials_from_node
            )
        )

        # set 3 of the 4 trials to status == completed
        for _idx, trial in experiment.trials.items():
            trial._status = TrialStatus.COMPLETED
            if _idx == 2:
                break
        self.assertTrue(
            max_criterion_only_statuses.is_met(
                experiment, trials_from_node=gs._steps[0].trials_from_node
            )
        )
        self.assertFalse(
            max_criterion_not_in_statuses.is_met(
                experiment, trials_from_node=gs._steps[0].trials_from_node
            )
        )

    def test_trials_from_node_none(self) -> None:
        """Tests MinTrials and MaxTrials default to experiment
        level trials when trials_from_node is None.
        """
        experiment = get_experiment()
        gs = GenerationStrategy(
            name="SOBOL::default",
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=4,
                    min_trials_observed=2,
                    enforce_num_trials=True,
                ),
            ],
        )
        max_criterion_with_status = MaxTrials(
            threshold=2,
            block_gen_if_met=True,
            only_in_statuses=[TrialStatus.COMPLETED],
        )
        max_criterion = MaxTrials(threshold=2, block_gen_if_met=True)
        warning_msg = (
            "`trials_from_node` is None, will check threshold on" + " experiment level."
        )

        # no trials so criterion should be false, then add trials to pass criterion
        with self.assertLogs(TransitionCriterion.__module__, logging.WARNING) as logger:
            self.assertFalse(max_criterion.is_met(experiment, trials_from_node=None))
            self.assertTrue(
                any(warning_msg in output for output in logger.output),
                logger.output,
            )
        for _i in range(3):
            experiment.new_trial(gs.gen(experiment=experiment))
        self.assertTrue(max_criterion.is_met(experiment, trials_from_node=None))

        # Before marking trial status it should be false, until trials are completed
        self.assertFalse(
            max_criterion_with_status.is_met(experiment, trials_from_node=None)
        )
        for idx, trial in experiment.trials.items():
            trial._status = TrialStatus.COMPLETED
            if idx == 1:
                break
        self.assertTrue(
            max_criterion_with_status.is_met(experiment, trials_from_node=None)
        )

        # Check MinTrials
        min_criterion = MinTrials(
            threshold=3,
            only_in_statuses=[TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED],
        )
        with self.assertLogs(TransitionCriterion.__module__, logging.WARNING) as logger:
            self.assertFalse(min_criterion.is_met(experiment, trials_from_node=None))
            self.assertTrue(
                any(warning_msg in output for output in logger.output),
                logger.output,
            )
        for _idx, trial in experiment.trials.items():
            trial._status = TrialStatus.COMPLETED
        self.assertTrue(min_criterion.is_met(experiment, trials_from_node=None))

    def test_repr(self) -> None:
        """Tests that the repr string is correctly formatted for all
        TransitionCriterion child classes.
        """
        self.maxDiff = None
        max_trials_criterion = MaxTrials(
            threshold=5,
            block_gen_if_met=True,
            block_transition_if_unmet=False,
            transition_to="GenerationStep_1",
            only_in_statuses=[TrialStatus.COMPLETED],
            not_in_statuses=[TrialStatus.FAILED],
        )
        self.assertEqual(
            str(max_trials_criterion),
            "MaxTrials({'threshold': 5, "
            + "'only_in_statuses': [<enum 'TrialStatus'>.COMPLETED], "
            + "'not_in_statuses': [<enum 'TrialStatus'>.FAILED], "
            + "'transition_to': 'GenerationStep_1', "
            + "'block_transition_if_unmet': False, "
            + "'block_gen_if_met': True, "
            + "'use_all_trials_in_exp': False})",
        )
        minimum_trials_in_status_criterion = MinTrials(
            only_in_statuses=[TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED],
            threshold=0,
            transition_to="GenerationStep_2",
            block_gen_if_met=True,
            block_transition_if_unmet=False,
            not_in_statuses=[TrialStatus.FAILED],
        )
        self.assertEqual(
            str(minimum_trials_in_status_criterion),
            "MinTrials({'threshold': 0, 'only_in_statuses': "
            + "[<enum 'TrialStatus'>.COMPLETED, <enum 'TrialStatus'>.EARLY_STOPPED], "
            + "'not_in_statuses': [<enum 'TrialStatus'>.FAILED], "
            + "'transition_to': 'GenerationStep_2', "
            + "'block_transition_if_unmet': False, "
            + "'block_gen_if_met': True, "
            + "'use_all_trials_in_exp': False})",
        )
        minimum_preference_occurances_criterion = MinimumPreferenceOccurances(
            metric_name="m1", threshold=3
        )
        self.assertEqual(
            str(minimum_preference_occurances_criterion),
            "MinimumPreferenceOccurances({'metric_name': 'm1', 'threshold': 3, "
            + "'transition_to': None, 'block_gen_if_met': False, "
            "'block_transition_if_unmet': True})",
        )
        deprecated_min_trials_criterion = MinimumTrialsInStatus(
            status=TrialStatus.COMPLETED, threshold=3
        )
        self.assertEqual(
            str(deprecated_min_trials_criterion),
            "MinimumTrialsInStatus({"
            + "'status': <enum 'TrialStatus'>.COMPLETED, "
            + "'threshold': 3, "
            + "'transition_to': None})",
        )
        max_parallelism = MaxGenerationParallelism(
            only_in_statuses=[TrialStatus.EARLY_STOPPED],
            threshold=3,
            transition_to="GenerationStep_2",
            block_gen_if_met=True,
            block_transition_if_unmet=False,
            not_in_statuses=[TrialStatus.FAILED],
        )
        self.assertEqual(
            str(max_parallelism),
            "MaxGenerationParallelism({'threshold': 3, 'only_in_statuses': "
            + "[<enum 'TrialStatus'>.EARLY_STOPPED], "
            + "'not_in_statuses': [<enum 'TrialStatus'>.FAILED], "
            + "'transition_to': 'GenerationStep_2', "
            + "'block_transition_if_unmet': False, "
            + "'block_gen_if_met': True, "
            + "'use_all_trials_in_exp': False})",
        )

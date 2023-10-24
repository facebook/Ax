# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pandas as pd
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.modelbridge.transition_criterion import (
    MaxTrials,
    MinimumPreferenceOccurances,
    MinimumTrialsInStatus,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment


class TestTransitionCriterion(TestCase):
    def test_minimum_preference_criterion(self) -> None:
        """Tests the minimum preference criterion subcalss of TransitionCriterion."""
        criterion = MinimumPreferenceOccurances(metric_name="m1", threshold=3)
        experiment = get_experiment()
        generation_strategy = GenerationStrategy(
            name="SOBOL+GPEI::default",
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
            self.assertEqual(generation_strategy._curr.model, Models.GPEI)

    def test_default_step_criterion_setup(self) -> None:
        """This test ensures that the default completion criterion for GenerationSteps
        is set as expected.

        The default completion criterion is to create two TransitionCriterion, one
        of type `MaximumTrialsInStatus` and one of type `MinimumTrialsInStatus`.
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
                    enforce_num_trials=False,
                ),
                GenerationStep(
                    model=Models.GPEI,
                    num_trials=4,
                    max_parallelism=1,
                    min_trials_observed=2,
                ),
                GenerationStep(
                    model=Models.GPEI,
                    num_trials=-1,
                    max_parallelism=1,
                ),
            ],
        )
        gs.experiment = experiment

        step_0_expected_transition_criteria = [
            MaxTrials(threshold=3, enforce=False, transition_to="GenerationStep_1"),
            MinimumTrialsInStatus(
                status=TrialStatus.COMPLETED,
                threshold=0,
                transition_to="GenerationStep_1",
            ),
        ]
        step_1_expected_transition_criteria = [
            MaxTrials(threshold=4, enforce=True, transition_to="GenerationStep_2"),
            MinimumTrialsInStatus(
                status=TrialStatus.COMPLETED,
                threshold=2,
                transition_to="GenerationStep_2",
            ),
        ]
        step_2_expected_transition_criteria = [
            MaxTrials(threshold=-1, enforce=True, transition_to="GenerationStep_3"),
            MinimumTrialsInStatus(
                status=TrialStatus.COMPLETED,
                threshold=0,
                transition_to="GenerationStep_3",
            ),
        ]
        self.assertEqual(
            gs._steps[0].transition_criteria, step_0_expected_transition_criteria
        )
        self.assertEqual(
            gs._steps[1].transition_criteria, step_1_expected_transition_criteria
        )
        self.assertEqual(
            gs._steps[2].transition_criteria, step_2_expected_transition_criteria
        )

        # Check default results for `is_met` call
        self.assertTrue(gs._steps[0].transition_criteria[0].is_met(experiment))
        self.assertTrue(gs._steps[0].transition_criteria[1].is_met(experiment))
        self.assertFalse(gs._steps[1].transition_criteria[0].is_met(experiment))
        self.assertFalse(gs._steps[1].transition_criteria[1].is_met(experiment))

    def test_max_trials_status_arg(self) -> None:
        """Tests the `only_in_status` argument checks the threshold based on the
        number of trials in specified status instead of all trials (which is the
        default behavior).
        """
        experiment = get_experiment()
        criterion = MaxTrials(
            threshold=5, only_in_status=TrialStatus.RUNNING, enforce=True
        )
        self.assertFalse(criterion.is_met(experiment))

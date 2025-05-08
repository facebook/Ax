# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

import pandas as pd
from ax.core.data import Data
from ax.core.trial_status import TrialStatus
from ax.generation_strategy.generation_strategy import (
    GenerationStep,
    GenerationStrategy,
)
from ax.generation_strategy.transition_criterion import (
    MinimumPreferenceOccurances,
    MinTrials,
)
from ax.modelbridge.registry import Generators
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment


class TestAEPsychCriterion(TestCase):
    """
    This test class tests the AEPsych usecase implementation. Previously, AEPsych
    used `CompletionCriterion` to determine when to move to the next generation.
    However, `CompletionCriterion` is deprecated and replaced by `TransitionCriterion`.
    We still want to test the bespoke TransitionCriterion used by AEPsych
    """

    def test_single_criterion(self) -> None:
        criterion = MinimumPreferenceOccurances(metric_name="m1", threshold=3)

        experiment = get_experiment()

        generation_strategy = GenerationStrategy(
            name="SOBOL+MBM::default",
            steps=[
                GenerationStep(
                    model=Generators.SOBOL,
                    num_trials=-1,
                    completion_criteria=[criterion],
                ),
                GenerationStep(
                    model=Generators.BOTORCH_MODULAR,
                    num_trials=-1,
                    max_parallelism=1,
                ),
            ],
        )
        generation_strategy.experiment = experiment

        # Has not seen enough of each preference
        self.assertFalse(
            generation_strategy._maybe_transition_to_next_node(
                raise_data_required_error=False
            )
        )
        # check the transition_to is being set
        self.assertEqual(
            generation_strategy._curr.transition_criteria[0].transition_to,
            "GenerationStep_1",
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
            move_to_next_node = generation_strategy._maybe_transition_to_next_node(
                raise_data_required_error=False
            )
        # We have seen three "yes" and three "no"
        self.assertTrue(move_to_next_node)

        self.assertEqual(
            generation_strategy._curr.model_spec_to_gen_from.model_enum,
            Generators.BOTORCH_MODULAR,
        )

    def test_many_criteria(self) -> None:
        criteria = [
            MinimumPreferenceOccurances(metric_name="m1", threshold=3),
            MinTrials(only_in_statuses=[TrialStatus.COMPLETED], threshold=5),
        ]

        experiment = get_experiment()

        generation_strategy = GenerationStrategy(
            name="SOBOL+MBM::default",
            steps=[
                GenerationStep(
                    model=Generators.SOBOL, num_trials=-1, completion_criteria=criteria
                ),
                GenerationStep(
                    model=Generators.BOTORCH_MODULAR,
                    num_trials=-1,
                    max_parallelism=1,
                ),
            ],
        )
        generation_strategy.experiment = experiment

        # Has not seen enough of each preference
        self.assertFalse(
            generation_strategy._maybe_transition_to_next_node(
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
            move_to_next_node = generation_strategy._maybe_transition_to_next_node(
                raise_data_required_error=False
            )
        # We have seen three "yes" and three "no", but not enough trials are
        # completed
        self.assertFalse(move_to_next_node)

        for _i in range(6):
            experiment.new_trial(generation_strategy.gen(experiment=experiment))
        for trial in experiment.trials.values():
            trial._status = TrialStatus.COMPLETED

        # Enough trials are completed but we have not seen three "yes" and three
        # "no"
        self.assertFalse(
            generation_strategy._maybe_transition_to_next_node(
                raise_data_required_error=False
            )
        )

        with patch.object(experiment, "fetch_data", return_value=data):
            move_to_next_node = generation_strategy._maybe_transition_to_next_node(
                raise_data_required_error=False
            )
        # Enough trials are completed but we have not seen three "yes" and three
        # "no"
        self.assertTrue(move_to_next_node)

        self.assertEqual(
            generation_strategy._curr.model_spec_to_gen_from.model_enum,
            Generators.BOTORCH_MODULAR,
        )

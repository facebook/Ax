# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pandas as pd
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.modelbridge.completion_criterion import (
    MinimumPreferenceOccurances,
    MinimumTrialsInStatus,
)
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment


class TestCompletionCritereon(TestCase):
    def test_single_criterion(self) -> None:
        criterion = MinimumPreferenceOccurances(metric_name="m1", threshold=3)

        experiment = get_experiment()

        generation_strategy = GenerationStrategy(
            name="SOBOL+GPEI::default",
            steps=[
                GenerationStep(
                    model=Models.SOBOL, num_trials=-1, completion_criteria=[criterion]
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

    def test_many_criteria(self) -> None:
        criteria = [
            MinimumPreferenceOccurances(metric_name="m1", threshold=3),
            MinimumTrialsInStatus(status=TrialStatus.COMPLETED, threshold=5),
        ]

        experiment = get_experiment()

        generation_strategy = GenerationStrategy(
            name="SOBOL+GPEI::default",
            steps=[
                GenerationStep(
                    model=Models.SOBOL, num_trials=-1, completion_criteria=criteria
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
            # We have seen three "yes" and three "no", but not enough trials
            # are completed
            self.assertFalse(
                generation_strategy._maybe_move_to_next_step(
                    raise_data_required_error=False
                )
            )

        experiment._trial_indices_by_status = {TrialStatus.COMPLETED: {*range(6)}}
        # Enough trials are completed but we have not seen three "yes" and three
        # "no"
        self.assertFalse(
            generation_strategy._maybe_move_to_next_step(
                raise_data_required_error=False
            )
        )

        with patch.object(experiment, "fetch_data", return_value=data):
            # Enough trials are completed but we have not seen three "yes" and three
            # "no"
            self.assertTrue(
                generation_strategy._maybe_move_to_next_step(
                    raise_data_required_error=False
                )
            )

            self.assertEqual(generation_strategy._curr.model, Models.GPEI)

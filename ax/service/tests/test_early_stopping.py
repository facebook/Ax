# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

from ax.service.utils import early_stopping as early_stopping_utils
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    DummyEarlyStoppingStrategy,
    get_branin_experiment,
)


class TestEarlyStoppingUtils(TestCase):
    """Testing the early stopping utilities functionality that is not tested in
    main `AxClient` testing suite (`TestServiceAPI`)."""

    def setUp(self) -> None:
        self.branin_experiment = get_branin_experiment()

    def test_should_stop_trials_early(self) -> None:
        expected: Dict[int, Optional[str]] = {
            1: "Stopped due to testing.",
            3: "Stopped due to testing.",
        }
        actual = early_stopping_utils.should_stop_trials_early(
            early_stopping_strategy=DummyEarlyStoppingStrategy(expected),
            # pyre-fixme[6]: For 2nd param expected `Set[int]` but got `List[int]`.
            trial_indices=[1, 2, 3],
            experiment=self.branin_experiment,
        )
        self.assertEqual(actual, expected)

    def test_should_stop_trials_early_no_strategy(self) -> None:
        actual = early_stopping_utils.should_stop_trials_early(
            early_stopping_strategy=None,
            # pyre-fixme[6]: For 2nd param expected `Set[int]` but got `List[int]`.
            trial_indices=[1, 2, 3],
            experiment=self.branin_experiment,
        )
        expected = {}
        self.assertEqual(actual, expected)

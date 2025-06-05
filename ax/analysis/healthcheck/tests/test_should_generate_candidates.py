# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from random import randint

from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.analysis.healthcheck.should_generate_candidates import ShouldGenerateCandidates
from ax.utils.common.testutils import TestCase


class TestShouldGenerateCandidates(TestCase):
    def test_should(self) -> None:
        trial_index = randint(0, 10)
        card = ShouldGenerateCandidates(
            should_generate=True,
            reason="Something reassuring",
            trial_index=trial_index,
        ).compute()
        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
        self.assertEqual(card.subtitle, "Something reassuring")

    def test_should_not(self) -> None:
        trial_index = randint(0, 10)
        card = ShouldGenerateCandidates(
            should_generate=False,
            reason="Something concerning",
            trial_index=trial_index,
        ).compute()
        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertEqual(card.subtitle, "Something concerning")

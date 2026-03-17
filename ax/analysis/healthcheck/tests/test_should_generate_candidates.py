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
    def test_should_generate_candidates(self) -> None:
        for should_generate, reason, expected_status in [
            # should_generate=True -> PASS status
            (True, "Something reassuring", HealthcheckStatus.PASS),
            # should_generate=False -> WARNING status
            (False, "Something concerning", HealthcheckStatus.WARNING),
        ]:
            with self.subTest(should_generate=should_generate):
                trial_index = randint(0, 10)
                card = ShouldGenerateCandidates(
                    should_generate=should_generate,
                    reason=reason,
                    trial_index=trial_index,
                ).compute()
                self.assertEqual(card.get_status(), expected_status)
                self.assertEqual(card.subtitle, reason)

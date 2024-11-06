# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from datetime import datetime, timedelta

import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.healthcheck.can_generate_candidates import (
    CanGenerateCandidatesAnalysis,
)
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.core.base_trial import TrialStatus
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from pandas import testing as pdt


class TestCanGenerateCandidates(TestCase):
    def test_passes_if_can_generate(self) -> None:
        # GIVEN we can generate candidates
        # WHEN we run the healthcheck
        card = CanGenerateCandidatesAnalysis(
            can_generate_candidates=True,
            reason="No problems found.",
            days_till_fail=0,
        ).compute(experiment=None, generation_strategy=None)
        # THEN it is PASSES
        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
        self.assertEqual(card.name, "CanGenerateCandidates")
        self.assertEqual(card.title, "Ax Candidate Generation Success")
        self.assertEqual(card.subtitle, "No problems found.")
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        pdt.assert_frame_equal(
            card.df,
            pd.DataFrame(
                {
                    "status": [HealthcheckStatus.PASS.value],
                    "reason": ["No problems found."],
                }
            ),
        )

    def test_warns_if_a_trial_was_recently_run(self) -> None:
        # GIVEN an experiment with a recently run trial
        experiment = get_branin_experiment(with_trial=True)
        trial = experiment.trials[0]
        trial.mark_running(no_runner_required=True)
        trial._time_run_started = datetime.now() - timedelta(days=1)
        # WHEN we run the healthcheck
        card = CanGenerateCandidatesAnalysis(
            can_generate_candidates=False,
            reason="The data is borked.",
            days_till_fail=2,
        ).compute(experiment=experiment, generation_strategy=None)
        # THEN it is a WARNING
        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertEqual(card.name, "CanGenerateCandidates")
        self.assertEqual(card.title, "Ax Candidate Generation Warning")
        self.assertEqual(
            card.subtitle,
            (
                f"{CanGenerateCandidatesAnalysis.REASON_PREFIX}"
                "The data is borked.\n\n"
                "LAST TRIAL RUN: 1 day(s) ago"
            ),
        )
        self.assertEqual(card.level, AnalysisCardLevel.MID)
        pdt.assert_frame_equal(
            card.df,
            pd.DataFrame(
                {
                    "status": [HealthcheckStatus.WARNING.value],
                    "reason": ["The data is borked."],
                }
            ),
        )

    def test_is_fail_no_trials_have_been_run(self) -> None:
        # GIVEN an experiment with a candidate trial
        experiment = get_branin_experiment(with_trial=True)
        trial = experiment.trials[0]
        self.assertEqual(trial.status, TrialStatus.CANDIDATE)
        # WHEN we run the healthcheck
        card = CanGenerateCandidatesAnalysis(
            can_generate_candidates=False,
            reason="The data is gone.",
            days_till_fail=2,
        ).compute(experiment=experiment, generation_strategy=None)
        # THEN it is an ERROR
        self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
        self.assertEqual(card.name, "CanGenerateCandidates")
        self.assertEqual(card.title, "Ax Candidate Generation Failure")
        self.assertEqual(
            card.subtitle,
            f"{CanGenerateCandidatesAnalysis.REASON_PREFIX}The data is gone.",
        )
        self.assertEqual(card.level, AnalysisCardLevel.HIGH)
        pdt.assert_frame_equal(
            card.df,
            pd.DataFrame(
                {
                    "status": [HealthcheckStatus.FAIL.value],
                    "reason": ["The data is gone."],
                }
            ),
        )

    def test_is_fail_if_no_trial_was_recently_run(self) -> None:
        # GIVEN an experiment with an old trial
        experiment = get_branin_experiment(with_trial=True)
        trial = experiment.trials[0]
        trial.mark_running(no_runner_required=True)
        trial._time_run_started = datetime.now() - timedelta(days=3)
        trial.mark_completed()
        # WHEN we run the healthcheck
        card = CanGenerateCandidatesAnalysis(
            can_generate_candidates=False,
            reason="The data is old.",
            days_till_fail=1,
        ).compute(experiment=experiment, generation_strategy=None)
        # THEN it is an ERROR
        self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
        self.assertEqual(card.name, "CanGenerateCandidates")
        self.assertEqual(card.title, "Ax Candidate Generation Failure")
        self.assertEqual(
            card.subtitle,
            (
                f"{CanGenerateCandidatesAnalysis.REASON_PREFIX}"
                "The data is old.\n\n"
                "LAST TRIAL RUN: 3 day(s) ago"
            ),
        )
        self.assertEqual(card.level, AnalysisCardLevel.HIGH)
        pdt.assert_frame_equal(
            card.df,
            pd.DataFrame(
                {
                    "status": [HealthcheckStatus.FAIL.value],
                    "reason": ["The data is old."],
                }
            ),
        )

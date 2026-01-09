#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterable
from unittest import mock

from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.runner import Runner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_batch_trial, get_trial


class DummyRunner(Runner):
    # pyre-fixme[3]: Return type must be annotated.
    def run(self, trial: BaseTrial):
        return {"metadatum": f"value_for_trial_{trial.index}"}


class RunnerWithSuccessfulBatchPoll(Runner):
    """Runner where poll_trial_status succeeds for any number of trials."""

    def run(self, trial: BaseTrial) -> dict[str, str]:
        return {"metadatum": f"value_for_trial_{trial.index}"}

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        return {TrialStatus.COMPLETED: {t.index for t in trials}}


class RunnerWithFailingBatchPoll(Runner):
    """Runner where batch poll fails but individual poll succeeds."""

    def run(self, trial: BaseTrial) -> dict[str, str]:
        return {"metadatum": f"value_for_trial_{trial.index}"}

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        trials_list = list(trials)
        if len(trials_list) > 1:
            raise RuntimeError("Batch poll failure")
        return {TrialStatus.COMPLETED: {trials_list[0].index}}


class RunnerWithAllPollsFailing(Runner):
    """Runner where poll_trial_status always fails."""

    def run(self, trial: BaseTrial) -> dict[str, str]:
        return {"metadatum": f"value_for_trial_{trial.index}"}

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        raise RuntimeError("Poll failure")


class RunnerTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dummy_runner = DummyRunner()
        self.trials = [get_trial(), get_batch_trial()]

    def test_base_runner_staging_required(self) -> None:
        self.assertFalse(self.dummy_runner.staging_required)

    def test_base_runner_stop(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.dummy_runner.stop(trial=mock.Mock(), reason="")

    def test_base_runner_clone(self) -> None:
        runner_clone = self.dummy_runner.clone()
        self.assertIsInstance(runner_clone, DummyRunner)
        self.assertEqual(runner_clone, self.dummy_runner)

    def test_base_runner_run_multiple(self) -> None:
        metadata = self.dummy_runner.run_multiple(trials=self.trials)
        self.assertEqual(
            metadata,
            {t.index: {"metadatum": f"value_for_trial_{t.index}"} for t in self.trials},
        )
        self.assertEqual({}, self.dummy_runner.run_multiple(trials=[]))

    def test_base_runner_poll_trial_status(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.dummy_runner.poll_trial_status(trials=self.trials)

    def test_base_runner_poll_exception(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.dummy_runner.poll_exception(trial=self.trials[0])

    def test_poll_available_capacity(self) -> None:
        self.assertEqual(self.dummy_runner.poll_available_capacity(), -1)

    def test_run_metadata_report_keys(self) -> None:
        self.assertEqual(self.dummy_runner.run_metadata_report_keys, [])

    def test_robust_poll_trial_status_empty_trials(self) -> None:
        """Test that robust_poll_trial_status returns empty dict for empty trials."""
        runner = RunnerWithSuccessfulBatchPoll()
        result = runner.robust_poll_trial_status(trials=[])
        self.assertEqual(result, {})

    def test_robust_poll_trial_status_batch_success(self) -> None:
        """Test that robust_poll_trial_status uses batch poll when it succeeds."""
        runner = RunnerWithSuccessfulBatchPoll()
        result = runner.robust_poll_trial_status(trials=self.trials)
        self.assertEqual(
            result, {TrialStatus.COMPLETED: {t.index for t in self.trials}}
        )

    def test_robust_poll_trial_status_fallback_to_individual(self) -> None:
        """Test fallback to individual polling when batch poll fails."""
        runner = RunnerWithFailingBatchPoll()
        with self.assertLogs(logger="ax.core.runner", level="WARNING") as lg:
            result = runner.robust_poll_trial_status(trials=self.trials)

        # Check that the fallback warning was logged
        self.assertTrue(
            any(
                "Failed to poll all trial statuses at once" in msg
                and "Falling back to polling trials individually" in msg
                for msg in lg.output
            )
        )

        # All trials should still be marked as completed via individual polling
        self.assertEqual(
            result, {TrialStatus.COMPLETED: {t.index for t in self.trials}}
        )

    def test_robust_poll_trial_status_abandons_on_individual_failure(self) -> None:
        """Test that trials are marked ABANDONED when individual polling fails."""
        runner = RunnerWithAllPollsFailing()
        with self.assertLogs(logger="ax.core.runner", level="WARNING") as lg:
            result = runner.robust_poll_trial_status(trials=self.trials)

        # Check that the abandonment warning was logged for each trial
        for trial in self.trials:
            self.assertTrue(
                any(
                    f"Failed to retrieve status of trial {trial.index}" in msg
                    and "Setting trial status to ABANDONED" in msg
                    for msg in lg.output
                ),
                f"Expected abandonment warning for trial {trial.index} not found",
            )

        # All trials should be marked as abandoned
        self.assertEqual(
            result, {TrialStatus.ABANDONED: {t.index for t in self.trials}}
        )

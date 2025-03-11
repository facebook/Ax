#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterable
from datetime import datetime, timedelta
from random import randint
from typing import Any

from ax.core.base_trial import BaseTrial, TrialStatus
from ax.runners.single_running_trial_mixin import SingleRunningTrialMixin
from ax.runners.synthetic import SyntheticRunner
from ax.service.scheduler import Scheduler
from pyre_extensions import none_throws

DUMMY_EXCEPTION = "test_exception"
TEST_MEAN = 1.0


class SyntheticRunnerWithStatusPolling(SyntheticRunner):
    """Test runner that implements `poll_trial_status`, required for compatibility
    with the ``Scheduler``."""

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        # Pretend that sometimes trials take a few seconds to complete and that they
        # might get completed out of order.
        if randint(0, 3) > 0:
            running = [t.index for t in trials]
            return {TrialStatus.COMPLETED: {running[randint(0, len(running) - 1)]}}
        return {}


class SyntheticRunnerWithSingleRunningTrial(SingleRunningTrialMixin, SyntheticRunner):
    pass


class SyntheticRunnerWithPredictableStatusPolling(SyntheticRunner):
    """Test runner that implements `poll_trial_status`, required for compatibility
    with the ``Scheduler``, which polls completed."""

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        completed = {t.index for t in trials}
        return {TrialStatus.COMPLETED: completed}


class TestScheduler(Scheduler):
    """Test scheduler that only implements ``report_results`` for convenience in
    testing.
    """

    def report_results(self, force_refit: bool = False) -> dict[str, Any]:
        return {
            # Use `set` constructor to copy the set, else the value
            # will be a pointer and all will be the same.
            "trials_completed_so_far": set(
                self.experiment.trial_indices_by_status[TrialStatus.COMPLETED]
            ),
            "trials_early_stopped_so_far": set(
                self.experiment.trial_indices_by_status[TrialStatus.EARLY_STOPPED]
            ),
        }


# ---- Runners below simulate different usage and failure modes for scheduler ----


class RunnerWithFrequentFailedTrials(SyntheticRunner):
    poll_failed_next_time = True

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        running = [t.index for t in trials]
        status = (
            TrialStatus.FAILED if self.poll_failed_next_time else TrialStatus.COMPLETED
        )
        # Poll different status next time.
        self.poll_failed_next_time = not self.poll_failed_next_time
        return {status: {running[randint(0, len(running) - 1)]}}


class RunnerWithFailedAndAbandonedTrials(SyntheticRunner):
    poll_failed_next_time = True
    status_idx = 0

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        trial_statuses_dummy = {
            0: TrialStatus.ABANDONED,
            1: TrialStatus.FAILED,
            2: TrialStatus.COMPLETED,
        }
        running = [t.index for t in trials]
        status = trial_statuses_dummy[self.status_idx]

        # Poll different status next time.
        self.status_idx = (self.status_idx + 1) % 3

        return {status: {running[randint(0, len(running) - 1)]}}


class RunnerWithAllFailedTrials(SyntheticRunner):
    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        running = [t.index for t in trials]
        return {TrialStatus.FAILED: {running[randint(0, len(running) - 1)]}}


class NoReportResultsRunner(SyntheticRunner):
    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        if randint(0, 3) > 0:
            running = [t.index for t in trials]
            return {TrialStatus.COMPLETED: {running[randint(0, len(running) - 1)]}}
        return {}


class InfinitePollRunner(SyntheticRunner):
    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        return {}


class RunnerWithEarlyStoppingStrategy(SyntheticRunner):
    poll_trial_status_count = 0

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        self.poll_trial_status_count += 1

        # In the first step, don't complete any trials
        # Trial #1 will be early stopped
        if self.poll_trial_status_count == 1:
            return {}

        if self.poll_trial_status_count == 2:
            return {TrialStatus.COMPLETED: {2}}

        return {TrialStatus.COMPLETED: {0}}


class BrokenRunnerValueError(SyntheticRunnerWithStatusPolling):
    run_trial_call_count = 0

    def run_multiple(self, trials: Iterable[BaseTrial]) -> dict[int, dict[str, Any]]:
        self.run_trial_call_count += 1
        raise ValueError("Failing for testing purposes.")


class BrokenRunnerRuntimeError(SyntheticRunnerWithStatusPolling):
    run_trial_call_count = 0

    def run_multiple(self, trials: Iterable[BaseTrial]) -> dict[int, dict[str, Any]]:
        self.run_trial_call_count += 1
        raise RuntimeError("Failing for testing purposes.")


class RunnerToAllowMultipleMapMetricFetches(SyntheticRunnerWithStatusPolling):
    """``Runner`` that gives a trial 3 seconds to run before considering
    the trial completed, which gives us some time to fetch the ``MapMetric``
    a few times, if there is one on the experiment. Useful for testing behavior
    with repeated ``MapMetric`` fetches.
    """

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        running_trials = next(iter(trials)).experiment.trials_by_status[
            TrialStatus.RUNNING
        ]
        completed, still_running = set(), set()
        for t in running_trials:
            if datetime.now() - none_throws(t.time_run_started) > timedelta(seconds=3):
                completed.add(t.index)
            else:
                still_running.add(t.index)

        return {
            TrialStatus.COMPLETED: completed,
            TrialStatus.RUNNING: still_running,
        }

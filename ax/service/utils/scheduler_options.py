# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from logging import INFO
from typing import Optional

from ax.early_stopping.strategies import BaseEarlyStoppingStrategy
from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy


class TrialType(Enum):
    TRIAL = 0
    BATCH_TRIAL = 1


@dataclass(frozen=True)
class SchedulerOptions:
    """Settings for a scheduler instance.

    Attributes:
        max_pending_trials: Maximum number of pending trials the scheduler
            can have ``STAGED`` or ``RUNNING`` at once, required. If looking
            to use ``Runner.poll_available_capacity`` as a primary guide for
            how many trials should be pending at a given time, set this limit
            to a high number, as an upper bound on number of trials that
            should not be exceeded.
        trial_type: Type of trials (1-arm ``Trial`` or multi-arm ``Batch
            Trial``) that will be deployed using the scheduler. Defaults
            to 1-arm `Trial`. NOTE: use ``BatchTrial`` only if need to
            evaluate multiple arms *together*, e.g. in an A/B-test
            influenced by data nonstationarity. For cases where just
            deploying multiple arms at once is beneficial but the trials
            are evaluated *independently*, implement ``run_trials`` method
            in scheduler subclass, to deploy multiple 1-arm trials at
            the same time.
        batch_size: If using BatchTrial the number of arms to be generated and
            deployed per trial.
        total_trials: Limit on number of trials a given ``Scheduler``
            should run. If no stopping criteria are implemented on
            a given scheduler, exhaustion of this number of trials
            will be used as default stopping criterion in
            ``Scheduler.run_all_trials``. Required to be non-null if
            using ``Scheduler.run_all_trials`` (not required for
            ``Scheduler.run_n_trials``).
        tolerated_trial_failure_rate: Fraction of trials in this
            optimization that are allowed to fail without the whole
            optimization ending. Expects value between 0 and 1.
            NOTE: Failure rate checks begin once
            min_failed_trials_for_failure_rate_check trials have
            failed; after that point if the ratio of failed trials
            to total trials ran so far exceeds the failure rate,
            the optimization will halt.
        min_failed_trials_for_failure_rate_check: The minimum number
            of trials that must fail in `Scheduler` in order to start
            checking failure rate.
        log_filepath: File, to which to write optimization logs.
        logging_level: Minimum level of logging statements to log,
            defaults to ``logging.INFO``.
        ttl_seconds_for_trials: Optional TTL for all trials created
            within this ``Scheduler``, in seconds. Trials that remain
            ``RUNNING`` for more than their TTL seconds will be marked
            ``FAILED`` once the TTL elapses and may be re-suggested by
            the Ax optimization models.
        init_seconds_between_polls: Initial wait between rounds of
            polling, in seconds. Relevant if using the default wait-
            for-completed-runs functionality of the base ``Scheduler``
            (if ``wait_for_completed_trials_and_report_results`` is not
            overridden). With the default waiting, every time a poll
            returns that no trial evaluations completed, wait
            time will increase; once some completed trial evaluations
            are found, it will reset back to this value. Specify 0
            to not introduce any wait between polls.
        min_seconds_before_poll: Minimum number of seconds between
            beginning to run a trial and the first poll to check
            trial status.
        timeout_hours: Number of hours after which the optimization will abort.
        seconds_between_polls_backoff_factor: The rate at which the poll
            interval increases.
        run_trials_in_batches: If True and ``poll_available_capacity`` is
            implemented to return non-null results, trials will be dispatched
            in groups via `run_trials` instead of one-by-one via ``run_trial``.
            This allows to save time, IO calls or computation in cases where
            dispatching trials in groups is more efficient then sequential
            deployment. The size of the groups will be determined as
            the minimum of ``self.poll_available_capacity()`` and the number
            of generator runs that the generation strategy is able to produce
            without more data or reaching its allowed max paralellism limit.
        debug_log_run_metadata: Whether to log run_metadata for debugging purposes.
        early_stopping_strategy: A ``BaseEarlyStoppingStrategy`` that determines
            whether a trial should be stopped given the current state of
            the experiment. Used in ``should_stop_trials_early``.
        global_stopping_strategy: A ``BaseGlobalStoppingStrategy`` that determines
            whether the full optimization should be stopped or not.
        suppress_storage_errors_after_retries: Whether to fully suppress SQL
            storage-related errors if encounted, after retrying the call
            multiple times. Only use if SQL storage is not important for the given
            use case, since this will only log, but not raise, an exception if
            it's encountered while saving to DB or loading from it.
    """

    max_pending_trials: int = 10
    trial_type: TrialType = TrialType.TRIAL
    batch_size: Optional[int] = None
    total_trials: Optional[int] = None
    tolerated_trial_failure_rate: float = 0.5
    min_failed_trials_for_failure_rate_check: int = 5
    log_filepath: Optional[str] = None
    logging_level: int = INFO
    ttl_seconds_for_trials: Optional[int] = None
    init_seconds_between_polls: Optional[int] = 1
    min_seconds_before_poll: float = 1.0
    seconds_between_polls_backoff_factor: float = 1.5
    timeout_hours: Optional[float] = None
    run_trials_in_batches: bool = False
    debug_log_run_metadata: bool = False
    early_stopping_strategy: Optional[BaseEarlyStoppingStrategy] = None
    global_stopping_strategy: Optional[BaseGlobalStoppingStrategy] = None
    suppress_storage_errors_after_retries: bool = False

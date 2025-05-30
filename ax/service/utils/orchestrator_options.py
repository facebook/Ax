# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from enum import Enum
from logging import INFO
from typing import Any

from ax.early_stopping.strategies import BaseEarlyStoppingStrategy
from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy


class TrialType(Enum):
    TRIAL = 0
    BATCH_TRIAL = 1


@dataclass(frozen=True)
class OrchestratorOptions:
    """Settings for a Orchestrator instance.

    Attributes:
        max_pending_trials: Maximum number of pending trials the Orchestrator
            can have ``STAGED`` or ``RUNNING`` at once, required. If looking
            to use ``Runner.poll_available_capacity`` as a primary guide for
            how many trials should be pending at a given time, set this limit
            to a high number, as an upper bound on number of trials that
            should not be exceeded.
        trial_type: Type of trials (1-arm ``Trial`` or multi-arm ``Batch
            Trial``) that will be deployed using the orchestrator. Defaults
            to 1-arm `Trial`. NOTE: use ``BatchTrial`` only if need to
            evaluate multiple arms *together*, e.g. in an A/B-test
            influenced by data nonstationarity. For cases where just
            deploying multiple arms at once is beneficial but the trials
            are evaluated *independently*, implement ``run_trials`` method
            in Orchestrator subclass, to deploy multiple 1-arm trials at
            the same time.
        batch_size: If using BatchTrial the number of arms to be generated and
            deployed per trial.
        total_trials: Limit on number of trials a given ``Orchestrator``
            should run. If no stopping criteria are implemented on
            a given Orchestrator, exhaustion of this number of trials
            will be used as default stopping criterion in
            ``orchestrator.run_all_trials``. Required to be non-null if
            using ``orchestrator.run_all_trials`` (not required for
            ``orchestrator.run_n_trials``).
        tolerated_trial_failure_rate: Fraction of trials in this
            optimization that are allowed to fail without the whole
            optimization ending. Expects value between 0 and 1.
            NOTE: Failure rate checks begin once
            min_failed_trials_for_failure_rate_check trials have
            failed; after that point if the ratio of failed trials
            to total trials ran so far exceeds the failure rate,
            the optimization will halt.
        min_failed_trials_for_failure_rate_check: The minimum number
            of trials that must fail in `Orchestrator` in order to start
            checking failure rate.
        log_filepath: File, to which to write optimization logs.
        logging_level: Minimum level of logging statements to log,
            defaults to ``logging.INFO``.
        ttl_seconds_for_trials: Optional TTL for all trials created
            within this ``Orchestrator``, in seconds. Trials that remain
            ``RUNNING`` for more than their TTL seconds will be marked
            ``FAILED`` once the TTL elapses and may be re-suggested by
            the Ax optimization models.
        init_seconds_between_polls: Initial wait between rounds of
            polling, in seconds. Relevant if using the default wait-
            for-completed-runs functionality of the base ``Orchestrator``
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
            storage-related errors if encountered, after retrying the call
            multiple times. Only use if SQL storage is not important for the given
            use case, since this will only log, but not raise, an exception if
            it's encountered while saving to DB or loading from it.
        wait_for_running_trials: Whether the Orchestrator should wait for running trials
            or exit.
        fetch_kwargs: Kwargs to be used when fetching data.
        validate_metrics: Whether to raise an error if there is a problem with the
            metrics attached to the experiment.
        status_quo_weight: The weight of the status quo arm. This is only used
            if the Orchestrator is using a BatchTrial. This requires that the status_quo
            be set on the experiment.
        enforce_immutable_search_space_and_opt_config: Whether to enforce that the
            search space and optimization config are immutable.  If true, will add
            `"immutable_search_space_and_opt_config": True` to experiment properties
        mt_experiment_trial_type: Type of trial to run for MultiTypeExperiments. This
            is currently required for MultiTypeExperiments. This is ignored for
            "regular" or single type experiments. If you don't know what a single type
            experiment is, you don't need this.
        force_candidate_generation: Whether to force candidate generation even if the
            generation strategy is not ready to generate candidates, meaning one of the
            transition criteria with block_gen_if_met is met.
            **This is not yet implemented.**
    """

    max_pending_trials: int = 10
    trial_type: TrialType = TrialType.TRIAL
    batch_size: int | None = None
    total_trials: int | None = None
    tolerated_trial_failure_rate: float = 0.5
    min_failed_trials_for_failure_rate_check: int = 5
    log_filepath: str | None = None
    logging_level: int = INFO
    ttl_seconds_for_trials: int | None = None
    init_seconds_between_polls: int | None = 1
    min_seconds_before_poll: float = 1.0
    seconds_between_polls_backoff_factor: float = 1.5
    run_trials_in_batches: bool = False
    debug_log_run_metadata: bool = False
    early_stopping_strategy: BaseEarlyStoppingStrategy | None = None
    global_stopping_strategy: BaseGlobalStoppingStrategy | None = None
    suppress_storage_errors_after_retries: bool = False
    wait_for_running_trials: bool = True
    fetch_kwargs: dict[str, Any] = field(default_factory=dict)
    validate_metrics: bool = True
    status_quo_weight: float = 0.0
    enforce_immutable_search_space_and_opt_config: bool = True
    mt_experiment_trial_type: str | None = None
    force_candidate_generation: bool = False

    def __post_init__(self) -> None:
        if self.early_stopping_strategy is not None:
            object.__setattr__(self, "seconds_between_polls_backoff_factor", 1)

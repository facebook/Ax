#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from logging import INFO, LoggerAdapter
from time import sleep
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    cast,
)

from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.trial import Trial
from ax.early_stopping.strategies import BaseEarlyStoppingStrategy
from ax.exceptions.core import (
    AxError,
    DataRequiredError,
    OptimizationComplete,
    UnsupportedError,
)
from ax.exceptions.generation_strategy import MaxParallelismReachedException
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.modelbridge_utils import (
    get_pending_observation_features_based_on_trial_status,
)
from ax.service.utils.with_db_settings_base import DBSettings, WithDBSettingsBase
from ax.utils.common.constants import Keys
from ax.utils.common.executils import retry_on_exception
from ax.utils.common.logger import build_file_handler, get_logger, set_stderr_log_level
from ax.utils.common.timeutils import current_timestamp_in_millis
from ax.utils.common.typeutils import not_none


NOT_IMPLEMENTED_IN_BASE_CLASS_MSG = """
This method is not implemented in the base `Scheduler` class.
If this functionality is desired, specify the method in the
scheduler subclass.
"""
GS_TYPE_MSG = "This optimization run uses a '{gs_name}' generation strategy."
OPTIMIZATION_COMPLETION_MSG = """Optimization completed with total of {num_trials}
trials attached to the underlying Ax experiment '{experiment_name}'.
"""


# Wait time b/w polls will not exceed 5 mins.
MAX_SECONDS_BETWEEN_POLLS = 300


class OptimizationResult(NamedTuple):  # TODO[T61776778]
    pass  # TBD


class SchedulerInternalError(AxError):
    """Error that indicates an error within the `Scheduler` logic."""

    pass


class FailureRateExceededError(AxError):
    """Error that indicates the sweep was aborted due to excessive failure rate."""

    pass


NO_RETRY_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    cast(Type[Exception], SchedulerInternalError),
    cast(Type[Exception], NotImplementedError),
    cast(Type[Exception], UnsupportedError),
)


class ExperimentStatusProperties(str, Enum):
    """Enum for keys in experiment properties that represent status of
    optimization run through scheduler."""

    # Number of trials run in each call to `Scheduler.run_trials_and_
    # yield_results`.
    NUM_TRIALS_RUN_PER_CALL = "num_trials_run_per_call"
    # Status of each run of `Scheduler.run_trials_and_
    # yield_results`. Recorded twice in a successful/aborted run; first
    # "started" is recorded, then "success" or "aborted". If no second
    # status is recorded, run must have encountered an exception.
    RUN_TRIALS_STATUS = "run_trials_success"
    # Timestamps of when the experiment was resumed from storage.
    RESUMED_FROM_STORAGE_TIMESTAMPS = "resumed_from_storage_timestamps"


class RunTrialsStatus(str, Enum):
    """Possible statuses for each call to ``Scheduler.run_trials_and_
    yield_results``, used in recording experiment status.
    """

    STARTED = "started"
    SUCCESS = "success"
    ABORTED = "aborted"


@dataclass(frozen=True)
class SchedulerOptions:
    """Settings for a scheduler instance.

    Attributes:
        trial_type: Type of trials (1-arm ``Trial`` or multi-arm ``Batch
            Trial``) that will be deployed using the scheduler. Defaults
            to 1-arm `Trial`. NOTE: use ``BatchTrial`` only if need to
            evaluate multiple arms *together*, e.g. in an A/B-test
            influenced by data nonstationarity. For cases where just
            deploying multiple arms at once is beneficial but the trials
            are evaluated *independently*, implement ``run_trials`` method
            in scheduler subclass, to deploy multiple 1-arm trials at
            the same time.
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
        suppress_storage_errors_after_retries: Whether to fully suppress SQL
            storage-related errors if encounted, after retrying the call
            multiple times. Only use if SQL storage is not important for the given
            use case, since this will only log, but not raise, an exception if
            it's encountered while saving to DB or loading from it.
    """

    trial_type: Type[BaseTrial] = Trial
    total_trials: Optional[int] = None
    tolerated_trial_failure_rate: float = 0.5
    min_failed_trials_for_failure_rate_check: int = 5
    log_filepath: Optional[str] = None
    logging_level: int = INFO
    ttl_seconds_for_trials: Optional[int] = None
    init_seconds_between_polls: Optional[int] = 1
    min_seconds_before_poll: float = 1.0
    seconds_between_polls_backoff_factor: float = 1.5
    run_trials_in_batches: bool = False
    debug_log_run_metadata: bool = False
    early_stopping_strategy: Optional[BaseEarlyStoppingStrategy] = None
    suppress_storage_errors_after_retries: bool = False


class Scheduler(WithDBSettingsBase, ABC):
    """Closed-loop manager class for Ax optimization.

    Attributes:
        experiment: Experiment, in which results of the optimization
            will be recorded.
        generation_strategy: Generation strategy for the optimization,
            describes models that will be used in optimization.
        options: `SchedulerOptions` for this scheduler instance.
        db_settings: Settings for saving and reloading the underlying experiment
            to a database. Expected to be of type
            ax.storage.sqa_store.structs.DBSettings and require SQLAlchemy.
        _skip_experiment_save: If True, scheduler will not re-save the
            experiment passed to it. **Use only if the experiment had just
            been saved, as otherwise experiment state could get corrupted.**
    """

    experiment: Experiment
    generation_strategy: GenerationStrategy
    options: SchedulerOptions
    logger: LoggerAdapter
    # Mapping of form {short string identifier -> message to show in reported
    # results}. This is a mapping and not a list to allow for changing of
    # some sweep messages throughout the course of the optimization (e.g. progress
    # report of the optimization).
    markdown_messages: Dict[str, str]

    # Number of trials that existed on the scheduler's experiment before
    # the scheduler instantiation with that experiment.
    _num_preexisting_trials: int
    # Timestamp of last optimization start time (milliseconds since Unix epoch);
    # recorded in each `run_n_trials`.
    _latest_optimization_start_timestamp: Optional[int] = None
    # Timeout setting for current optimization.
    _timeout_hours: Optional[int] = None
    # Timestamp of when the last deployed trial started running.
    _latest_trial_start_timestamp: Optional[float] = None
    # Will be set to `True` if generation strategy signals that the optimization
    # is complete, in which case the optimization should gracefully exit early.
    _optimization_complete: bool = False

    def __init__(
        self,
        experiment: Experiment,
        generation_strategy: GenerationStrategy,
        options: SchedulerOptions,
        db_settings: Optional[DBSettings] = None,
        _skip_experiment_save: bool = False,
    ) -> None:
        # Initialize options used in `__repr__` upfront, before any errors
        # might be enncountered, reporting of which would call `__repr__`.
        self.options = options
        self.experiment = experiment
        # NOTE: Parallelism schedule is embedded in the generation
        # strategy, as `GenerationStep.max_parallelism`.
        self.generation_strategy = generation_strategy

        if not isinstance(experiment, Experiment):  # pragma: no cover
            raise TypeError("{experiment} is not an Ax experiment.")
        if not isinstance(generation_strategy, GenerationStrategy):  # pragma: no cover
            raise TypeError("{generation_strategy} is not a generation strategy.")
        self._validate_options(options=options)

        # Initialize storage layer for the scheduler.
        super().__init__(
            db_settings=db_settings,
            logging_level=self.options.logging_level,
            suppress_all_errors=self.options.suppress_storage_errors_after_retries,
        )

        # Set up logger with an optional filepath handler
        self._set_logger()

        # Validate experiment and GS; ensure that experiment has immutable
        # search space and opt. config to avoid storing their  copies on each
        # generator run.
        self._validate_remaining_trials(experiment=experiment)
        self._validate_implemented_metrics(experiment=experiment)
        self._enforce_immutable_search_space_and_opt_config()
        self._initialize_experiment_status_properties()

        if self.db_settings_set and not _skip_experiment_save:
            self._maybe_save_experiment_and_generation_strategy(
                experiment=experiment, generation_strategy=generation_strategy
            )

        # Number of trials that existed on experiment before this scheduler.
        self._num_preexisting_trials = len(experiment.trials)
        # Whether to log the reason why no trials were generated next time
        # we prepare new trials for deployment. Used to avoid spamming logs
        # when trials are not generated for the same reason multiple times in
        # a row.
        self._log_next_no_trials_reason = True
        self.markdown_messages = {
            "generation_strategy": GS_TYPE_MSG.format(gs_name=generation_strategy.name)
        }

    @classmethod
    def get_default_db_settings(cls) -> DBSettings:
        raise NotImplementedError(  # pragma: no cover
            "Base `Scheduler` does not specify default `DBSettings`. "
            "DBSettings are required to leverage SQL storage functionality "
            "and can be specified as argument to `Scheduler` constructor or "
            "via `get_default_db_settings` implementation on given scheduler."
        )

    @classmethod
    def from_stored_experiment(
        cls,
        experiment_name: str,
        options: SchedulerOptions,
        db_settings: Optional[DBSettings] = None,
        generation_strategy: Optional[GenerationStrategy] = None,
        **kwargs: Any,
    ) -> Scheduler:
        """Create a ``Scheduler`` with a previously stored experiment, which
        the scheduler should resume.

        Args:
            experiment_name: Experiment to load and resume.
            options: ``SchedulerOptions``, with which to set up the new scheduler.
            db_settings: Optional ``DBSettings``, which to use for reloading the
                experiment; also passed as ``db_settings`` argument to the
                scheduler constructor.
            generation_strategy: Generation strategy to use to provide candidates
                for the resumed optimization. Provide this argument only if
                the experiment does not already have a generation strategy
                associated with it.
            kwargs: Kwargs to pass through to the ``Scheduler`` constructor.
        """
        dbs = WithDBSettingsBase(
            db_settings=db_settings or cls.get_default_db_settings()
        )
        exp, gs = dbs._load_experiment_and_generation_strategy(
            experiment_name=experiment_name, reduced_state=True
        )
        if db_settings:
            kwargs = {**kwargs, "db_settings": db_settings}
        if not exp:  # pragma: no cover
            raise ValueError(f"Experiment {experiment_name} not found.")

        if not gs and not generation_strategy:  # pragma: no cover
            raise ValueError(
                f"Experiment {experiment_name} did not have a generation "
                "strategy associated with in in database, so a new "
                "generation strategy must be provided as argument to "
                "`Scheduler.from_stored_experiment`."
            )

        if gs and generation_strategy and gs != generation_strategy:
            # NOTE: In the future we may want to allow overriding of GS,
            # in which case we can add a flag to this function and allow
            # the override with warning.
            raise UnsupportedError(  # pragma: no cover
                "Experiment was associated with generation strategy "
                f"{gs.name} in DB, but a new generation strategy "
                f"{generation_strategy.name} was provided. To use "
                "the generation strategy currently in DB, do not "
                "specify the `geneneration_strategy` kwarg."
            )

        # pyre-ignore[45]: Let Python error if instantiation of abstract
        # base `Scheduler` is attempted, as error will be informative.
        scheduler = cls(
            experiment=exp,
            generation_strategy=not_none(generation_strategy or gs),
            options=options,
            # No need to resave the experiment we just reloaded.
            _skip_experiment_save=True,
            # NOTE: `kwargs` can include `db_settings` if those were
            # provided to this function.
            **kwargs,
        )
        ts = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S.%f")
        scheduler._append_to_experiment_properties(
            to_append={
                ExperimentStatusProperties.RESUMED_FROM_STORAGE_TIMESTAMPS: ts,
            }
        )
        return scheduler

    @property
    def running_trials(self) -> List[BaseTrial]:
        """Currently running trials.

        Returns:
            List of trials that are currently running.
        """
        return self.experiment.trials_by_status[TrialStatus.RUNNING]

    @property
    def candidate_trials(self) -> List[BaseTrial]:
        """Candidate trials on the experiment this scheduler is running.

        Returns:
            List of trials that are currently candidates.
        """
        return self.experiment.trials_by_status[TrialStatus.CANDIDATE]

    @property
    def has_trials_in_flight(self) -> bool:
        """Whether the experiment on this scheduler currently has running or staged
        trials.
        """
        return (
            len(self.running_trials) > 0
            or len(self.experiment.trial_indices_by_status[TrialStatus.STAGED]) > 0
        )

    def __repr__(self) -> str:
        """Short user-friendly string representation."""
        if not hasattr(self, "experiment"):  # pragma: no cover
            # Experiment, generation strategy, etc. attributes have not
            # yet been set.
            return f"{self.__class__.__name__}"
        return (
            f"{self.__class__.__name__}(experiment={self.experiment}, "
            f"generation_strategy={self.generation_strategy}, options="
            f"{self.options})"
        )

    # ----------------- User-defined, required. -----------------

    @abstractmethod
    def poll_trial_status(self) -> Dict[TrialStatus, Set[int]]:
        """Required polling function, checks the status of any non-terminal trials
        and returns their indices as a mapping from TrialStatus to a list of indices.

        NOTE: Does not need to handle waiting between polling while trials
        are running; that logic is handled in `Scheduler.poll`, which calls
        this function.

        Returns:
            A dictionary mapping TrialStatus to a list of trial indices that have
            the respective status at the time of the polling. This does not need to
            include trials that at the time of polling already have a terminal
            (ABANDONED, FAILED, COMPLETED) status (but it may).
        """
        ...  # pragma: no cover

    # ----------------- User-defined, optional. -----------------

    def has_capacity(self, n: int = 1) -> bool:
        """Optional method to checks if there is available capacity to
        schedule `n` trials.

        Args:
            n: Number of trials, the capacity to run which is being checked.
                Defaults to 1.

        Returns:
            A boolean, representing whether `n` trials can be ran.
        """
        return True

    def poll_available_capacity(self) -> Optional[int]:
        """Optional method to checks how much available capacity there is
        to schedule trial evaluations.

        Returns:
            An optional integer, representing how many trials there is
            available capacity for, if available. If not available,
            returns `None`.
        """
        return None

    def completion_criterion(self) -> bool:
        """Optional stopping criterion for optimization, defaults to a check
        of whether `total_trials` trials have been run.

        Returns:
            Boolean representing whether the optimization should be stopped.
        """
        # TODO[Max, T61776778]: Default model-informed stopping criterion.

        if self.options.total_trials is None:
            # We validate that `total_trials` is set in `run_all_trials`,
            # so it will not run infinitely.
            return False

        expecting_data = sum(  # Number of `RUNNING` + `COMPLETED` trials
            1 for t in self.experiment.trials.values() if t.status.expecting_data
        )
        return expecting_data >= not_none(self.options.total_trials)

    def report_results(self) -> Dict[str, Any]:
        """Optional user-defined function for reporting intermediate
        and final optimization results (e.g. make some API call, write to some
        other db). This function is called whenever new results are available during
        the optimization.

        Returns:
            An optional dictionary with any relevant data about optimization.
        """
        # TODO[T61776778]: add utility to get best trial from arbitrary exp.
        return {}

    @retry_on_exception(retries=3, no_retry_on_exception_types=NO_RETRY_EXCEPTIONS)
    def run_trial(self, trial: BaseTrial) -> Dict[str, Any]:
        """Optional deployment function, runs a single evaluation of the
        given trial. Can be used instead of `runner.run(trial)` if no
        runner is defined on the experiment; will be required in that case.

        NOTE: the `retry_on_exception` decorator applied to this function should also
        be applied to its subclassing override if one is provided and retry behavior
        is desired.

        Args:
            trial: Trial to be deployed, contains arms with
                parameterizations to be evaluated. Can be a `Trial`
                if contains only one arm or a `BatchTrial` if contains
                multiple arms.

        Returns:
            Dict of run metadata from the deployment process.
        """
        if self.experiment.runner is None:
            raise NotImplementedError(
                "A runner is required on experiment to use its `run` method to "
                "run a trial evaluation. Alternatively, `run_trial` can be defined "
                "on a subclass of `Scheduler` as a substitute of a runner."
            )
        return not_none(self.experiment.runner).run(trial=trial)

    def stop_trial_runs(
        self, trials: List[BaseTrial], reasons: Optional[List[Optional[str]]] = None
    ) -> None:
        """Stops the jobs that execute given trials.

        Used if, for example, TTL for a trial was specified and expired, or poor
        early results suggest the trial is not worth running to completion.

        Requires a runner to be defined on the experiment in this base class
        implementation, but can be overridden in subclasses to not require a runner.

        Overwrite default implementation if its desirable to stop trials in bulk.

        Args:
            trials: Trials to be stopped.
            reasons: A list of strings describing the reasons for why the
                trials are to be stopped (in the same order).
        """
        if len(trials) == 0:
            return

        if self.experiment.runner is None:
            raise NotImplementedError(  # pragma: no cover
                "A runner is required on experiment to use its `stop` method to "
                "stop a trial evaluation. Alternatively, `stop_trial` can be defined "
                "on a subclass of `Scheduler` as a substitute of a runner."
            )

        runner = not_none(self.experiment.runner)
        if reasons is None:
            reasons = [None] * len(trials)

        for trial, reason in zip(trials, reasons):
            runner.stop(trial=trial, reason=reason)

    def stop_trial_run(self, trial: BaseTrial, reason: Optional[str] = None) -> None:
        """Stops the job that executes a given trial.

        Args:
            trial: Trial to be stopped.
            reason: The reason the trial is to be stopped.
        """
        self.stop_trial_runs(trials=[trial], reasons=[reason])

    @retry_on_exception(retries=3, no_retry_on_exception_types=NO_RETRY_EXCEPTIONS)
    def run_trials(self, trials: Iterable[BaseTrial]) -> Dict[int, Dict[str, Any]]:
        """Optional deployment function, runs a single evaluation for each of the
        given trials. By default simply loops over `run_trial`. Should be overwritten
        if deploying multiple trials in batch is preferable.

        NOTE: the `retry_on_exception` decorator applied to this function should also
        be applied to its subclassing override if one is provided and retry behavior
        is desired.

        Args:
            trials: Iterable of trials to be deployed, each containing arms with
                parameterizations to be evaluated. Can be a `Trial`
                if contains only one arm or a `BatchTrial` if contains
                multiple arms.

        Returns:
            Dict of trial index to the run metadata of that trial from the deployment
            process.
        """
        return {trial.index: self.run_trial(trial=trial) for trial in trials}

    def wait_for_completed_trials_and_report_results(self) -> Dict[str, Any]:
        """Continuously poll for successful trials, with limited exponential
        backoff, and process the results. Stop once at least one successful
        trial has been found. This function can be overridden to a different
        waiting function as needed; it must call `poll_and_process_results`
        to ensure that trials that completed their evaluation are appropriately
        marked as 'COMPLETED' in Ax.

        Returns: Results of the optimization so far, represented as a
        dict. The contents of the dict depend on the implementation of
        `report_results` in the given `Scheduler` subclass.
        """
        if self.options.init_seconds_between_polls is None:
            raise ValueError(  # pragma: no cover
                "Default `wait_for_completed_trials_and_report_results` in base "
                "`Scheduler` relies on non-null `init_seconds_between_polls` scheduler "
                "option."
            )
        seconds_between_polls = not_none(self.options.init_seconds_between_polls)
        while self.has_trials_in_flight and not self.poll_and_process_results():
            if seconds_between_polls > MAX_SECONDS_BETWEEN_POLLS:
                break  # If maximum wait time reached, check the stopping
                # criterion again and and re-attempt scheduling more trials.
            log_seconds = (
                int(seconds_between_polls)
                if seconds_between_polls > 2
                else seconds_between_polls
            )
            self.logger.info(
                f"Waiting for completed trials (for {log_seconds} sec, "
                f"currently running trials: {len(self.running_trials)})."
            )
            sleep(seconds_between_polls)
            seconds_between_polls *= self.options.seconds_between_polls_backoff_factor
        return self.report_results()

    def should_consider_optimization_complete(self) -> bool:
        """Whether this scheduler should consider this optimization complete and not
        run more trials (and conclude the optimization via ``_complete_optimization``).
        An optimization is considered complete when a generation strategy signalled
        completion or when the custom ``completion_criterion`` on this scheduler
        evaluates to ``True``.
        """
        if self._optimization_complete:
            return True

        return self.completion_criterion()

    def should_abort_optimization(self) -> bool:
        """Checks whether this scheduler has reached some intertuption / abort
        criterion, such as an overall optimization timeout, tolerated failure rate, etc.
        """
        # if failure rate is exceeded, raise an exception.
        # this check should precede others to ensure it is not skipped.
        self.error_if_failure_rate_exceeded()

        # if sweep is timed out, return True, else return False
        timed_out = (
            self._timeout_hours is not None
            and self._latest_optimization_start_timestamp is not None
            and current_timestamp_in_millis()
            - not_none(self._latest_optimization_start_timestamp)
            >= not_none(self._timeout_hours) * 60 * 60 * 1000
        )
        if timed_out:
            self.logger.error(
                "Optimization timed out (timeout hours: " f"{self._timeout_hours})!"
            )
        return timed_out

    def error_if_failure_rate_exceeded(self, force_check: bool = False) -> None:
        """Checks if the failure rate (set in scheduler options) has been exceeded.

        Args:
            force_check: Indicates whether to force a failure-rate check
                regardless of the number of trials that have been executed. If False
                (default), the check will be skipped if the sweep has fewer than five
                failed iterations. If True, the check will be performed unless there
                are 0 failures.
        """
        failed_idcs = self.experiment.trial_indices_by_status[TrialStatus.FAILED]
        # We only count failed trials with indices that came after the preexisting
        # trials on experiment before scheduler use.
        num_failed_in_scheduler = sum(
            1 for f in failed_idcs if f >= self._num_preexisting_trials
        )

        # skip check if 0 failures
        if num_failed_in_scheduler == 0:
            return

        # skip check if fewer than min_failed_trials_for_failure_rate_check failures
        # unless force_check is True
        if (
            num_failed_in_scheduler
            < self.options.min_failed_trials_for_failure_rate_check
            and not force_check
        ):
            return

        num_ran_in_scheduler = (
            len(self.experiment.trials) - self._num_preexisting_trials
        )

        failure_rate_exceeded = (
            num_failed_in_scheduler / num_ran_in_scheduler
        ) > self.options.tolerated_trial_failure_rate

        if failure_rate_exceeded:
            raise FailureRateExceededError(
                "Tolerated trial failure rate exceeded (at least "
                f"{num_failed_in_scheduler} out of first {num_ran_in_scheduler} trials "
                " failed)."
            )

    def summarize_final_result(self) -> OptimizationResult:
        """Get some summary of result: which trial did best, what
        were the metric values, what were encountered failures, etc.
        """
        return OptimizationResult()  # pragma: no cover, TODO[T61776778]

    # ---------- Methods below should generally not be modified in subclasses. ---------

    def run_trials_and_yield_results(
        self, max_trials: int, timeout_hours: Optional[int] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Make continuous calls to `run` and `process_results` to run up to
        ``max_trials`` trials, until completion criterion is reached. This is the 'main'
        method of a ``Scheduler``.

        Args:
            max_trials: Maximum number of trials to run in this generator. The
                generator will run trials
            timeout_hours: Maximum number of hours, for which
                to run the optimization. This function will abort after running
                for `timeout_hours` even if stopping criterion has not been reached.
                If set to `None`, no optimization timeout will be applied.
        """
        self._latest_optimization_start_timestamp = current_timestamp_in_millis()
        if timeout_hours is not None:
            if timeout_hours < 0:  # pragma: no cover
                raise ValueError(f"Expected `timeout_hours` >= 0, got {timeout_hours}.")
            self._timeout_hours = timeout_hours

        if max_trials < 0:
            raise ValueError(f"Expected `max_trials` >= 0, got {max_trials}.")
        trials = self.experiment.trials
        n_existing = len(self.experiment.trials)

        self._record_run_trials_status(
            num_preexisting_trials=None, status=RunTrialsStatus.STARTED
        )

        while len(self.candidate_trials) > 0:
            self.run(max_new_trials=0)
            # only wait for trials to complete if max_pending_trials are already running
            if not self.has_capacity():
                yield self.wait_for_completed_trials_and_report_results()

        # Until completion criterion is reached or `max_trials` is scheduled,
        # schedule new trials and poll existing ones in a loop.
        while (
            not self.should_consider_optimization_complete()
            and len(trials) - n_existing < max_trials
        ):
            if self.should_abort_optimization():
                yield self._abort_optimization(num_preexisting_trials=n_existing)
                return

            # Run new trial evaluations until `run` returns `False`, which
            # means that there was a reason not to run more evaluations yet.
            # Also check that `max_trials` is not reached to not exceed it.
            remaining_to_run = max_trials + n_existing - len(self.experiment.trials)
            while remaining_to_run > 0 and self.run(max_new_trials=remaining_to_run):
                # Not checking `should_abort_optimization` on every iteration for perf.
                # reasons.
                remaining_to_run = max_trials + n_existing - len(self.experiment.trials)

            # Wait for trial evaluations to complete and process results.
            yield self.wait_for_completed_trials_and_report_results()

        # When done scheduling, wait for the remaining trials to finish running
        # (unless optimization is aborting, in which case stop right away).
        if self.running_trials:
            self.logger.info(
                "Done submitting trials, waiting for remaining "
                f"{len(self.running_trials)} running trials..."
            )

        while self.running_trials:
            if self.should_abort_optimization():
                yield self._abort_optimization(num_preexisting_trials=n_existing)
                return

            yield self.wait_for_completed_trials_and_report_results()

        yield self._complete_optimization(num_preexisting_trials=n_existing)
        return

    def run_n_trials(
        self, max_trials: int, timeout_hours: Optional[int] = None
    ) -> OptimizationResult:
        """Run up to ``max_trials`` trials; will run all ``max_trials`` unless
        completion criterion is reached. For base ``Scheduler``, completion criterion
        is reaching total number of trials set in ``SchedulerOptions``, so if that
        option is not specified, this function will run exactly ``max_trials`` trials
        always.

        Args:
            max_trials: Maximum number of trials to run.
            timeout_hours: Limit on length of ths optimization; if reached, the
                optimization will abort even if completon criterion is not yet reached.
        """
        for _ in self.run_trials_and_yield_results(
            max_trials=max_trials, timeout_hours=timeout_hours
        ):
            pass
        return self.summarize_final_result()

    def run_all_trials(self, timeout_hours: Optional[int] = None) -> OptimizationResult:
        """Run all trials until ``completion_criterion`` is reached (by default, completion
        criterion is reaching the ``num_trials`` setting, passed to scheduler on
        instantiation as part of ``SchedulerOptions``).

        NOTE: This function is available only when ``SchedulerOptions.num_trials`` is
        specified.

        Args:
            timeout_hours: Limit on length of ths optimization; if reached, the
                optimization will abort even if completon criterion is not yet reached.
        """
        if self.options.total_trials is None:
            # NOTE: Capping on number of trials will likely be needed as fallback
            # for most stopping criteria, so we ensure `num_trials` is specified.
            raise ValueError(  # pragma: no cover
                "Please either specify `num_trials` in `SchedulerOptions` input "
                "to the `Scheduler` or use `run_n_trials` instead of `run_all_trials`."
            )
        for _ in self.run_trials_and_yield_results(
            max_trials=not_none(self.options.total_trials), timeout_hours=timeout_hours
        ):
            pass
        return self.summarize_final_result()

    def run(self, max_new_trials: int) -> bool:
        """Schedules trial evaluation(s) if stopping criterion is not triggered,
        maximum parallelism is not currently reached, and capacity allows.
        Logs any failures / issues.

        Args:
            max_new_trials: Maximum number of new trials this function should generate
                and run (useful when generating and running trials in batches). Note
                that this function might also re-deploy existing ``CANDIDATE`` trials
                that failed to deploy before, which will not count against this number.

        Returns:
            Boolean representing success status.
        """
        if self.should_consider_optimization_complete():
            self.logger.info(
                "`completion_criterion` is `True`, not running more trials."
            )
            return False

        if self.should_abort_optimization():
            self.logger.info(
                "`should_abort_optimization` is `True`, not running more trials."
            )
            return False

        # Check if capacity allows for running new evaluations and generate as many
        # trials as possible, limited by capacity and model requirements.
        self._sleep_if_too_early_to_poll()
        existing_trials, new_trials = self._prepare_trials(
            max_new_trials=max_new_trials
        )

        if not existing_trials and not new_trials:
            # Unable to gen. new run due to max parallelism limit or need for data
            # or unable to run trials due to lack of capacity.
            if self._optimization_complete:
                return False

            if not self.has_trials_in_flight:
                raise SchedulerInternalError(  # pragma: no cover
                    "No trials are running but model requires more data. This is an "
                    "invalid state of the scheduler, as no more trials can be produced "
                    "but also no more data is expected as there are no running trials."
                    "This should be investigated."
                )
            self._log_next_no_trials_reason = False
            return False  # Nothing to run.

        if existing_trials:
            idcs = sorted(t.index for t in existing_trials)
            self.logger.debug(f"Will run pre-existing candidate trials: {idcs}.")

        all_trials = [*existing_trials, *new_trials]
        idcs = sorted(t.index for t in all_trials)
        contiguous = len(idcs) > 1 and (idcs[-1] - idcs[0] == len(idcs) - 1)
        idcs_str = f"{idcs[0]} - {idcs[-1]}" if contiguous else f"{idcs}"
        self.logger.info(f"Running trials {idcs_str}...")
        # TODO: Add optional timeout between retries of `run_trial(s)`.
        metadata = self.run_trials(trials=all_trials)
        self.logger.debug(f"Ran trials {idcs_str}.")
        if self.options.debug_log_run_metadata:
            self.logger.debug(f"Run metadata: {metadata}.")
        self._latest_trial_start_timestamp = current_timestamp_in_millis()
        self._update_and_save_trials(
            existing_trials=existing_trials, new_trials=new_trials, metadata=metadata
        )
        self._log_next_no_trials_reason = True
        return True

    def _update_status_dict(
        self,
        status_dict: Dict[TrialStatus, Set[int]],
        updating_status_dict: Dict[TrialStatus, Set[int]],
    ) -> Dict[TrialStatus, Set[int]]:
        """Helper method to elements of a dict of sets.

        Avoids leaving trial_index in sets corresponding to two different
        statuses."""
        # Convert Dict[TrialStatus, Set[int]] to Dict[int, TrialStatus]
        trial_index_to_status = {
            trial_index: status
            for status, trial_indices in status_dict.items()
            for trial_index in trial_indices
        }
        # Convert Dict[TrialStatus, Set[int]] to Dict[int, TrialStatus]
        trial_index_to_updating_status = {
            trial_index: status
            for status, trial_indices in updating_status_dict.items()
            for trial_index in trial_indices
        }
        # Safely update new statuses, then convert back to Dict[TrialStatus, Set[int]]
        trial_index_to_status.update(trial_index_to_updating_status)
        updated_status_dict = defaultdict(set)
        for trial_index, status in trial_index_to_status.items():
            updated_status_dict[status].add(trial_index)
        return updated_status_dict

    def poll_and_process_results(self) -> bool:
        """Takes the following actions:
            1. Poll trial runs for their statuses
            2. If any experiment metrics are available while running,
               fetch data for running trials
            3. Determine which trials should be early stopped
            4. Early-stop those trials
            5. Update the experiment with the new trial statuses
            6. Fetch the data for newly completed trials

        Returns:
            A boolean representing whether any trial evaluations completed
            of have been marked as failed or abandoned, changing the number of
            currently running trials.
        """
        self._sleep_if_too_early_to_poll()

        updated_any_trial = False  # Whether any trial updates were performed.
        prev_completed_trial_idcs = set(
            self.experiment.trial_indices_by_status[TrialStatus.COMPLETED]
        )

        # 1. Poll trial statuses
        new_status_to_trial_idcs = self.poll_trial_status()

        # Note: We could use `new_status_to_trial_idcs[TrialStatus.Running]`
        # for the running_trial_indices, but we don't enforce
        # that users return the status of trials that are not being updated.
        # Thus, if a trial was running in the last poll and is still running
        # in this poll, it might not appear in new_status_to_trial_idcs.
        # Instead, to get the list of all currently running trials at this
        # point in time, we look at self.running_trials, which contains trials
        # that were running in the last poll, and we exclude trials that were
        # newly terminated in this poll.
        terminated_trial_idcs = {
            index
            for status, indices in new_status_to_trial_idcs.items()
            if status.is_terminal
            for index in indices
        }
        running_trial_indices = {
            trial.index
            for trial in self.running_trials
            if trial.index not in terminated_trial_idcs
        }

        # 2. If any experiment metrics are available while running,
        #    fetch data for running trials
        already_fetched_trial_idcs = set()
        if any(m.is_available_while_running for m in self.experiment.metrics.values()):
            # Note: Metrics that are *not* available_while_running will be skipped
            # in fetch_trials_data
            self.experiment.fetch_trials_data(trial_indices=running_trial_indices)
            already_fetched_trial_idcs = running_trial_indices

        # 3. Determine which trials to stop early
        stop_trial_info = self.should_stop_trials_early(
            trial_indices=running_trial_indices
        )

        # 4. Stop trials early
        self.stop_trial_runs(
            trials=[self.experiment.trials[trial_idx] for trial_idx in stop_trial_info],
            reasons=list(stop_trial_info.values()),
        )

        # 5. Update trial statuses on the experiment
        new_status_to_trial_idcs = self._update_status_dict(
            status_dict=new_status_to_trial_idcs,
            updating_status_dict={TrialStatus.EARLY_STOPPED: set(stop_trial_info)},
        )
        updated_trials = []
        for status, trial_idcs in new_status_to_trial_idcs.items():
            if status.is_candidate or status.is_deployed:
                # No need to consider candidate, staged or running trials here (none of
                # these trials should actually be candidates, but we can filter on that)
                continue

            if len(trial_idcs) > 0:
                self.logger.info(f"Retrieved {status.name} trials: {trial_idcs}.")
                updated_any_trial = True

            # Update trial statuses and record which trials were updated.
            trials = self.experiment.get_trials_by_indices(trial_idcs)
            for trial in trials:
                trial.mark_as(status=status)

            # 6. Fetch data for newly completed trials
            if status.is_completed:
                newly_completed = (
                    trial_idcs - prev_completed_trial_idcs - already_fetched_trial_idcs
                )
                # Fetch the data for newly completed trials; this will cache the data
                # for all metrics. By pre-caching the data now, we remove the need to
                # fetch it during candidate generation.
                self.experiment.fetch_trials_data(trial_indices=newly_completed)

            updated_trials.extend(trials)

        if not updated_any_trial:  # Did not update anything, nothing to save.
            return False

        self.logger.debug(f"Updating {len(updated_trials)} trials in DB.")
        self._save_or_update_trials_in_db_if_possible(
            experiment=self.experiment,
            trials=updated_trials,
        )
        return updated_any_trial

    def should_stop_trials_early(
        self, trial_indices: Set[int]
    ) -> Dict[int, Optional[str]]:
        """Evaluate whether to early-stop running trials.

        Args:
            trial_indices: Indices of trials to consider for early stopping.

        Returns:
            A set of indices of trials to early-stop (will be a subset of
            initially-passed trials).
        """
        if self.options.early_stopping_strategy is None:
            return {}

        early_stopping_strategy = not_none(self.options.early_stopping_strategy)
        return early_stopping_strategy.should_stop_trials_early(
            trial_indices=trial_indices, experiment=self.experiment
        )

    def _abort_optimization(self, num_preexisting_trials: int) -> Dict[str, Any]:
        """Conclude optimization without waiting for anymore running trials and
        return results so far via `report_results`.
        """
        self._record_optimization_complete_message()
        self._record_run_trials_status(
            num_preexisting_trials=num_preexisting_trials,
            status=RunTrialsStatus.ABORTED,
        )
        return self.report_results()

    def _complete_optimization(self, num_preexisting_trials: int) -> Dict[str, Any]:
        """Conclude optimization with waiting for anymore running trials and
        return final results via `wait_for_completed_trials_and_report_results`.
        """
        self._record_optimization_complete_message()
        res = self.wait_for_completed_trials_and_report_results()
        # raise an error if the failure rate exceeds tolerance at the end of the sweep
        self.error_if_failure_rate_exceeded(force_check=True)
        self._record_run_trials_status(
            num_preexisting_trials=num_preexisting_trials,
            status=RunTrialsStatus.SUCCESS,
        )
        return res

    def _validate_options(self, options: SchedulerOptions) -> None:
        """Validates `SchedulerOptions` for compatibility with given
        `Scheduler` class.
        """
        if options.trial_type is BatchTrial:  # TODO[T61776778]: support batches
            raise NotImplementedError("Support for batched trials coming soon.")
        if not (0.0 <= options.tolerated_trial_failure_rate < 1.0):
            raise ValueError("`tolerated_trial_failure_rate` must be in [0, 1).")
        if options.early_stopping_strategy is not None and not any(
            m.is_available_while_running() for m in self.experiment.metrics.values()
        ):
            raise ValueError(
                "Can only specify an early stopping strategy if at least one metric "
                "is marked as `is_available_while_running`. Otherwise, we will be "
                "unable to fetch intermediate results with which to evaluate "
                "early stopping criteria."
            )

    def _prepare_trials(
        self, max_new_trials: int
    ) -> Tuple[List[BaseTrial], List[BaseTrial]]:
        """Prepares one trial or multiple trials for deployment, based on
        whether `run_trials_in_batches` is set to `True` in this scheduler's
        options.

        NOTE: If running trials in batches, exact number of trials run at once
        is determined by available capacity and generation strategy's
        requirement for more data and parallelism limitation.

        Args:
            max_new_trials: Maximum number of new trials to generate.

        Returns:
            Two lists of trials:
            - list of existing candidate trials whose deployment was attempted
              but failed before (empty if there were no such trials),
            - list of new candidate trials that were created in the course of
              this function (empty if no new trials were generated).
        """
        if self.options.run_trials_in_batches:
            n = self.poll_available_capacity()
            if n is None:
                raise UnsupportedError(
                    "Running trials in batches is supported only if "
                    "`poll_available_capacity` returns a non-null value."
                )
            if self.options.total_trials:
                n = min(
                    n,
                    not_none(self.options.total_trials)
                    - len(self.experiment.trials_expecting_data),
                )
        else:  # Running 1 trial at a time, sequentially.
            n = 1 if self.has_capacity() else 0
        if n < 1:
            self.logger.debug("There is no capacity to run any trials.")
        existing = self.candidate_trials[:n]
        n_new = min(n - len(existing), max_new_trials)
        new = self._get_next_trials(num_trials=n_new) if n_new > 0 else []
        return existing, new

    def _get_next_trials(self, num_trials: int = 1) -> List[BaseTrial]:
        """Produce up to `num_trials` new generator runs from the underlying
        generation strategy and create new trials with them. Logs errors
        encountered during generation.

        NOTE: Fewer than `num_trials` trials may be produced if generation
        strategy runs into its parallelism limit or needs more data to proceed.

        Returns:
            List of trials, empty if generation is not possible.
        """
        pending = get_pending_observation_features_based_on_trial_status(
            experiment=self.experiment
        )
        try:
            generator_runs = self.generation_strategy._gen_multiple(
                experiment=self.experiment,
                num_generator_runs=num_trials,
                pending_observations=pending,
            )
        except OptimizationComplete as err:
            self.logger.info(f"Optimization complete: {err}.")
            self._optimization_complete = True
            return []
        except DataRequiredError as err:
            # TODO[T62606107]: consider adding a `more_data_required` property to
            # check to generation strategy to avoid running into this exception.
            if self._log_next_no_trials_reason:
                self.logger.info(
                    "Generated all trials that can be generated currently. "
                    "Model requires more data to generate more trials."
                )
            self.logger.debug(f"Message from generation strategy: {err}")
            return []
        except MaxParallelismReachedException as err:
            # TODO[T62606107]: consider adding a `step_max_parallelism_reached`
            # check to generation strategy to avoid running into this exception.
            if self._log_next_no_trials_reason:
                self.logger.info(
                    "Generated all trials that can be generated currently. "
                    "Max parallelism currently reached."
                )
            self.logger.debug(f"Message from generation strategy: {err}")
            return []

        if self.options.trial_type is Trial and len(generator_runs[0].arms) > 1:
            raise SchedulerInternalError(
                "Generation strategy produced multiple arms when only one was expected."
            )

        return [
            self.experiment.new_batch_trial(
                generator_run=generator_run,
                ttl_seconds=self.options.ttl_seconds_for_trials,
            )
            if self.options.trial_type is BatchTrial
            else self.experiment.new_trial(
                generator_run=generator_run,
                ttl_seconds=self.options.ttl_seconds_for_trials,
            )
            for generator_run in generator_runs
        ]

    def _update_and_save_trials(
        self,
        existing_trials: List[BaseTrial],
        new_trials: List[BaseTrial],
        metadata: Dict[int, Dict[str, Any]],
    ) -> None:
        """Updates trials with new run metadata and status; saves updates to DB.

        Args:
            exiting_trials: Trials that existed on this experiment during the
                previous call to this function (these are trials, deployment of
                which has already been attempted but failed, so we are
                re-attempting it; these trials are already saved in DB if using
                storage functionality).
            new_trials: Trials that were newly created (these trials are not
                yet saved in the DB if using storage functionality).
            metadata: Run metadata for the trials, from `scheduler.run_trials`.
                Format is {trial index -> trial run metadata}. Trials present in
                the metadata dict will be considered `RUNNING`, and the rest of
                trials in `existing_trials` or `new_trials` (that are not present
                in `metadata`) will be left as `CANDIDATE`.
        """

        def _process_trial(trial):
            if trial.index in metadata:
                trial.update_run_metadata(metadata=metadata[trial.index])
                trial.mark_running(no_runner_required=True)
            else:
                self.logger.debug(
                    f"Trial {trial.index} did not deploy, status: {trial.status}."
                )

        new_generator_runs = []
        for trial in existing_trials:
            _process_trial(trial)
        for trial in new_trials:
            new_generator_runs.extend(trial.generator_runs)
            _process_trial(trial)

        self._save_or_update_trials_and_generation_strategy_if_possible(
            experiment=self.experiment,
            trials=[*existing_trials, *new_trials],
            generation_strategy=self.generation_strategy,
            new_generator_runs=new_generator_runs,
        )

    def _sleep_if_too_early_to_poll(self) -> None:
        """Wait to query for capacity unless there has been enough time since last
        scheduling.
        """
        if self._latest_trial_start_timestamp is not None:
            seconds_since_run_trial = (
                current_timestamp_in_millis()
                - not_none(self._latest_trial_start_timestamp)
            ) * 1000
            if seconds_since_run_trial < self.options.min_seconds_before_poll:
                sleep(self.options.min_seconds_before_poll - seconds_since_run_trial)

    def _set_logger(self) -> None:
        """Set up the logger with appropriate logging levels."""
        cls_name = self.__class__.__name__
        logger = get_logger(name=f"{__name__}.{cls_name}@{hex(id(self))}")
        set_stderr_log_level(self.options.logging_level)
        if self.options.log_filepath is not None:
            handler = build_file_handler(
                filepath=not_none(self.options.log_filepath),
                level=self.options.logging_level,
            )
            logger.addHandler(handler)
        self.logger = LoggerAdapter(logger, extra={"output_name": cls_name})

    def _validate_remaining_trials(self, experiment: Experiment) -> None:
        """Check how many trials are remaining in `total_trials` given the trials
        already on experiment and make sure that there will be trials for the
        scheduler to run.
        """
        if not experiment.trials or not self.options.total_trials:
            return

        total_trials = not_none(self.options.total_trials)
        preexisting = len(experiment.trials)
        msg = (
            f"{experiment} already has {preexisting} trials associated with it. "
            f"Total trials setting for this scheduler is {total_trials}, so "
        )
        if preexisting >= total_trials:
            self.logger.warning(
                msg + "no more trials would be run in this scheduler if "
                "`Scheduler.run_all_trials` is called (but you can still use "
                "`Scheduler.run_n_trials` to run a fixed number of trials)."
            )
        else:
            self.logger.info(
                msg + "number of trials ran by `Scheduler.run_all_trials` would be "
                f"{total_trials - preexisting}."
            )

    def _validate_implemented_metrics(self, experiment: Experiment) -> None:
        """Ensure that the experiment specifies metrics and that they are not base
        `Metric`-s, which do not implement fetching logic.
        """
        msg = (
            "`Scheduler` requires that experiment specifies metrics "
            "with implemented fetching logic."
        )
        if not experiment.metrics:
            raise UnsupportedError(msg)
        else:
            base_metrics = {
                m_name for m_name, m in experiment.metrics.items() if type(m) == Metric
            }
            if base_metrics:
                msg += f" Metrics {base_metrics} do not implement fetching logic."
                raise UnsupportedError(msg)

    def _enforce_immutable_search_space_and_opt_config(self) -> None:
        """Experiments with immutable search space and optimization config don't
        need to keep copies of those objects on each generator run in the experiment,
        resulting in large performance gain in storage layer. In `Scheduler`, we
        force-set this immutability on `Experiment`, since scheduler experiments
        are typically not human-in-the-loop.
        """
        if self.experiment.immutable_search_space_and_opt_config:
            return

        self.logger.info(
            f"`Scheduler` requires experiment to have immutable search "
            "space and optimization config. Setting property "
            f"{Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF.value} "
            "to `True` on experiment."
        )
        self.experiment._properties[
            Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF.value
        ] = True

    def _initialize_experiment_status_properties(self) -> None:
        """Initializes status-tracking properties of the experiment, which will
        be appended to in ``run_trials_and_yield_results``."""
        for status_prop_enum_member in ExperimentStatusProperties:
            if status_prop_enum_member not in self.experiment._properties:
                self.experiment._properties[status_prop_enum_member.value] = []

    def _record_run_trials_status(
        self, num_preexisting_trials: Optional[int], status: RunTrialsStatus
    ) -> None:
        """Records status of each call to ``Scheduler.run_trials_and_yield_results``
        in properties of this experiment for monitoring of experiment success.
        """
        to_append: Dict[str, Any] = {
            ExperimentStatusProperties.RUN_TRIALS_STATUS.value: status.value
        }
        if num_preexisting_trials is not None:
            new_trials = len(self.experiment.trials) - num_preexisting_trials
            to_append[
                ExperimentStatusProperties.NUM_TRIALS_RUN_PER_CALL.value
            ] = new_trials
        self._append_to_experiment_properties(to_append=to_append)

    def _record_optimization_complete_message(self) -> None:
        """Adds a simple optimization completion message to this scheduler's markdown
        messages.
        """
        self.markdown_messages[
            "optimization_completion"
        ] = OPTIMIZATION_COMPLETION_MSG.format(
            num_trials=len(self.experiment.trials),
            experiment_name=self.experiment.name
            if self.experiment._name is not None
            else "unnamed",
        )

    def _append_to_experiment_properties(self, to_append: Dict[str, Any]) -> None:
        """Appends to list fields in experiment properties based on ``to_append``
        input dict of form {property_name: value_to_append}.
        """
        for prop, val_to_append in to_append.items():
            if prop in self.experiment._properties:
                self.experiment._properties[prop].append(val_to_append)
            else:
                self.experiment._properties[prop] = [val_to_append]
        self._update_experiment_properties_in_db(
            experiment_with_updated_properties=self.experiment
        )

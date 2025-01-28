#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from logging import LoggerAdapter
from time import sleep
from typing import Any, cast, NamedTuple, Optional

import ax.service.utils.early_stopping as early_stopping_utils
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.multi_type_experiment import (
    filter_trials_by_type,
    get_trial_indices_for_statuses,
    MultiTypeExperiment,
)
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.runner import Runner
from ax.core.types import TModelPredictArm, TParameterization
from ax.core.utils import get_pending_observation_features_based_on_trial_status

from ax.exceptions.core import (
    AxError,
    DataRequiredError,
    OptimizationComplete,
    UnsupportedError,
    UserInputError,
)
from ax.exceptions.generation_strategy import (
    AxGenerationException,
    MaxParallelismReachedException,
    OptimizationConfigRequired,
)
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.modelbridge_utils import get_fixed_features_from_experiment
from ax.service.utils.analysis_base import AnalysisBase
from ax.service.utils.best_point_mixin import BestPointMixin
from ax.service.utils.scheduler_options import SchedulerOptions, TrialType
from ax.service.utils.with_db_settings_base import DBSettings, WithDBSettingsBase
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.executils import retry_on_exception
from ax.utils.common.logger import (
    build_file_handler,
    get_logger,
    make_indices_str,
    set_ax_logger_levels,
)
from ax.utils.common.timeutils import current_timestamp_in_millis
from pyre_extensions import assert_is_instance, none_throws


NOT_IMPLEMENTED_IN_BASE_CLASS_MSG = """ \
This method is not implemented in the base `Scheduler` class. \
If this functionality is desired, specify the method in the \
scheduler subclass.
"""
GS_TYPE_MSG = "This optimization run uses a '{gs_name}' generation strategy."
OPTIMIZATION_COMPLETION_MSG = """Optimization completed with total of {num_trials} \
trials attached to the underlying Ax experiment '{experiment_name}'.
"""
FAILURE_EXCEEDED_MSG = (
    "Failure rate exceeds the tolerated trial failure rate of {f_rate} (at least "
    "{n_failed} out of first {n_ran} trials failed or were abandoned). Checks are "
    "triggered both at the end of a optimization and if at least {min_failed} trials "
    "have either failed, or have been abandoned, potentially automatically due to "
    "issues with the trial."
)


# Wait time b/w reports will not exceed 15 mins.
MAX_SECONDS_BETWEEN_REPORTS = 900


class OptimizationResult(NamedTuple):  # TODO[T61776778]
    pass  # TBD


class SchedulerInternalError(AxError):
    """Error that indicates an error within the `Scheduler` logic."""

    pass


class FailureRateExceededError(AxError):
    """Error that indicates the optimization was aborted due to excessive
    failure rate.
    """

    pass


NO_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    cast(type[Exception], SchedulerInternalError),
    cast(type[Exception], NotImplementedError),
    cast(type[Exception], UnsupportedError),
)


class OutputPriority(IntEnum):
    """Priority of a message. Messages with higher priority will be shown first, and
    messages with the same priority will be sorted alphabetically."""

    NOTSET = 0
    DEBUG = 10
    INFO = 20
    TOPLINE = 30
    WARNING = 40
    ERROR = 50


@dataclass
class MessageOutput:
    """Message to be shown in the output of the scheduler."""

    text: str
    priority: OutputPriority | int

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"MessageOutput(text={self.text}, priority={self.priority})"

    def append(self, text: str) -> None:
        """Append text to the text of an existing message."""
        self.text += text


class Scheduler(AnalysisBase, BestPointMixin):
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
    generation_strategy: GenerationStrategyInterface
    # pyre-fixme[24]: Generic type `LoggerAdapter` expects 1 type parameter.
    logger: LoggerAdapter
    # Mapping of form {short string identifier -> message to show in reported
    # results}. This is a mapping and not a list to allow for changing of
    # some optimization messages throughout the course of the optimization
    # (e.g. progress report of the optimization).
    markdown_messages: dict[str, MessageOutput]

    # Number of trials that existed on the scheduler's experiment before
    # the scheduler instantiation with that experiment.
    _num_preexisting_trials: int
    # Number of trials remaining to be scheduled during run_trials_and_yield_results.
    # Saved as a property so that it can be accessed after optimization is complex (ex.
    # for global stopping saving calculation).
    _num_remaining_requested_trials: int = 0
    # Total number of MetricFetchEs encountered during the course of optimization. Note
    # this is different from and may be greater than the number of trials that have
    # been marked either FAILED or ABANDONED due to metric fetching errors.
    _num_metric_fetch_e_encountered: int = 0
    # Number of trials that have been marked either FAILED or ABANDONED due to
    # MetricFetchE being encountered during _fetch_and_process_trials_data_results
    _num_trials_bad_due_to_err: int = 0
    # Keeps track of whether the allowed failure rate has been exceeded during
    # the optimization. If true, allows any pending trials to finish and raises
    # an error through self._complete_optimization.
    _failure_rate_has_been_exceeded: bool = False
    # Timestamp of last optimization start time (milliseconds since Unix epoch);
    # recorded in each `run_n_trials`.
    _latest_optimization_start_timestamp: int | None = None
    # Timeout setting for current optimization.
    _timeout_hours: float | None = None
    # Timestamp of when the last deployed trial started running.
    _latest_trial_start_timestamp: float | None = None
    # Will be set to `True` if generation strategy signals that the optimization
    # is complete, in which case the optimization should gracefully exit early.
    _optimization_complete: bool = False
    # This will disable the global stopping strategy. It is useful in some
    # applications where the user wants to run the optimization loop to exhaust
    # the declared number of trials.
    __ignore_global_stopping_strategy: bool = False
    # Default kwargs passed when fetching data if not overridden on `SchedulerOptions`
    DEFAULT_FETCH_KWARGS = {
        "overwrite_existing_data": True,
    }

    def __init__(
        self,
        experiment: Experiment,
        generation_strategy: GenerationStrategyInterface,
        options: SchedulerOptions,
        db_settings: Optional[DBSettings] = None,
        _skip_experiment_save: bool = False,
    ) -> None:
        self.experiment = experiment
        # Set up logger with an optional filepath handler. Note: we set the
        # logger before setting options since that can trigger errors.
        self._set_logger(options=options)
        self.options = options
        # NOTE: Parallelism schedule is embedded in the generation
        # strategy, as `GenerationStep.max_parallelism`.
        self.generation_strategy = generation_strategy

        if not isinstance(experiment, Experiment):
            raise TypeError("{experiment} is not an Ax experiment.")
        if not isinstance(generation_strategy, GenerationStrategyInterface):
            raise TypeError("{generation_strategy} is not a generation strategy.")

        # Initialize storage layer for the scheduler.
        super().__init__(
            db_settings=db_settings,
            logging_level=self.options.logging_level,
            suppress_all_errors=self.options.suppress_storage_errors_after_retries,
        )

        # Validate experiment and GS; ensure that experiment has immutable
        # search space and opt. config to avoid storing their  copies on each
        # generator run.
        self._validate_remaining_trials(experiment=experiment)
        if self.options.enforce_immutable_search_space_and_opt_config:
            self._enforce_immutable_search_space_and_opt_config()

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
            "Generation strategy": MessageOutput(
                text=GS_TYPE_MSG.format(gs_name=generation_strategy.name),
                priority=OutputPriority.DEBUG,
            ),
        }

    @classmethod
    def get_default_db_settings(cls) -> DBSettings:
        raise NotImplementedError(
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
        generation_strategy: GenerationStrategy | None = None,
        reduced_state: bool = True,
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
            experiment_name=experiment_name,
            reduced_state=reduced_state,
        )
        if db_settings:
            kwargs = {**kwargs, "db_settings": db_settings}
        if not exp:
            raise ValueError(f"Experiment {experiment_name} not found.")

        if not gs and not generation_strategy:
            raise ValueError(
                f"Experiment {experiment_name} did not have a generation "
                "strategy associated with it in the database, so a new "
                "generation strategy must be provided as argument to "
                "`Scheduler.from_stored_experiment`."
            )

        if gs and generation_strategy and gs != generation_strategy:
            # NOTE: In the future we may want to allow overriding of GS,
            # in which case we can add a flag to this function and allow
            # the override with warning.
            raise UnsupportedError(
                "Experiment was associated with generation strategy "
                f"{gs.name} in DB, but a new generation strategy "
                f"{generation_strategy.name} was provided. To use "
                "the generation strategy currently in DB, do not "
                "specify the `geneneration_strategy` kwarg."
            )

        scheduler = cls(
            experiment=exp,
            generation_strategy=none_throws(generation_strategy or gs),
            options=options,
            # No need to resave the experiment we just reloaded.
            _skip_experiment_save=True,
            # NOTE: `kwargs` can include `db_settings` if those were
            # provided to this function.
            **kwargs,
        )
        return scheduler

    @property
    def options(self) -> SchedulerOptions:
        """Scheduler options."""
        return self._options  # pyre-ignore [16]

    @options.setter
    def options(self, options: SchedulerOptions) -> None:
        """Set scheduler options."""
        self._validate_options(options=options)
        self._options = options
        # validate runners and metrics since validate_metrics is an option
        self._validate_runner_and_implemented_metrics(experiment=self.experiment)

    @property
    def trial_type(self) -> str | None:
        """Trial type for the experiment this scheduler is running.

        This returns None if the experiment is not a MultitypeExperiment

        Returns:
            Trial type for the experiment this scheduler is running if the
            experiment is a MultiTypeExperiment and None otherwise.
        """
        if isinstance(self.experiment, MultiTypeExperiment):
            return self.options.mt_experiment_trial_type
        return None

    @property
    def running_trials(self) -> list[BaseTrial]:
        """Currently running trials.

        Note: if the experiment is a MultiTypeExperiment, then this will
        only fetch trials of type `Scheduler.trial_type`.


        Returns:
            List of trials that are currently running.
        """
        return filter_trials_by_type(
            trials=self.experiment.trials_by_status[TrialStatus.RUNNING],
            trial_type=self.trial_type,
        )

    @property
    def trials(self) -> list[BaseTrial]:
        """All trials.

        Note: if the experiment is a MultiTypeExperiment, then this will
        only fetch trials of type `Scheduler.trial_type`.

        Returns:
            List of trials that are currently running.
        """
        return filter_trials_by_type(
            trials=list(self.experiment.trials.values()), trial_type=self.trial_type
        )

    @property
    def running_trial_indices(self) -> set[int]:
        """Currently running trials.

        Returns:
            List of trials that are currently running.
        """
        return get_trial_indices_for_statuses(
            experiment=self.experiment,
            statuses={TrialStatus.RUNNING},
            trial_type=self.trial_type,
        )

    @property
    def failed_abandoned_trial_indices(self) -> set[int]:
        """Failed or abandoned trials.

        Note: if the experiment is a MultiTypeExperiment, then this will
        only fetch trials of type `Scheduler.trial_type`.

        Returns:
            List of trials that are currently running.
        """
        return get_trial_indices_for_statuses(
            experiment=self.experiment,
            statuses={TrialStatus.ABANDONED, TrialStatus.FAILED},
            trial_type=self.trial_type,
        )

    @property
    def pending_trials(self) -> list[BaseTrial]:
        """Running or staged trials on the experiment this scheduler is
        running.

        Note: if the experiment is a MultiTypeExperiment, then this will
        only fetch trials of type `Scheduler.trial_type`.

        Returns:
            List of trials that are currently running or staged.
        """
        staged_trials = filter_trials_by_type(
            trials=self.experiment.trials_by_status[TrialStatus.STAGED],
            trial_type=self.trial_type,
        )
        return self.running_trials + staged_trials

    @property
    def candidate_trials(self) -> list[BaseTrial]:
        """Candidate trials on the experiment this scheduler is running.

        Note: if the experiment is a MultiTypeExperiment, then this will
        only fetch trials of type `Scheduler.trial_type`.

        Returns:
            List of trials that are currently candidates.
        """
        return filter_trials_by_type(
            trials=self.experiment.trials_by_status[TrialStatus.CANDIDATE],
            trial_type=self.trial_type,
        )

    @property
    def trials_expecting_data(self) -> list[BaseTrial]:
        """Trials expecting data.

        Note: if the experiment is a MultiTypeExperiment, then this will
        only fetch trials of type `Scheduler.trial_type`.
        """
        trials = []
        for trial in self.experiment.trials.values():
            if trial.status.expecting_data:
                if self.trial_type is None or trial.trial_type == self.trial_type:
                    trials.append(trial)
        return trials

    @property
    def runner(self) -> Runner:
        """``Runner`` specified on the experiment associated with this ``Scheduler``
        instance.
        """
        if self.trial_type is not None:
            runner = assert_is_instance(
                self.experiment, MultiTypeExperiment
            ).runner_for_trial_type(trial_type=none_throws(self.trial_type))
        else:
            runner = self.experiment.runner
        if runner is None:
            raise UnsupportedError(
                "`Scheduler` requires that experiment specifies a `Runner`."
            )
        return runner

    @property
    def standard_generation_strategy(self) -> GenerationStrategy:
        """Used for operations in the scheduler that can only be done with
        and instance of ``GenerationStrategy``.
        """
        gs = self.generation_strategy
        if not isinstance(gs, GenerationStrategy):
            raise NotImplementedError(
                "This functionality is only supported with instances of "
                "`GenerationStrategy` (one that uses `GenerationStrategy` "
                "class) and not yet with other types of "
                "`GenerationStrategyInterface`."
            )
        return gs

    def __repr__(self) -> str:
        """Short user-friendly string representation."""
        if not hasattr(self, "experiment"):
            # Experiment, generation strategy, etc. attributes have not
            # yet been set.
            return f"{self.__class__.__name__}"
        return (
            f"{self.__class__.__name__}(experiment={self.experiment}, "
            f"generation_strategy={self.generation_strategy}, options="
            f"{self.options})"
        )

    # ---------- Methods below should generally not be modified in subclasses! ---------
    # ---------- I. Methods that are often called outside the `Scheduler`. ---------

    def generate_candidates(
        self,
        num_trials: int = 1,
        reduce_state_generator_runs: bool = False,
        remove_stale_candidates: bool = False,
    ) -> tuple[list[BaseTrial], Exception | None]:
        """Fetch the latest data and generate new candidate trials.

        Args:
            num_trials: Number of candidate trials to generate.
            reduce_state_generator_runs: Flag to determine
                whether to save model state for every generator run (default)
                or to only save model state on the final generator run of each
                batch.
            remove_stale_candidates: If true, mark any existing candidate trials
                failed before trial generation because:
                - they should not be treated as pending points
                - they will no longer be relevant

        Returns:
            List of trials, empty if generation is not possible.
        """
        if remove_stale_candidates:
            stale_candidate_trials = self.experiment.trials_by_status[
                TrialStatus.CANDIDATE
            ]
            self.logger.info(
                "Marking the following trials as failed because they are stale: "
                f"{[t.index for t in stale_candidate_trials]}"
            )
            for trial in stale_candidate_trials:
                trial.mark_failed(reason="Newer candidates generated.", unsafe=True)
        else:
            stale_candidate_trials = []
        new_trials, err = self._get_next_trials(
            num_trials=num_trials,
            n=self.options.batch_size,
        )
        if len(new_trials) > 0:
            new_generator_runs = [gr for t in new_trials for gr in t.generator_runs]
            self._save_or_update_trials_and_generation_strategy_if_possible(
                experiment=self.experiment,
                trials=new_trials + stale_candidate_trials,
                generation_strategy=self.generation_strategy,
                new_generator_runs=new_generator_runs,
                reduce_state_generator_runs=reduce_state_generator_runs,
            )
        return new_trials, err

    def run_n_trials(
        self,
        max_trials: int,
        ignore_global_stopping_strategy: bool = False,
        timeout_hours: float | None = None,
        idle_callback: Optional[Callable[[Scheduler], None]] = None,
    ) -> OptimizationResult:
        """Run up to ``max_trials`` trials; will run all ``max_trials`` unless
        completion criterion is reached. For base ``Scheduler``, completion criterion
        is reaching total number of trials set in ``SchedulerOptions``, so if that
        option is not specified, this function will run exactly ``max_trials`` trials
        always.

        Args:
            max_trials: Maximum number of trials to run.
            ignore_global_stopping_strategy: If set, Scheduler will skip the global
                stopping strategy in ``should_consider_optimization_complete``.
            timeout_hours: Limit on length of ths optimization; if reached, the
                optimization will abort even if completon criterion is not yet reached.
            idle_callback: Callable that takes a Scheduler instance as an argument to
                deliver information while the trials are still running. Any output of
                `idle_callback` will not be returned, so `idle_callback` must expose
                information in some other way. For example, it could print something
                about the state of the scheduler or underlying experiment to STDOUT,
                write something to a database, or modify a Plotly figure or other object
                in place. `ax.service.utils.report_utils.get_figure_and_callback` is a
                helper function for generating a callback that will update a Plotly
                figure.

        Example:
            >>> trials_info = {"n_completed": None}
            >>>
            >>> def write_n_trials(scheduler: Scheduler) -> None:
            ...     trials_info["n_completed"] = len(scheduler.experiment.trials)
            >>>
            >>> scheduler.run_n_trials(
            ...     max_trials=3, idle_callback=write_n_trials
            ... )
            >>> print(trials_info["n_completed"])
            3
        """
        self.poll_and_process_results()
        for _ in self.run_trials_and_yield_results(
            max_trials=max_trials,
            ignore_global_stopping_strategy=ignore_global_stopping_strategy,
            timeout_hours=timeout_hours,
            idle_callback=idle_callback,
        ):
            pass
        return self.summarize_final_result()

    def run_all_trials(
        self,
        timeout_hours: float | None = None,
        idle_callback: Optional[Callable[[Scheduler], None]] = None,
    ) -> OptimizationResult:
        """Run all trials until ``should_consider_optimization_complete`` yields
        true (by default, ``should_consider_optimization_complete`` will yield true when
        reaching the ``num_trials`` setting, passed to scheduler on instantiation as
        part of ``SchedulerOptions``).

        NOTE: This function is available only when ``SchedulerOptions.num_trials`` is
        specified.

        Args:
            timeout_hours: Limit on length of ths optimization; if reached, the
                optimization will abort even if completon criterion is not yet reached.
            idle_callback: Callable that takes a Scheduler instance as an argument to
                deliver information while the trials are still running. Any output of
                `idle_callback` will not be returned, so `idle_callback` must expose
                information in some other way. For example, it could print something
                about the state of the scheduler or underlying experiment to STDOUT,
                write something to a database, or modify a Plotly figure or other object
                in place. `ax.service.utils.report_utils.get_figure_and_callback` is a
                helper function for generating a callback that will update a Plotly
                figure.

        Example:
            >>> trials_info = {"n_completed": None}
            >>>
            >>> def write_n_trials(scheduler: Scheduler) -> None:
            ...     trials_info["n_completed"] = len(scheduler.experiment.trials)
            >>>
            >>> scheduler.run_all_trials(
            ...     timeout_hours=0.1, idle_callback=write_n_trials
            ... )
            >>> print(trials_info["n_completed"])
        """
        if self.options.total_trials is None:
            # NOTE: Capping on number of trials will likely be needed as fallback
            # for most stopping criteria, so we ensure `num_trials` is specified.
            raise ValueError(
                "Please either specify `num_trials` in `SchedulerOptions` input "
                "to the `Scheduler` or use `run_n_trials` instead of `run_all_trials`."
            )
        return self.run_n_trials(
            max_trials=none_throws(self.options.total_trials),
            timeout_hours=timeout_hours,
            idle_callback=idle_callback,
        )

    def run_trials_and_yield_results(
        self,
        max_trials: int,
        ignore_global_stopping_strategy: bool = False,
        timeout_hours: float | None = None,
        idle_callback: Callable[[Scheduler], None] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Make continuous calls to `run` and `process_results` to run up to
        ``max_trials`` trials, until completion criterion is reached. This is the 'main'
        method of a ``Scheduler``.

        Args:
            max_trials: Maximum number of trials to run in this generator. The
                generator will run trials until a completion criterion is reached,
                a completion signal is received from the generation strategy, or
                ``max_trials`` trials have been run (whichever happens first).
            ignore_global_stopping_strategy: If set, Scheduler will skip the global
                stopping strategy in ``should_consider_optimization_complete``.
            timeout_hours: Maximum number of hours, for which
                to run the optimization. This function will abort after running
                for `timeout_hours` even if stopping criterion has not been reached.
                If set to `None`, no optimization timeout will be applied.
            idle_callback: Callable that takes a Scheduler instance as an argument to
                deliver information while the trials are still running. Any output of
                `idle_callback` will not be returned, so `idle_callback` must expose
                information in some other way. For example, it could print something
                about the state of the scheduler or underlying experiment to STDOUT,
                write something to a database, or modify a Plotly figure or other object
                in place. `ax.service.utils.report_utils.get_figure_and_callback` is a
                helper function for generating a callback that will update a Plotly
                figure.
        """
        if max_trials < 0:
            raise ValueError(f"Expected `max_trials` >= 0, got {max_trials}.")

        if timeout_hours is not None:
            if timeout_hours < 0:
                raise UserInputError(
                    f"Expected `timeout_hours` >= 0, got {timeout_hours}."
                )

        self._latest_optimization_start_timestamp = current_timestamp_in_millis()
        self.__ignore_global_stopping_strategy = ignore_global_stopping_strategy

        n_initial_candidate_trials = len(self.candidate_trials)
        if n_initial_candidate_trials == 0 and max_trials < 0:
            raise UserInputError(f"Expected `max_trials` >= 0, got {max_trials}.")

        # trials are pre-existing only if they do not still require running
        n_existing = len(self.trials) - n_initial_candidate_trials

        # Until completion criterion is reached or `max_trials` is scheduled,
        # schedule new trials and poll existing ones in a loop.
        self._num_remaining_requested_trials = max_trials
        while (
            self._num_remaining_requested_trials > 0
            and not self.should_consider_optimization_complete()[0]
        ):
            if self.should_abort_optimization(timeout_hours=timeout_hours):
                yield self._abort_optimization(num_preexisting_trials=n_existing)
                return

            # Run new trial evaluations until `run` returns `False`, which
            # means that there was a reason not to run more evaluations yet.
            # Also check that `max_trials` is not reached to not exceed it.
            n_remaining_to_generate = self._num_remaining_requested_trials - len(
                self.candidate_trials
            )
            while self._num_remaining_requested_trials > 0 and self.run(
                max_new_trials=n_remaining_to_generate,
                timeout_hours=timeout_hours,
            ):
                # Not checking `should_abort_optimization` on every trial for perf.
                # reasons.
                n_already_run_by_scheduler = (
                    len(self.trials) - n_existing - len(self.candidate_trials)
                )
                self._num_remaining_requested_trials = (
                    max_trials - n_already_run_by_scheduler
                )
                n_remaining_to_generate = self._num_remaining_requested_trials - len(
                    self.candidate_trials
                )
            # this is safeguard in case no trial statuses have been updated, and
            # wait_for_running_trials=False, in which case we do not want to continue
            # to loop and poll
            report_results = self._check_exit_status_and_report_results(
                n_existing=n_existing, idle_callback=idle_callback, force_refit=False
            )
            if report_results is None:
                return
            else:
                yield report_results

        # When done scheduling, wait for the remaining trials to finish running
        # (unless optimization is aborting, in which case stop right away).
        if self.running_trials:
            self.logger.info(
                "Done submitting trials, waiting for remaining "
                f"{len(self.running_trials)} running trials..."
            )

        while self.running_trials:
            if self.should_abort_optimization(timeout_hours=timeout_hours):
                yield self._abort_optimization(num_preexisting_trials=n_existing)
                return
            report_results = self._check_exit_status_and_report_results(
                n_existing=n_existing, idle_callback=idle_callback, force_refit=True
            )
            if report_results is None:
                return
            else:
                yield report_results

        yield self._complete_optimization(
            num_preexisting_trials=n_existing, idle_callback=idle_callback
        )
        return

    # ---------- II. Methods that are typically called within the `Scheduler`. ---------

    @retry_on_exception(retries=3, no_retry_on_exception_types=NO_RETRY_EXCEPTIONS)
    def run_trials(self, trials: Iterable[BaseTrial]) -> dict[int, dict[str, Any]]:
        """Deployment function, runs a single evaluation for each of the
        given trials.

        Override default implementation on the ``Runner`` if its desirable to deploy
        trials in bulk.

        NOTE: the `retry_on_exception` decorator applied to this function should also
        be applied to its subclassing override if one is provided and retry behavior
        is desired.

        Args:
            trials: Iterable of trials to be deployed, each containing arms with
                parameterizations to be evaluated. Can be a ``Trial``
                if contains only one arm or a ``BatchTrial`` if contains
                multiple arms.

        Returns:
            Dict of trial index to the run metadata of that trial from the deployment
            process.
        """
        return self.runner.run_multiple(trials=trials)

    @retry_on_exception(retries=3, no_retry_on_exception_types=NO_RETRY_EXCEPTIONS)
    def poll_trial_status(
        self, poll_all_trial_statuses: bool = False
    ) -> dict[TrialStatus, set[int]]:
        """Polling function, checks the status of any non-terminal trials
        and returns their indices as a mapping from TrialStatus to a list of indices.

        NOTE: Does not need to handle waiting between polling while trials
        are running; that logic is handled in ``Scheduler.poll``, which calls
        this function.

        Returns:
            A dictionary mapping TrialStatus to a list of trial indices that have
            the respective status at the time of the polling. This does not need to
            include trials that at the time of polling already have a terminal
            (ABANDONED, FAILED, COMPLETED) status (but it may).
        """
        trials = (
            list(self.experiment.trials.values())
            if poll_all_trial_statuses
            else self.pending_trials
        )
        trials = filter_trials_by_type(trials=trials, trial_type=self.trial_type)
        if len(trials) == 0:
            return {}
        return self.runner.poll_trial_status(trials=trials)

    def wait_for_completed_trials_and_report_results(
        self,
        idle_callback: Callable[[Scheduler], None] | None = None,
        force_refit: bool = False,
    ) -> dict[str, Any]:
        """Continuously poll for successful trials, with limited exponential
        backoff, and process the results. Stop once at least one successful
        trial has been found. This function can be overridden to a different
        waiting function as needed; it must call `poll_and_process_results`
        to ensure that trials that completed their evaluation are appropriately
        marked as 'COMPLETED' in Ax.

        Args:
            idle_callback: Callable that takes a Scheduler instance as an argument to
                deliver information while the trials are still running. Any output of
                `idle_callback` will not be returned, so `idle_callback` must expose
                information in some other way. For example, it could print something
                about the state of the scheduler or underlying experiment to STDOUT,
                write something to a database, or modify a Plotly figure or other object
                in place. `ax.service.utils.report_utils.get_figure_and_callback` is a
                helper function for generating a callback that will update a Plotly
                figure.
            force_refit: Whether to force a refit of the model during report_results.

        Returns:
            Results of the optimization so far, represented as a
            dict. The contents of the dict depend on the implementation of
            `report_results` in the given `Scheduler` subclass.
        """
        if self.options.init_seconds_between_polls is None:
            raise ValueError(
                "Default `wait_for_completed_trials_and_report_results` in base "
                "`Scheduler` relies on non-null `init_seconds_between_polls` scheduler "
                "option."
            )

        seconds_between_polls = self.options.init_seconds_between_polls
        backoff_factor = self.options.seconds_between_polls_backoff_factor

        total_seconds_elapsed = 0
        while len(self.pending_trials) > 0 and not self.poll_and_process_results():
            if total_seconds_elapsed > MAX_SECONDS_BETWEEN_REPORTS:
                break  # If maximum wait time reached, check the stopping
                # criterion again and and re-attempt scheduling more trials.

            if idle_callback is not None:
                try:
                    idle_callback(self)
                except Exception as e:
                    self.logger.warning(
                        f"Exception raised in ``idle_callback``: {e}. "
                        "Continuing to poll for completed trials."
                    )

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

            total_seconds_elapsed += seconds_between_polls
            seconds_between_polls *= backoff_factor

        if idle_callback is not None:
            idle_callback(self)
        return self.report_results(force_refit=force_refit)

    def should_consider_optimization_complete(self) -> tuple[bool, str]:
        """Whether this scheduler should consider this optimization complete and not
        run more trials (and conclude the optimization via ``_complete_optimization``).

        NOTE: An optimization is considered complete when a generation strategy signaled
        completion or when the ``should_consider_optimization_complete`` method on this
        scheduler evaluates to ``True``. The ``should_consider_optimization_complete``
        method is also responsible for checking global_stopping_strategy's decision as
        well. Alongside the stop decision, this function returns a string describing the
        reason for stopping the optimization.
        """
        if self._optimization_complete:
            return True, ""
        if len(self.pending_trials) == 0 and self._get_max_pending_trials() == 0:
            return (
                True,
                "All pending trials have completed and max_pending_trials is zero.",
            )

        should_stop, message = self._should_stop_due_to_global_stopping_strategy()
        if not should_stop:
            if self.options.total_trials is None:
                return False, ""
            should_stop, message = self._should_stop_due_to_total_trials()

        if should_stop:
            self.logger.info(
                f"Completing the optimization: {message}. "
                f"`should_consider_optimization_complete` "
                f"is `True`, not running more trials."
            )
        return should_stop, message

    def should_abort_optimization(self, timeout_hours: float | None = None) -> bool:
        """Checks whether this scheduler has reached some intertuption / abort
        criterion, such as an overall optimization timeout, tolerated failure rate, etc.
        """
        # If failure rate has been exceeded, log a warning and make sure we are not
        # scheduling additional trials. Raises an exception after pending trials have
        # completed, but does not abort the optimization immediately.
        self.error_if_failure_rate_exceeded()

        # if optimization is timed out, return True, else return False
        latest_optimization_start_timestamp = self._latest_optimization_start_timestamp
        timeout_in_millis = (
            timeout_hours * 60 * 60 * 1000 if timeout_hours is not None else None
        )
        timed_out = False

        if (
            latest_optimization_start_timestamp is not None
            and timeout_in_millis is not None
        ):
            time_elapsed_in_millis = (
                current_timestamp_in_millis() - latest_optimization_start_timestamp
            )
            timed_out = time_elapsed_in_millis >= timeout_in_millis

        if timed_out:
            self.logger.error(
                "Optimization timed out (timeout hours: " f"{timeout_hours})!"
            )

        return timed_out

    def report_results(self, force_refit: bool = False) -> dict[str, Any]:
        """Optional user-defined function for reporting intermediate
        and final optimization results (e.g. make some API call, write to some
        other db). This function is called whenever new results are available during
        the optimization.

        Args:
            force_refit: Whether to force the implementation of this method to
                refit the model on generation strategy before using it to produce
                results to report (e.g. if using model to visualize data).

        Returns:
            An optional dictionary with any relevant data about optimization.
        """
        # TODO[T61776778]: add utility to get best trial from arbitrary exp.
        return {}

    def summarize_final_result(self) -> OptimizationResult:
        """Get some summary of result: which trial did best, what
        were the metric values, what were encountered failures, etc.
        """
        return OptimizationResult()

    def _check_if_failure_rate_exceeded(self, force_check: bool = False) -> bool:
        """Checks if the failure rate (set in scheduler options) has been exceeded at
        any point during the optimization.

        NOTE: Both FAILED and ABANDONED trial statuses count towards the failure rate.

        Args:
            force_check: Indicates whether to force a failure-rate check
                regardless of the number of trials that have been executed. If False
                (default), the check will be skipped if the optimization has fewer than
                five failed trials. If True, the check will be performed unless there
                are 0 failures.

        Effect on state:
            If the failure rate has been exceeded, a warning is logged and the private
            attribute `_failure_rate_has_been_exceeded` is set to True, which causes the
            `_get_max_pending_trials` to return zero, so that no further trials are
            scheduled and an error is raised at the end of the optimization.

        Returns:
            Boolean representing whether the failure rate has been exceeded.
        """
        if self._failure_rate_has_been_exceeded:
            return True

        num_bad_in_scheduler = self._num_bad_in_scheduler()
        # skip check if 0 failures
        if num_bad_in_scheduler == 0:
            return False

        # skip check if fewer than min_failed_trials_for_failure_rate_check failures
        # unless force_check is True
        if (
            num_bad_in_scheduler < self.options.min_failed_trials_for_failure_rate_check
            and not force_check
        ):
            return False

        num_ran_in_scheduler = self._num_ran_in_scheduler()
        failure_rate_exceeded = (
            num_bad_in_scheduler / num_ran_in_scheduler
        ) > self.options.tolerated_trial_failure_rate

        if failure_rate_exceeded:
            if self._num_trials_bad_due_to_err > num_bad_in_scheduler / 2:
                self.logger.warning(
                    "MetricFetchE INFO: Sweep aborted due to an exceeded error rate, "
                    "which was primarily caused by failure to fetch metrics. Please "
                    "check if anything could cause your metrics to be flaky or "
                    "broken."
                )
            # NOTE: this private attribute causes `_get_max_pending_trials` to
            # return zero, which causes no further trials to be scheduled.
            self._failure_rate_has_been_exceeded = True
            return True

        if failure_rate_exceeded:
            if self._num_trials_bad_due_to_err > num_bad_in_scheduler / 2:
                self.logger.warning(
                    "MetricFetchE INFO: Sweep aborted due to an exceeded error rate, "
                    "which was primarily caused by failure to fetch metrics. Please "
                    "check if anything could cause your metrics to be flaky or "
                    "broken."
                )

            raise self._get_failure_rate_exceeded_error(
                num_bad_in_scheduler=num_bad_in_scheduler,
                num_ran_in_scheduler=num_ran_in_scheduler,
            )
        return False

    def error_if_failure_rate_exceeded(self, force_check: bool = False) -> None:
        """Raises an exception if the failure rate (set in scheduler options) has been
        exceeded at any point during the optimization.

        NOTE: Both FAILED and ABANDONED trial statuses count towards the failure rate.

        Args:
            force_check: Indicates whether to force a failure-rate check
                regardless of the number of trials that have been executed. If False
                (default), the check will be skipped if the optimization has fewer than
                five failed trials. If True, the check will be performed unless there
                are 0 failures.
        """
        if self._check_if_failure_rate_exceeded(force_check=force_check):
            raise self._get_failure_rate_exceeded_error(
                num_bad_in_scheduler=self._num_bad_in_scheduler(),
                num_ran_in_scheduler=self._num_ran_in_scheduler(),
            )

    def _check_exit_status_and_report_results(
        self,
        n_existing: int,
        idle_callback: Callable[[Scheduler], None] | None,
        force_refit: bool,
    ) -> dict[str, Any] | None:
        if not self.options.wait_for_running_trials:
            return None
        return self.wait_for_completed_trials_and_report_results(
            idle_callback, force_refit=True
        )

    def run(self, max_new_trials: int, timeout_hours: float | None = None) -> bool:
        """Schedules trial evaluation(s) if stopping criterion is not triggered,
        maximum parallelism is not currently reached, and capacity allows.
        Logs any failures / issues.

        Args:
            max_new_trials: Maximum number of new trials this function should generate
                and run (useful when generating and running trials in batches). Note
                that this function might also re-deploy existing ``CANDIDATE`` trials
                that failed to deploy before, which will not count against this number.
            timeout_hours: Maximum number of hours, for which
                to run the optimization. This function will abort after running
                for `timeout_hours` even if stopping criterion has not been reached.
                If set to `None`, no optimization timeout will be applied.

        Returns:
            Boolean representing success status.
        """
        (
            optimization_complete,
            completion_message,
        ) = self.should_consider_optimization_complete()
        if optimization_complete:
            return False

        if self.should_abort_optimization(timeout_hours=timeout_hours):
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

            if len(self.pending_trials) < 1:
                raise SchedulerInternalError(
                    "No trials are running but model requires more data. This is an "
                    "invalid state of the scheduler, as no more trials can be produced "
                    "but also no more data is expected as there are no running trials. "
                    "This should be investigated."
                )
            self._log_next_no_trials_reason = False
            return False  # Nothing to run.

        if existing_trials:
            idcs = sorted(t.index for t in existing_trials)
            self.logger.debug(f"Will run pre-existing candidate trials: {idcs}.")

        all_trials = [*existing_trials, *new_trials]
        idcs_str = make_indices_str(indices=(t.index for t in all_trials))
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

    def poll_and_process_results(self, poll_all_trial_statuses: bool = False) -> bool:
        """Takes the following actions:
            1. Poll trial runs for their statuses
            2. Find trials to fetch data for
            3. Apply new trial statuses
            4. Fetch data
            5. Early-stop trials where possible
            6. Save modified trials, having either new statuses or new data

        Returns:
            A boolean representing whether any trial evaluations completed
            or have been marked as failed or abandoned, changing the number of
            currently running trials.
        """
        self._sleep_if_too_early_to_poll()

        # POLL TRIAL STATUSES
        new_status_to_trial_idcs = self.poll_trial_status(
            poll_all_trial_statuses=poll_all_trial_statuses
        )

        trial_indices_with_updated_data_or_status = set()

        # GET TRIALS TO FETCH DATA FOR
        # This must be done before updating the trial statuses, so we can differentiate
        # newly and previously completed trials.
        trial_indices_to_fetch = self._get_trial_indices_to_fetch(
            new_status_to_trial_idcs=new_status_to_trial_idcs
        )

        # UPDATE TRIAL STATUSES
        trial_indices_with_updated_statuses = self._apply_new_trial_statuses(
            new_status_to_trial_idcs=new_status_to_trial_idcs,
        )
        updated_any_trial_status = len(trial_indices_with_updated_statuses) > 0
        trial_indices_with_updated_data_or_status.update(
            trial_indices_with_updated_statuses
        )

        # FETCH DATA FOR TRIALS EXPECTING DATA
        trial_indices_with_new_data = (
            self._fetch_data_and_return_trial_indices_with_new_data(
                trial_idcs=trial_indices_to_fetch,
            )
        )
        trial_indices_with_updated_data_or_status.update(trial_indices_with_new_data)

        # EARLY STOP TRIALS
        stop_trial_info = early_stopping_utils.should_stop_trials_early(
            early_stopping_strategy=self.options.early_stopping_strategy,
            trial_indices=self.running_trial_indices,
            experiment=self.experiment,
        )
        self.experiment.stop_trial_runs(
            trials=[self.experiment.trials[trial_idx] for trial_idx in stop_trial_info],
            reasons=list(stop_trial_info.values()),
        )
        if len(stop_trial_info) > 0:
            trial_indices_with_updated_data_or_status.update(set(stop_trial_info))
            updated_any_trial_status = True

        # UPDATE TRIALS IN DB
        if (
            len(trial_indices_with_updated_data_or_status) > 0
        ):  # Only save if there were updates.
            self.logger.debug(
                f"Updating {len(trial_indices_with_updated_data_or_status)} "
                "trials in DB."
            )
            self._save_or_update_trials_in_db_if_possible(
                experiment=self.experiment,
                trials=[
                    self.experiment.trials[i]
                    for i in trial_indices_with_updated_data_or_status
                ],
            )

        return updated_any_trial_status

    @copy_doc(BestPointMixin.get_best_trial)
    def get_best_trial(
        self,
        optimization_config: OptimizationConfig | None = None,
        trial_indices: Iterable[int] | None = None,
        use_model_predictions: bool = True,
    ) -> tuple[int, TParameterization, TModelPredictArm | None] | None:
        return self._get_best_trial(
            experiment=self.experiment,
            generation_strategy=self.standard_generation_strategy,
            optimization_config=optimization_config,
            trial_indices=trial_indices,
            use_model_predictions=use_model_predictions,
        )

    @copy_doc(BestPointMixin.get_pareto_optimal_parameters)
    def get_pareto_optimal_parameters(
        self,
        optimization_config: OptimizationConfig | None = None,
        trial_indices: Iterable[int] | None = None,
        use_model_predictions: bool = True,
    ) -> dict[int, tuple[TParameterization, TModelPredictArm]]:
        return self._get_pareto_optimal_parameters(
            experiment=self.experiment,
            generation_strategy=self.standard_generation_strategy,
            optimization_config=optimization_config,
            trial_indices=trial_indices,
            use_model_predictions=use_model_predictions,
        )

    @copy_doc(BestPointMixin.get_hypervolume)
    def get_hypervolume(
        self,
        optimization_config: MultiObjectiveOptimizationConfig | None = None,
        trial_indices: Iterable[int] | None = None,
        use_model_predictions: bool = True,
    ) -> float:
        return BestPointMixin._get_hypervolume(
            experiment=self.experiment,
            generation_strategy=self.standard_generation_strategy,
            optimization_config=optimization_config,
            trial_indices=trial_indices,
            use_model_predictions=use_model_predictions,
        )

    @copy_doc(BestPointMixin.get_trace)
    def get_trace(
        self,
        optimization_config: OptimizationConfig | None = None,
    ) -> list[float]:
        return BestPointMixin._get_trace(
            experiment=self.experiment,
            optimization_config=optimization_config,
        )

    @copy_doc(BestPointMixin.get_trace_by_progression)
    def get_trace_by_progression(
        self,
        optimization_config: OptimizationConfig | None = None,
        bins: list[float] | None = None,
        final_progression_only: bool = False,
    ) -> tuple[list[float], list[float]]:
        return BestPointMixin._get_trace_by_progression(
            experiment=self.experiment,
            optimization_config=optimization_config,
            bins=bins,
            final_progression_only=final_progression_only,
        )

    # ------------------------- III. Protected helpers. -----------------------

    def _fetch_data_and_return_trial_indices_with_new_data(
        self, trial_idcs: set[int]
    ) -> set[int]:
        """Fetch data for any trials on the experiment that are expecting new data.

        Args:
            trial_idcs: A set of trial indices to fetch data for.

        Returns:
            Set of trial indices that were updated with new data.  We're not asserting
            that the new data is different than the old data, but may want to
            in the future.
        """
        if len(trial_idcs) > 0:
            results = self._fetch_and_process_trials_data_results(
                trial_indices=trial_idcs,
            )
            return {
                i
                for i, results_by_metric_name in results.items()
                for r in results_by_metric_name.values()
                if r.is_ok()
            }
        return set()

    def _num_bad_in_scheduler(self) -> int:
        """Returns the number of trials that have failed or been abandoned in the
        scheduler.
        """
        # We only count failed trials with indices that came after the preexisting
        # trials on experiment before scheduler use.
        return sum(
            1
            for f in self.failed_abandoned_trial_indices
            if f >= self._num_preexisting_trials
        )

    def _num_ran_in_scheduler(self) -> int:
        """Returns the number of trials that have been run by the scheduler."""
        return len(self.experiment.trials) - self._num_preexisting_trials

    def _apply_new_trial_statuses(
        self, new_status_to_trial_idcs: dict[TrialStatus, set[int]]
    ) -> set[int]:
        """Apply new trial statuses to the experiment according to poll results.

        Args:
            new_status_to_trial_idcs: Changes to be applied to trial statuses from
                poll_trial_status.

        Returns:
            Set of trial indices that were updated with new statuses.
        """
        updated_trial_indices = set()
        for status, trial_idcs in new_status_to_trial_idcs.items():
            if status.is_candidate or status.is_deployed:
                # No need to consider candidate, staged or running trials here (none of
                # these trials should actually be candidates, but we can filter on that)
                continue

            if len(trial_idcs) > 0:
                idcs = make_indices_str(indices=trial_idcs)
                self.logger.info(f"Retrieved {status.name} trials: {idcs}.")

            # Update trial statuses and record which trials were updated.
            trials = self.experiment.get_trials_by_indices(trial_idcs)
            updated_trial_indices.update(trial_idcs)
            for trial in trials:
                if status.is_failed or status.is_abandoned:
                    try:
                        reason = self.runner.poll_exception(trial)
                        trial.mark_as(status=status, unsafe=True, reason=reason)
                    except NotImplementedError:
                        # Some runners do not implement poll_failure_reason, so
                        # we fall back to marking the without a reason.
                        trial.mark_as(status=status, unsafe=True)
                else:
                    trial.mark_as(status=status, unsafe=True)
        return updated_trial_indices

    def _identify_trial_indices_to_fetch(
        self,
        old_status_to_trial_idcs: Mapping[TrialStatus, set[int]],
        new_status_to_trial_idcs: Mapping[TrialStatus, set[int]],
    ) -> set[int]:
        """
        Identify trial indices to fetch data for based on changes in trial statuses.

        Args:
            old_status_to_trial_idcs: Mapping of old trial statuses
                to their corresponding trial indices.
            new_status_to_trial_idcs: Mapping of new trial statuses
                to their corresponding trial indices.
        Returns:
            Set of trial indices to fetch data for.
        """
        # Get newly completed trials
        prev_completed_trial_idcs = old_status_to_trial_idcs.get(
            TrialStatus.COMPLETED, set()
        ) | old_status_to_trial_idcs.get(TrialStatus.EARLY_STOPPED, set())

        newly_completed = (
            new_status_to_trial_idcs.get(TrialStatus.COMPLETED, set())
            - prev_completed_trial_idcs
        )

        idcs = make_indices_str(indices=newly_completed)
        if newly_completed:
            self.logger.debug(f"Will fetch data for newly completed trials: {idcs}.")
        else:
            self.logger.debug("No newly completed trials; not fetching data for any.")

        # Get running trials with metrics available while running
        running_trial_indices_with_metrics = set()
        if any(
            m.is_available_while_running() for m in self.experiment.metrics.values()
        ):
            running_trial_indices_with_metrics = new_status_to_trial_idcs.get(
                TrialStatus.RUNNING, set()
            ) | old_status_to_trial_idcs.get(TrialStatus.RUNNING, set())

            for status, indices in new_status_to_trial_idcs.items():
                if status.is_terminal and indices:
                    running_trial_indices_with_metrics -= indices

            if running_trial_indices_with_metrics:
                idcs = make_indices_str(indices=running_trial_indices_with_metrics)
                self.logger.debug(
                    f"Will fetch data for trials: {idcs} because some metrics "
                    "on experiment are available while trials are running."
                )

        # Get previously completed trials with new data after completion
        recently_completed_trial_indices = self._get_recently_completed_trial_indices()
        if len(recently_completed_trial_indices) > 0:
            idcs = make_indices_str(indices=recently_completed_trial_indices)
            self.logger.debug(
                f"Will fetch data for trials: {idcs} because some metrics "
                "on experiment have new data after completion."
            )

        # Combine all trial indices to fetch data for
        trial_indices_to_fetch = (
            newly_completed
            | running_trial_indices_with_metrics
            | recently_completed_trial_indices
        )

        return trial_indices_to_fetch

    def _get_trial_indices_to_fetch(
        self, new_status_to_trial_idcs: Mapping[TrialStatus, set[int]]
    ) -> set[int]:
        """Get trial indices to fetch data for the experiment given
        `new_status_to_trial_idcs` and metric properties.  This should include:
            - newly completed trials
            - running trials if the experiment has metrics available while running
            - previously completed (or early stopped) trials if the experiment
                has metrics with new data after completion which finished recently

        Args:
            new_status_to_trial_idcs: Changes about to be applied to trial statuses.

        Returns:
            Set of trial indices to fetch data for.
        """
        old_status_to_trial_idcs = {status: set() for status in TrialStatus}

        for trial in self.trials:
            old_status_to_trial_idcs[trial.status].add(trial.index)

        return self._identify_trial_indices_to_fetch(
            old_status_to_trial_idcs=old_status_to_trial_idcs,
            new_status_to_trial_idcs=new_status_to_trial_idcs,
        )

    def _get_recently_completed_trial_indices(self) -> set[int]:
        """Get trials that have completed within the max period specified by metrics."""
        if len(self.experiment.metrics) == 0:
            return set()

        max_period = max(
            m.period_of_new_data_after_trial_completion()
            for m in self.experiment.metrics.values()
        )
        return {
            t.index
            for t in self.trials_expecting_data
            if t.time_completed is not None
            and datetime.now() - none_throws(t.time_completed) < max_period
        }

    def _process_completed_trials(self, newly_completed: set[int]) -> None:
        # Fetch the data for newly completed trials; this will cache the data
        # for all metrics. By pre-caching the data now, we remove the need to
        # fetch it during candidate generation.
        idcs = make_indices_str(indices=newly_completed)
        self.logger.info(f"Fetching data for trials: {idcs}.")
        self._fetch_and_process_trials_data_results(
            trial_indices=newly_completed,
        )

    def _abort_optimization(self, num_preexisting_trials: int) -> dict[str, Any]:
        """Conclude optimization without waiting for anymore running trials and
        return results so far via `report_results`.
        """
        self._record_optimization_complete_message()
        return self.report_results(force_refit=True)

    def _complete_optimization(
        self,
        num_preexisting_trials: int,
        idle_callback: Optional[Callable[[Scheduler], None]] = None,
    ) -> dict[str, Any]:
        """Conclude optimization with waiting for anymore running trials and
        return final results via `wait_for_completed_trials_and_report_results`.
        """
        self._record_optimization_complete_message()
        res = self.wait_for_completed_trials_and_report_results(
            idle_callback=idle_callback, force_refit=True
        )
        # Raise an error if the failure rate exceeds tolerance at the
        # end of the optimization.
        self.error_if_failure_rate_exceeded(force_check=True)
        self._warn_if_non_terminal_trials()
        return res

    def _validate_options(self, options: SchedulerOptions) -> None:
        """Validates `SchedulerOptions` for compatibility with given
        `Scheduler` class.
        """
        if not (0.0 <= options.tolerated_trial_failure_rate < 1.0):
            raise ValueError("`tolerated_trial_failure_rate` must be in [0, 1).")

        if options.early_stopping_strategy is not None and options.validate_metrics:
            if not any(
                m.is_available_while_running() for m in self.experiment.metrics.values()
            ):
                raise ValueError(
                    "Can only specify an early stopping strategy if at least one "
                    "metric is marked as `is_available_while_running`. Otherwise, we "
                    "will be unable to fetch intermediate results with which to "
                    "evaluate early stopping criteria."
                )
        if isinstance(self.experiment, MultiTypeExperiment):
            if options.mt_experiment_trial_type is None:
                raise UserInputError(
                    "Must specify `mt_experiment_trial_type` for MultiTypeExperiment."
                )
            if not self.experiment.supports_trial_type(
                options.mt_experiment_trial_type
            ):
                raise ValueError(
                    "Experiment does not support trial type "
                    f"{options.mt_experiment_trial_type}."
                )
        elif options.mt_experiment_trial_type is not None:
            raise UserInputError(
                "`mt_experiment_trial_type` must be None unless the experiment is a "
                "MultiTypeExperiment."
            )

    def _get_max_pending_trials(self) -> int:
        """Returns the maximum number of pending trials specified in the options, or
        zero, if the failure rate limit has been exceeded at any point during the
        optimization.
        """
        if self._failure_rate_has_been_exceeded:
            return 0
        return self.options.max_pending_trials

    def _prepare_trials(
        self, max_new_trials: int
    ) -> tuple[list[BaseTrial], list[BaseTrial]]:
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
        # 1. Determine available capacity for running trials.
        capacity = self.runner.poll_available_capacity()
        if capacity != -1 and capacity < 1:  # -1 indicates unlimited capacity.
            self.logger.debug("There is no capacity to run any trials.")
            return [], []

        # 2. Determine actual number of trials to run based on capacity,
        # limit on pending trials and limit on total trials.
        n = capacity if self.options.run_trials_in_batches else 1
        total_trials = self.options.total_trials
        max_pending_trials = self._get_max_pending_trials()

        num_pending_trials = len(self.pending_trials)
        max_pending_upper_bound = max_pending_trials - num_pending_trials
        if max_pending_upper_bound < 1:
            self.logger.debug(
                f"`max_pending_trials={max_pending_trials}` and {num_pending_trials} "
                "trials are currently pending; not initiating any additional trials."
            )
            return [], []
        n = max_pending_upper_bound if n == -1 else min(max_pending_upper_bound, n)

        if total_trials is not None:
            left_in_total = total_trials - len(self.trials_expecting_data)
            n = min(n, left_in_total)

        existing_candidate_trials = self.candidate_trials[:n]
        n_new = min(n - len(existing_candidate_trials), max_new_trials)
        new_trials, _err = (
            self._get_next_trials(num_trials=n_new, n=self.options.batch_size)
            if n_new > 0
            else (
                [],
                None,
            )
        )
        return existing_candidate_trials, new_trials

    def _get_next_trials(
        self, num_trials: int = 1, n: int | None = None
    ) -> tuple[list[BaseTrial], Exception | None]:
        """Produce up to `num_trials` new generator runs from the underlying
        generation strategy and create new trials with them. Logs errors
        encountered during generation.

        NOTE: Fewer than `num_trials` trials may be produced if generation
        strategy runs into its parallelism limit or needs more data to proceed.

        Returns:
            List of trials, empty if generation is not possible.
        """
        try:
            generator_runs = self._gen_new_trials_from_generation_strategy(
                num_trials=num_trials, n=n
            )
        except OptimizationComplete as err:
            completion_str = f"Optimization complete: {err}"
            self.logger.info(completion_str)
            self.markdown_messages["Optimization complete"] = MessageOutput(
                text=completion_str,
                priority=OutputPriority.DEBUG,
            )
            self._optimization_complete = True
            return [], err
        except DataRequiredError as err:
            # TODO[T62606107]: consider adding a `more_data_required` property to
            # check to generation strategy to avoid running into this exception.
            if self._log_next_no_trials_reason:
                self.logger.info(
                    "Generated all trials that can be generated currently. "
                    "Model requires more data to generate more trials."
                )
            self.logger.debug(f"Message from generation strategy: {err}")
            return [], err
        except MaxParallelismReachedException as err:
            # TODO[T62606107]: consider adding a `step_max_parallelism_reached`
            # check to generation strategy to avoid running into this exception.
            if self._log_next_no_trials_reason:
                self.logger.info(
                    "Generated all trials that can be generated currently. "
                    "Max parallelism currently reached."
                )
            self.logger.debug(f"Message from generation strategy: {err}")
            return [], err
        except AxGenerationException as err:
            if self._log_next_no_trials_reason:
                self.logger.info(
                    "Generated all trials that can be generated currently. "
                    "`generation_strategy` encountered an error "
                    f"{err}."
                )
            self.logger.debug(f"Message from generation strategy: {err}")
            return [], err
        except OptimizationConfigRequired as err:
            if self._log_next_no_trials_reason:
                self.logger.info(
                    "Generated all trials that can be generated currently. "
                    "`generation_strategy` requires an optimization config "
                    "to be set before generating more trials."
                )
            self.logger.debug(f"Message from generation strategy: {err}")
            return [], err

        if self.options.trial_type == TrialType.TRIAL and any(
            len(generator_run_list[0].arms) > 1 or len(generator_run_list) > 1
            for generator_run_list in generator_runs
        ):
            raise SchedulerInternalError(
                "Generation strategy produced multiple arms when only one was expected."
            )
        trials = []
        for generator_run_list in generator_runs:
            if self.options.trial_type == TrialType.BATCH_TRIAL:
                trial = self.experiment.new_batch_trial(
                    generator_runs=generator_run_list,
                    ttl_seconds=self.options.ttl_seconds_for_trials,
                    trial_type=self.trial_type,
                )
                if self.options.status_quo_weight > 0:
                    trial.set_status_quo_with_weight(
                        status_quo=self.experiment.status_quo,
                        weight=self.options.status_quo_weight,
                    )
            else:
                trial = self.experiment.new_trial(
                    generator_run=generator_run_list[0],
                    ttl_seconds=self.options.ttl_seconds_for_trials,
                    trial_type=self.trial_type,
                )

            trials.append(trial)
        return trials, None

    def _gen_new_trials_from_generation_strategy(
        self,
        num_trials: int,
        n: int | None = None,
    ) -> list[list[GeneratorRun]]:
        """Generates a list ``GeneratorRun``s of length of ``num_trials`` using the
        ``_gen_multiple`` method of the scheduler's ``generation_strategy``, taking
        into account any ``pending`` observations.
        """
        self.generation_strategy.experiment = self.experiment
        # For ``BatchTrial`-s, we generate trials using the new method that can
        # produce GRs for multiple trials, with multiple nodes. But we don't yet
        # want to enable that functionality for single-arm use cases of the
        # ``Scheduler``, as it's still in development.
        if self.options.trial_type == TrialType.BATCH_TRIAL:
            grs = self.generation_strategy.gen_for_multiple_trials_with_multiple_models(
                experiment=self.experiment,
                num_trials=num_trials,
                n=n,
            )
            return grs
        else:
            assert self.options.trial_type == TrialType.TRIAL  # Sanity check.
            pending = get_pending_observation_features_based_on_trial_status(
                experiment=self.experiment
            )
            grs = self.generation_strategy._gen_multiple(
                experiment=self.experiment,
                num_generator_runs=num_trials,
                n=1,
                pending_observations=pending,
                fixed_features=get_fixed_features_from_experiment(
                    experiment=self.experiment
                ),
            )
            return [[gr] for gr in grs]
        # TODO: pass self.trial_type to GS.gen for multi-type experiments

    def _update_and_save_trials(
        self,
        existing_trials: list[BaseTrial],
        new_trials: list[BaseTrial],
        metadata: dict[int, dict[str, Any]],
        reduce_state_generator_runs: bool = False,
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
            reduce_state_generator_runs: Flag to determine
                whether to save model state for every generator run (default)
                or to only save model state on the final generator run of each
                batch.
        """

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def _process_trial(trial):
            if trial.index in metadata:
                trial.update_run_metadata(metadata=metadata[trial.index])
                try:
                    trial.mark_running(no_runner_required=True)
                except ValueError as e:
                    self.logger.warning(
                        "Unable to mark trial as RUNNING due to the following error:\n"
                        + str(e)
                    )
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
            reduce_state_generator_runs=reduce_state_generator_runs,
        )

    def _sleep_if_too_early_to_poll(self) -> None:
        """Wait to query for capacity unless there has been enough time since last
        scheduling.
        """
        if self._latest_trial_start_timestamp is not None:
            seconds_since_run_trial = (
                current_timestamp_in_millis()
                - none_throws(self._latest_trial_start_timestamp)
            ) * 1000
            if seconds_since_run_trial < self.options.min_seconds_before_poll:
                sleep(self.options.min_seconds_before_poll - seconds_since_run_trial)

    def _set_logger(self, options: SchedulerOptions) -> None:
        """Set up the logger with appropriate logging levels."""
        cls_name = self.__class__.__name__
        logger = get_logger(name=f"{__name__}.{cls_name}@{hex(id(self))}")
        set_ax_logger_levels(level=options.logging_level)
        if options.log_filepath is not None:
            handler = build_file_handler(
                filepath=none_throws(options.log_filepath),
                level=options.logging_level,
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

        total_trials = none_throws(self.options.total_trials)
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

    def _validate_runner_and_implemented_metrics(self, experiment: Experiment) -> None:
        """Ensure that the experiment specifies runner and metrics; check that metrics
        are not base ``Metric``-s, which do not implement fetching logic.
        """
        # this will raise an exception if no runner is set on the expeirment
        self.runner
        metrics_are_invalid = False
        if not experiment.metrics:
            msg = "`Scheduler` requires that `experiment.metrics` not be None."
            metrics_are_invalid = True
        else:
            msg = (
                "`Scheduler` requires that experiment specifies metrics "
                "with implemented fetching logic."
            )
            base_metrics = {
                m_name for m_name, m in experiment.metrics.items() if type(m) is Metric
            }
            if base_metrics:
                msg += f" Metrics {base_metrics} do not implement fetching logic."
                metrics_are_invalid = True

        if metrics_are_invalid:
            if self.options.validate_metrics:
                raise UnsupportedError(msg)
            else:
                self.logger.warning(msg)

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
        self.experiment._properties[Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF.value] = (
            True
        )

    def _record_optimization_complete_message(self) -> None:
        """Adds a simple optimization completion message to this scheduler's markdown
        messages.
        """
        completion_msg = OPTIMIZATION_COMPLETION_MSG.format(
            num_trials=len(self.experiment.trials),
            experiment_name=(
                self.experiment.name if self.experiment._name is not None else "unnamed"
            ),
        )
        if "Optimization complete" in self.markdown_messages:
            self.markdown_messages["Optimization complete"].append(text=completion_msg)
        else:
            self.markdown_messages["Optimization complete"] = MessageOutput(
                text=completion_msg,
                priority=OutputPriority.DEBUG,
            )

    def _fetch_and_process_trials_data_results(
        self,
        trial_indices: Iterable[int],
    ) -> dict[int, dict[str, MetricFetchResult]]:
        """
        Fetches results from experiment and modifies trial statuses depending on
        success or failure.
        """

        try:
            kwargs = deepcopy(self.options.fetch_kwargs)
            for k, v in self.DEFAULT_FETCH_KWARGS.items():
                kwargs.setdefault(k, v)
            if kwargs.get("overwrite_existing_data") and kwargs.get(
                "combine_with_last_data"
            ):
                # to avoid error https://fburl.com/code/ilix4okj
                kwargs["overwrite_existing_data"] = False
            if self.trial_type is not None:
                metrics = assert_is_instance(
                    self.experiment, MultiTypeExperiment
                ).metrics_for_trial_type(trial_type=none_throws(self.trial_type))
                kwargs["metrics"] = metrics
            results = self.experiment.fetch_trials_data_results(
                trial_indices=trial_indices,
                **kwargs,
            )
        except Exception as e:
            self.logger.exception(
                f"Failed to fetch data for trials {trial_indices} with error: {e}"
            )
            return {}

        for trial_index, results_by_metric_name in results.items():
            for metric_name, result in results_by_metric_name.items():
                # If the fetch call succeeded, continue.
                if result.is_ok():
                    continue

                # Log the Err so the user is aware that something has failed, even if
                # we do not do anything
                metric_fetch_e = result.unwrap_err()
                self.logger.warning(
                    f"Failed to fetch {metric_name} for trial {trial_index}, found "
                    f"{metric_fetch_e}."
                )

                # If the metric is available while running just continue (we can try
                # again later).
                # NOTE: We don't need to report fetching errors in this case either
                metric = self.experiment.metrics[metric_name]
                status = self.experiment.trials[trial_index].status
                if (
                    metric.is_available_while_running()
                    and status == TrialStatus.RUNNING
                ):
                    self.logger.info(
                        f"MetricFetchE INFO: Because {metric_name} is "
                        f"available_while_running and trial {trial_index} is still "
                        "RUNNING continuing the experiment and retrying on next "
                        "poll..."
                    )
                    continue

                self._num_metric_fetch_e_encountered += 1
                self._report_metric_fetch_e(
                    trial=self.experiment.trials[trial_index],
                    metric_name=metric_name,
                    metric_fetch_e=metric_fetch_e,
                )

                # If the fetch failure was for a metric in the optimization config (an
                # objective or constraint) mark the trial as failed
                optimization_config = self.experiment.optimization_config
                if (
                    optimization_config is not None
                    and metric_name in optimization_config.metrics.keys()
                    and not self.experiment.metrics[
                        metric_name
                    ].is_reconverable_fetch_e(metric_fetch_e=metric_fetch_e)
                ):
                    status = self._mark_err_trial_status(
                        trial=self.experiment.trials[trial_index],
                        metric_name=metric_name,
                        metric_fetch_e=metric_fetch_e,
                    )
                    self.logger.warning(
                        f"MetricFetchE INFO: Because {metric_name} is an objective, "
                        f"marking trial {trial_index} as {status}."
                    )
                    self._num_trials_bad_due_to_err += 1
                    continue

                self.logger.info(
                    "MetricFetchE INFO: Continuing optimization even though "
                    "MetricFetchE encountered."
                )
                continue

        return results

    def _report_metric_fetch_e(
        self,
        trial: BaseTrial,
        metric_name: str,
        metric_fetch_e: MetricFetchE,
    ) -> None:
        pass

    def _mark_err_trial_status(
        self,
        trial: BaseTrial,
        metric_name: str | None = None,
        metric_fetch_e: MetricFetchE | None = None,
    ) -> TrialStatus:
        trial.mark_failed(unsafe=True)

        return TrialStatus.FAILED

    def _get_failure_rate_exceeded_error(
        self,
        num_bad_in_scheduler: int,
        num_ran_in_scheduler: int,
    ) -> FailureRateExceededError:
        return FailureRateExceededError(
            FAILURE_EXCEEDED_MSG.format(
                f_rate=self.options.tolerated_trial_failure_rate,
                n_failed=num_bad_in_scheduler,
                n_ran=num_ran_in_scheduler,
                min_failed=self.options.min_failed_trials_for_failure_rate_check,
            )
        )

    def _warn_if_non_terminal_trials(self) -> None:
        """Warns if there are any non-terminal trials on the experiment."""
        non_terminal_trials = [
            t.index for t in self.experiment.trials.values() if not t.status.is_terminal
        ]
        if len(non_terminal_trials) > 0:
            self.logger.warning(
                f"Found {len(non_terminal_trials)} non-terminal trials on "
                f"{self.experiment.name}: {non_terminal_trials}."
            )

    def _should_stop_due_to_global_stopping_strategy(self) -> tuple[bool, str]:
        """Check if optimization should stop due to global stopping strategy."""
        if (
            self.__ignore_global_stopping_strategy
            or self.options.global_stopping_strategy is None
        ):
            return False, ""
        gss = none_throws(self.options.global_stopping_strategy)
        num_trials = len(self.trials)
        if num_trials > 1000:
            self.logger.info(
                f"There are {num_trials} trials; performing "
                f"completion criterion check with {gss}..."
            )
        stop_optimization, global_stopping_msg = gss.should_stop_optimization(
            experiment=self.experiment
        )
        return stop_optimization, global_stopping_msg

    def _should_stop_due_to_total_trials(self) -> tuple[bool, str]:
        """Check if optimization should stop due to total number of trials."""
        num_trials = len(self.trials)
        should_stop = num_trials >= none_throws(self.options.total_trials)
        return (
            should_stop,
            "Exceeding the total number of trials." if should_stop else "",
        )


def get_fitted_model_bridge(
    scheduler: Scheduler, force_refit: bool = False
) -> ModelBridge:
    """Returns a fitted ModelBridge object. If the model is fit already, directly
    returns the already fitted model. Otherwise, fits and returns a new one.

    Args:
        scheduler: The scheduler object from which to get the fitted model.
        force_refit: If True, will force a data lookup and a refit of the model.

    Returns:
        A ModelBridge object fitted to the observations of the scheduler's experiment.
    """
    gs = scheduler.standard_generation_strategy
    model_bridge = gs.model  # Optional[ModelBridge]
    if model_bridge is None or force_refit:  # Need to re-fit the model.
        gs._fit_current_model(data=None)  # Will lookup_data if none is provided.
        model_bridge = cast(ModelBridge, gs.model)
    return model_bridge

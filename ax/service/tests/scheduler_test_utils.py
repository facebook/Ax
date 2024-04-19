#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
from datetime import datetime, timedelta
from logging import WARNING
from math import ceil
from random import randint
from tempfile import NamedTemporaryFile
from typing import Any, Callable, cast, Dict, Iterable, Optional, Set, Type
from unittest.mock import call, Mock, patch, PropertyMock

import pandas as pd

from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data

from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.runner import Runner
from ax.core.utils import get_pending_observation_features_based_on_trial_status
from ax.early_stopping.strategies import BaseEarlyStoppingStrategy
from ax.exceptions.core import OptimizationComplete, UnsupportedError, UserInputError
from ax.exceptions.generation_strategy import AxGenerationException
from ax.metrics.branin import BraninMetric
from ax.metrics.branin_map import BraninTimestampMapMetric
from ax.modelbridge.cross_validation import compute_model_fit_metrics_from_modelbridge
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models, ST_MTGP_trans
from ax.runners.single_running_trial_mixin import SingleRunningTrialMixin
from ax.runners.synthetic import SyntheticRunner
from ax.service.scheduler import (
    ExperimentStatusProperties,
    FailureRateExceededError,
    get_fitted_model_bridge,
    OptimizationResult,
    Scheduler,
    SchedulerInternalError,
)
from ax.service.utils.scheduler_options import SchedulerOptions, TrialType
from ax.service.utils.with_db_settings_base import WithDBSettingsBase
from ax.storage.json_store.encoders import runner_to_dict
from ax.storage.json_store.registry import CORE_DECODER_REGISTRY, CORE_ENCODER_REGISTRY
from ax.storage.runner_registry import CORE_RUNNER_REGISTRY
from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.save import save_experiment
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.sqa_store.structs import DBSettings
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.common.timeutils import current_timestamp_in_millis
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.testing.core_stubs import (
    DummyEarlyStoppingStrategy,
    DummyGlobalStoppingStrategy,
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_branin_experiment_with_timestamp_map_metric,
    get_branin_multi_objective_optimization_config,
    get_branin_search_space,
    get_generator_run,
    get_sobol,
    SpecialGenerationStrategy,
)
from ax.utils.testing.mock import fast_botorch_optimize
from pyre_extensions import none_throws

from sqlalchemy.orm.exc import StaleDataError

DUMMY_EXCEPTION = "test_exception"


class SyntheticRunnerWithStatusPolling(SyntheticRunner):
    """Test runner that implements `poll_trial_status`, required for compatibility
    with the ``Scheduler``."""

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
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
    ) -> Dict[TrialStatus, Set[int]]:
        completed = {t.index for t in trials}
        return {TrialStatus.COMPLETED: completed}


class TestScheduler(Scheduler):
    """Test scheduler that only implements ``report_results`` for convenience in
    testing.
    """

    def report_results(self, force_refit: bool = False) -> Dict[str, Any]:
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


class EarlyStopsInsteadOfNormalCompletionScheduler(TestScheduler):
    """Test scheduler that marks all trials as ones that should be early-stopped."""

    def should_stop_trials_early(
        self, trial_indices: Set[int]
    ) -> Dict[int, Optional[str]]:
        return {i: None for i in trial_indices}


# ---- Runners below simulate different usage and failure modes for scheduler ----


class RunnerWithFrequentFailedTrials(SyntheticRunner):

    poll_failed_next_time = True

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
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
    ) -> Dict[TrialStatus, Set[int]]:

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
    ) -> Dict[TrialStatus, Set[int]]:
        running = [t.index for t in trials]
        return {TrialStatus.FAILED: {running[randint(0, len(running) - 1)]}}


class NoReportResultsRunner(SyntheticRunner):
    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        if randint(0, 3) > 0:
            running = [t.index for t in trials]
            return {TrialStatus.COMPLETED: {running[randint(0, len(running) - 1)]}}
        return {}


class InfinitePollRunner(SyntheticRunner):
    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        return {}


class RunnerWithEarlyStoppingStrategy(SyntheticRunner):
    poll_trial_status_count = 0

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
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

    def run_multiple(self, trials: Iterable[BaseTrial]) -> Dict[int, Dict[str, Any]]:
        self.run_trial_call_count += 1
        raise ValueError("Failing for testing purposes.")


class BrokenRunnerRuntimeError(SyntheticRunnerWithStatusPolling):

    run_trial_call_count = 0

    def run_multiple(self, trials: Iterable[BaseTrial]) -> Dict[int, Dict[str, Any]]:
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
    ) -> Dict[TrialStatus, Set[int]]:
        running_trials = next(iter(trials)).experiment.trials_by_status[
            TrialStatus.RUNNING
        ]
        completed, still_running = set(), set()
        for t in running_trials:
            # pyre-ignore[58]: Operand is actually supported between these
            if datetime.now() - t.time_run_started > timedelta(seconds=3):
                completed.add(t.index)
            else:
                still_running.add(t.index)

        return {
            TrialStatus.COMPLETED: completed,
            TrialStatus.RUNNING: still_running,
        }


class AxSchedulerTestCase(TestCase):
    """Tests base `Scheduler` functionality.  This test case is meant to
    test Scheduler using `GenerationStrategy`, but be extensible so
    it can be applied to any type of `GenerationStrategyInterface`
    by overriding `GENERATION_STRATEGY_INTERFACE_CLASS` and
    `_get_generation_strategy_strategy_for_test()`. You may also need
    to subclass and change some specific tests that don't apply to
    your specific `GenerationStrategyInterface`."""

    GENERATION_STRATEGY_INTERFACE_CLASS: Type[GenerationStrategyInterface] = (
        GenerationStrategy
    )
    PENDING_FEATURES_CALL_LOCATION: str = str(GenerationStrategy.__module__)
    ALWAYS_USE_DB = False
    EXPECTED_SCHEDULER_REPR: str = (
        "Scheduler(experiment=Experiment(branin_test_experiment), "
        "generation_strategy=GenerationStrategy(name='Sobol+BoTorch', "
        "steps=[Sobol for 5 trials, BoTorch for subsequent trials]), "
        "options=SchedulerOptions(max_pending_trials=10, "
        "trial_type=<TrialType.TRIAL: 0>, batch_size=None, "
        "total_trials=0, tolerated_trial_failure_rate=0.2, "
        "min_failed_trials_for_failure_rate_check=5, log_filepath=None, "
        "logging_level=20, ttl_seconds_for_trials=None, init_seconds_between_"
        "polls=10, min_seconds_before_poll=1.0, seconds_between_polls_backoff_"
        "factor=1.5, timeout_hours=None, run_trials_in_batches=False, "
        "debug_log_run_metadata=False, early_stopping_strategy=None, "
        "global_stopping_strategy=None, suppress_storage_errors_after_"
        "retries=False, wait_for_running_trials=True, fetch_kwargs={}, "
        "validate_metrics=True, status_quo_weight=0.0, "
        "enforce_immutable_search_space_and_opt_config=True))"
    )

    def setUp(self) -> None:
        super().setUp()
        self.branin_experiment = get_branin_experiment()
        self.branin_timestamp_map_metric_experiment = (
            get_branin_experiment_with_timestamp_map_metric()
        )
        self.branin_timestamp_map_metric_experiment.runner = (
            RunnerToAllowMultipleMapMetricFetches()
        )

        self.runner = SyntheticRunnerWithStatusPolling()
        self.branin_experiment.runner = self.runner
        self.branin_experiment._properties[Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF] = (
            True
        )
        self.branin_experiment_no_impl_runner_or_metrics = Experiment(
            search_space=get_branin_search_space(),
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric(name="branin"), minimize=False)
            ),
            name="branin_experiment_no_impl_runner_or_metrics",
        )
        self.sobol_GPEI_GS = choose_generation_strategy(
            search_space=get_branin_search_space()
        )
        self.two_sobol_steps_GS = GenerationStrategy(  # Contrived GS to ensure
            steps=[  # that `DataRequiredError` is property handled in scheduler.
                GenerationStep(  # This error is raised when not enough trials
                    model=Models.SOBOL,  # have been observed to proceed to next
                    num_trials=5,  # geneneration step.
                    min_trials_observed=3,
                    max_parallelism=2,
                ),
                GenerationStep(model=Models.SOBOL, num_trials=-1, max_parallelism=3),
            ]
        )
        # GS to force the scheduler to poll completed trials after each ran trial.
        self.sobol_GS_no_parallelism = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=-1, max_parallelism=1)]
        )

    def _get_generation_strategy_strategy_for_test(
        self,
        experiment: Experiment,
        generation_strategy: Optional[GenerationStrategy] = None,
    ) -> GenerationStrategyInterface:
        return not_none(generation_strategy)

    @property
    def runner_registry(self) -> Dict[Type[Runner], int]:
        return {
            SyntheticRunnerWithStatusPolling: 1998,
            InfinitePollRunner: 1999,
            RunnerWithFailedAndAbandonedTrials: 2000,
            RunnerWithEarlyStoppingStrategy: 2001,
            RunnerWithFrequentFailedTrials: 2002,
            NoReportResultsRunner: 2003,
            BrokenRunnerValueError: 2004,
            RunnerWithAllFailedTrials: 2005,
            BrokenRunnerRuntimeError: 2006,
            SyntheticRunnerWithSingleRunningTrial: 2007,
            SyntheticRunnerWithPredictableStatusPolling: 2008,
            RunnerToAllowMultipleMapMetricFetches: 2009,
            **CORE_RUNNER_REGISTRY,
        }

    @property
    def db_config(self) -> SQAConfig:
        encoder_registry = {
            SyntheticRunnerWithStatusPolling: runner_to_dict,
            **CORE_ENCODER_REGISTRY,
        }
        decoder_registry = {
            SyntheticRunnerWithStatusPolling.__name__: SyntheticRunnerWithStatusPolling,
            **CORE_DECODER_REGISTRY,
        }

        return SQAConfig(
            json_encoder_registry=encoder_registry,
            json_decoder_registry=decoder_registry,
            runner_registry=self.runner_registry,
        )

    @property
    def db_settings(self) -> DBSettings:
        config = self.db_config
        encoder = Encoder(config=config)
        decoder = Decoder(config=config)
        return DBSettings(encoder=encoder, decoder=decoder)

    @property
    def db_settings_if_always_needed(self) -> Optional[DBSettings]:
        if self.ALWAYS_USE_DB:
            return self.db_settings
        return None

    def test_init_with_no_impl(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedError,
            "`Scheduler` requires that experiment specifies a `Runner`.",
        ):
            Scheduler(
                experiment=self.branin_experiment_no_impl_runner_or_metrics,
                generation_strategy=self._get_generation_strategy_strategy_for_test(
                    experiment=self.branin_experiment_no_impl_runner_or_metrics,
                    generation_strategy=self.sobol_GPEI_GS,
                ),
                options=SchedulerOptions(total_trials=10),
                db_settings=self.db_settings_if_always_needed,
            )

    def test_init_with_no_impl_with_runner(self) -> None:
        self.branin_experiment_no_impl_runner_or_metrics.runner = self.runner
        with self.assertRaisesRegex(
            UnsupportedError,
            ".*Metrics {'branin'} do not implement fetching logic.",
        ):
            Scheduler(
                experiment=self.branin_experiment_no_impl_runner_or_metrics,
                generation_strategy=self._get_generation_strategy_strategy_for_test(
                    experiment=self.branin_experiment_no_impl_runner_or_metrics,
                    generation_strategy=self.sobol_GPEI_GS,
                ),
                options=SchedulerOptions(total_trials=10),
                db_settings=self.db_settings_if_always_needed,
            )

    def test_init_with_branin_experiment(self) -> None:
        rgs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GPEI_GS,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=rgs,
            options=SchedulerOptions(
                total_trials=0,
                tolerated_trial_failure_rate=0.2,
                init_seconds_between_polls=10,
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        self.assertEqual(scheduler.experiment, self.branin_experiment)
        self.assertEqual(scheduler.generation_strategy, rgs)
        self.assertEqual(scheduler.options.total_trials, 0)
        self.assertEqual(scheduler.options.tolerated_trial_failure_rate, 0.2)
        self.assertEqual(scheduler.options.init_seconds_between_polls, 10)
        self.assertIsNone(scheduler._latest_optimization_start_timestamp)
        for status_prop in ExperimentStatusProperties:
            self.assertEqual(scheduler.experiment._properties[status_prop.value], [])
        scheduler.run_all_trials()  # Runs no trials since total trials is 0.
        # `_latest_optimization_start_timestamp` should be set now.
        self.assertLessEqual(
            scheduler._latest_optimization_start_timestamp,
            # pyre-fixme[6]: For 2nd param expected `SupportsDunderGT[Variable[_T]]`
            #  but got `int`.
            current_timestamp_in_millis(),
        )

    def test_repr(self) -> None:
        branin_gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GPEI_GS,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=branin_gs,
            options=SchedulerOptions(
                total_trials=0,
                tolerated_trial_failure_rate=0.2,
                init_seconds_between_polls=10,
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        self.maxDiff = None
        self.assertEqual(
            f"{scheduler}",
            self.EXPECTED_SCHEDULER_REPR,
        )

    def test_validate_early_stopping_strategy(self) -> None:
        branin_gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GPEI_GS,
        )
        with patch(
            f"{BraninMetric.__module__}.BraninMetric.is_available_while_running",
            return_value=False,
        ), self.assertRaises(ValueError):
            Scheduler(
                experiment=self.branin_experiment,
                generation_strategy=branin_gs,
                options=SchedulerOptions(
                    early_stopping_strategy=DummyEarlyStoppingStrategy()
                ),
                db_settings=self.db_settings_if_always_needed,
            )

        # should not error
        Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=branin_gs,
            options=SchedulerOptions(
                early_stopping_strategy=DummyEarlyStoppingStrategy()
            ),
            db_settings=self.db_settings_if_always_needed,
        )

    def test_run_multi_arm_generator_run_error(self) -> None:
        branin_gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GPEI_GS,
        )
        with patch.object(
            type(branin_gs),
            "gen_for_multiple_trials_with_multiple_models",
            return_value=[[get_generator_run()]],
        ):
            scheduler = Scheduler(
                experiment=self.branin_experiment,
                generation_strategy=branin_gs,
                options=SchedulerOptions(total_trials=1),
                db_settings=self.db_settings_if_always_needed,
            )
            with self.assertRaisesRegex(
                SchedulerInternalError, ".* only one was expected"
            ):
                scheduler.run_all_trials()

    def test_run_all_trials_using_runner_and_metrics(self) -> None:
        branin_gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        # With runners & metrics, `Scheduler.run_all_trials` should run.
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=branin_gs,
            options=SchedulerOptions(
                total_trials=8,
                init_seconds_between_polls=0,  # Short between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        with patch(
            # Record calls to function, but still execute it.
            (
                f"{self.PENDING_FEATURES_CALL_LOCATION}."
                "get_pending_observation_features_based_on_trial_status"
            ),
            side_effect=get_pending_observation_features_based_on_trial_status,
        ) as mock_get_pending:
            scheduler.run_all_trials()
            # Check that we got pending feat. at least 8 times (1 for each new trial and
            # maybe more for cases where we tried to generate trials but ran into limit
            # on parallel., as polling trial statuses is randomized in Scheduler),
            # so some trials might not yet have come back.
            self.assertGreaterEqual(len(mock_get_pending.call_args_list), 8)
        self.assertTrue(  # Make sure all trials got to complete.
            all(t.completed_successfully for t in scheduler.experiment.trials.values())
        )
        self.assertEqual(len(scheduler.experiment.trials), 8)
        # Check that all the data, fetched during optimization, was attached to the
        # experiment.
        dat = scheduler.experiment.fetch_data().df
        self.assertEqual(set(dat["trial_index"].values), set(range(8)))
        self.assertEqual(
            scheduler.experiment._properties[
                ExperimentStatusProperties.RUN_TRIALS_STATUS
            ],
            ["started", "success"],
        )
        self.assertEqual(
            scheduler.experiment._properties[
                ExperimentStatusProperties.NUM_TRIALS_RUN_PER_CALL
            ],
            [8],
        )
        self.assertEqual(
            scheduler.experiment._properties[
                ExperimentStatusProperties.RESUMED_FROM_STORAGE_TIMESTAMPS
            ],
            [],
        )

    def test_run_all_trials_callback(self) -> None:
        n_total_trials = 8

        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                total_trials=n_total_trials,
                init_seconds_between_polls=0,  # Short between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        trials_info = {"n_completed": 0}

        # pyre-fixme[53]: Captured variable `trials_info` is not annotated.
        def write_n_trials(scheduler: Scheduler) -> None:
            trials_info["n_completed"] = len(scheduler.experiment.trials)

        self.assertTrue(trials_info["n_completed"] == 0)
        scheduler.run_all_trials(idle_callback=write_n_trials)
        self.assertTrue(trials_info["n_completed"] == n_total_trials)

    def base_run_n_trials(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        idle_callback: Optional[Callable[[Scheduler], Any]],
    ) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        # With runners & metrics, `Scheduler.run_all_trials` should run.
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                init_seconds_between_polls=0,  # Short between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        scheduler.run_n_trials(max_trials=1, idle_callback=idle_callback)
        self.assertEqual(len(scheduler.experiment.trials), 1)
        scheduler.run_n_trials(max_trials=10, idle_callback=idle_callback)
        self.assertTrue(  # Make sure all trials got to complete.
            all(t.completed_successfully for t in scheduler.experiment.trials.values())
        )
        # Check that all the data, fetched during optimization, was attached to the
        # experiment.
        dat = scheduler.experiment.fetch_data().df
        self.assertEqual(set(dat["trial_index"].values), set(range(11)))

    def test_run_n_trials(self) -> None:
        self.base_run_n_trials(None)

    def test_run_n_trials_callback(self) -> None:
        test_obj = [0, 0]

        # pyre-fixme[53]: Captured variable `test_obj` is not annotated.
        def _callback(scheduler: Scheduler) -> None:
            test_obj[0] = scheduler._latest_optimization_start_timestamp
            test_obj[1] = "apple"
            return

        self.base_run_n_trials(_callback)

        self.assertFalse(test_obj[0] == 0)
        self.assertTrue(test_obj[1] == "apple")

    def test_run_n_trials_single_step_existing_experiment(
        self, all_completed_trials: bool = False
    ) -> None:
        # Test using the Scheduler to run a single experiment update step.
        # This is the typical behavior in Axolotl.
        branin_experiment = get_branin_experiment(
            with_status_quo=True,
            with_completed_trial=True,
        )
        branin_experiment.runner = SyntheticRunnerWithSingleRunningTrial()
        trial0 = branin_experiment.trials[0]
        trial0.assign_runner()
        sobol_generator = get_sobol(search_space=branin_experiment.search_space)
        sobol_run = sobol_generator.gen(n=15)
        trial1 = branin_experiment.new_batch_trial(optimize_for_power=False)
        trial1.add_generator_run(sobol_run)
        trial1.assign_runner()
        trial1.mark_running()
        if all_completed_trials:
            trial1.mark_completed()
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        # With runners & metrics, `Scheduler.run_all_trials` should run.
        scheduler = Scheduler(
            experiment=branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                # pyre-fixme[6]: For 1st param expected `Optional[int]` but got `float`.
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
                wait_for_running_trials=False,
                enforce_immutable_search_space_and_opt_config=False,
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        with patch.object(
            Scheduler,
            "poll_and_process_results",
            wraps=scheduler.poll_and_process_results,
        ) as mock_poll_and_process_results, patch.object(
            Scheduler,
            "run_trials_and_yield_results",
            wraps=scheduler.run_trials_and_yield_results,
        ) as mock_run_trials_and_yield_results:
            manager = Mock()
            manager.attach_mock(
                mock_poll_and_process_results, "poll_and_process_results"
            )
            manager.attach_mock(
                mock_run_trials_and_yield_results, "run_trials_and_yield_results"
            )
            scheduler.run_n_trials(max_trials=1)
            # test order of calls
            expected_calls = [
                call.poll_and_process_results(),
                call.run_trials_and_yield_results(
                    max_trials=1,
                    ignore_global_stopping_strategy=False,
                    timeout_hours=None,
                    idle_callback=None,
                ),
            ]
            self.assertEqual(manager.mock_calls, expected_calls)
            self.assertEqual(len(scheduler.experiment.trials), 3)
            # check status
            # Note: there is a one step delay here since we do no poll again
            # after running a new trial. So the previous trial is only marked as
            # completed when scheduler.run_n_trials is called again.
            self.assertEqual(
                scheduler.experiment.trials[0].status, TrialStatus.COMPLETED
            )
            self.assertEqual(
                scheduler.experiment.trials[1].status,
                TrialStatus.COMPLETED if all_completed_trials else TrialStatus.RUNNING,
            )
            self.assertEqual(scheduler.experiment.trials[2].status, TrialStatus.RUNNING)
            scheduler.run_n_trials(max_trials=1)
            self.assertEqual(len(scheduler.experiment.trials), 4)
            self.assertEqual(
                scheduler.experiment.trials[0].status, TrialStatus.COMPLETED
            )
            self.assertEqual(
                scheduler.experiment.trials[1].status, TrialStatus.COMPLETED
            )
            self.assertEqual(
                scheduler.experiment.trials[2].status,
                TrialStatus.RUNNING,
            )
            self.assertEqual(scheduler.experiment.trials[3].status, TrialStatus.RUNNING)

    def test_run_n_trials_single_step_all_completed_trials(self) -> None:
        # test that scheduler does not continue to loop, but rather exits it immediately
        # if wait_for_running_trials is False
        self.test_run_n_trials_single_step_existing_experiment(
            all_completed_trials=True
        )

    def test_run_preattached_trials_only(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        # assert that pre-attached trials run when max_trials = number of
        # pre-attached trials
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                init_seconds_between_polls=0,  # Short between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        trial = scheduler.experiment.new_trial()
        parameter_dict = {"x1": 5, "x2": 5}
        # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool, float,
        #  int, str]]` but got `Dict[str, int]`.
        trial.add_arm(Arm(parameters=parameter_dict))
        with self.assertRaisesRegex(
            UserInputError, "number of pre-attached candidate trials .* is greater than"
        ):
            scheduler.run_n_trials(max_trials=0)
        scheduler.run_n_trials(max_trials=1)
        self.assertEqual(len(scheduler.experiment.trials), 1)
        self.assertDictEqual(
            # pyre-fixme[16]: `BaseTrial` has no attribute `arm`.
            scheduler.experiment.trials[0].arm.parameters,
            parameter_dict,
        )
        self.assertTrue(  # Make sure all trials got to complete.
            all(t.completed_successfully for t in scheduler.experiment.trials.values())
        )

    def test_global_stopping(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GS_no_parallelism,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                # Stops the optimization after 5 trials.
                global_stopping_strategy=DummyGlobalStoppingStrategy(
                    min_trials=2, trial_to_stop=5
                ),
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        scheduler.run_n_trials(max_trials=10)
        self.assertEqual(len(scheduler.experiment.trials), 5)
        self.assertEqual(scheduler.estimate_global_stopping_savings(), 0.5)

    def test_ignore_global_stopping(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GS_no_parallelism,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                # Stops the optimization after 5 trials.
                global_stopping_strategy=DummyGlobalStoppingStrategy(
                    min_trials=2, trial_to_stop=5
                ),
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        scheduler.run_n_trials(max_trials=10, ignore_global_stopping_strategy=True)
        self.assertEqual(len(scheduler.experiment.trials), 10)

    def test_stop_trial(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        # With runners & metrics, `Scheduler.run_all_trials` should run.
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                init_seconds_between_polls=0,  # Short between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        with patch.object(
            scheduler.experiment.runner, "stop", return_value=None
        ) as mock_runner_stop, patch.object(
            BaseTrial, "mark_early_stopped"
        ) as mock_mark_stopped:
            scheduler.run_n_trials(max_trials=1)
            scheduler.stop_trial_runs(trials=[scheduler.experiment.trials[0]])
            mock_runner_stop.assert_called_once()
            mock_mark_stopped.assert_called_once()

    @patch(f"{Scheduler.__module__}.MAX_SECONDS_BETWEEN_REPORTS", 2)
    def test_stop_at_MAX_SECONDS_BETWEEN_REPORTS(self) -> None:
        self.branin_experiment.runner = InfinitePollRunner()
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GS_no_parallelism,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                total_trials=8,
                init_seconds_between_polls=0,  # No wait between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        with patch.object(
            scheduler, "wait_for_completed_trials_and_report_results", return_value=None
        ) as mock_await_trials:
            # pyre-fixme[6]: For 1st param expected `Optional[int]` but got `float`.
            scheduler.run_all_trials(timeout_hours=1 / 60 / 15)  # 4 second timeout.
            # We should be calling `wait_for_completed_trials_and_report_results`
            # N = total runtime / `test_stop_at_MAX_SECONDS_BETWEEN_REPORTS` times.
            self.assertEqual(
                len(mock_await_trials.call_args),
                2,  # test_stop_at_MAX_SECONDS_BETWEEN_REPORTS as patched in decorator
            )

    def test_timeout(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=gs,
            options=SchedulerOptions(
                total_trials=8,
                init_seconds_between_polls=0,  # No wait between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        scheduler.run_all_trials(timeout_hours=0)  # Forcing optimization to time out.
        self.assertEqual(len(scheduler.experiment.trials), 0)
        self.assertIn("aborted", scheduler.experiment._properties["run_trials_success"])

    def test_logging(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GPEI_GS,
        )
        with NamedTemporaryFile() as temp_file:
            Scheduler(
                experiment=self.branin_experiment,
                generation_strategy=gs,
                options=SchedulerOptions(
                    total_trials=1,
                    init_seconds_between_polls=0,  # No wait bw polls so test is fast.
                    log_filepath=temp_file.name,
                ),
                db_settings=self.db_settings_if_always_needed,
            ).run_all_trials()
            self.assertGreater(os.stat(temp_file.name).st_size, 0)
            self.assertIn("Running trials [0]", str(temp_file.read()))
            temp_file.close()

    def test_logging_level(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GPEI_GS,
        )
        # We don't have any warnings yet, so warning level of logging shouldn't yield
        # any logs as of now.
        with NamedTemporaryFile() as temp_file:
            Scheduler(
                experiment=self.branin_experiment,
                generation_strategy=gs,
                options=SchedulerOptions(
                    total_trials=3,
                    init_seconds_between_polls=0,  # No wait bw polls so test is fast.
                    log_filepath=temp_file.name,
                    logging_level=WARNING,
                ),
                db_settings=self.db_settings_if_always_needed,
            ).run_all_trials()
            # Ensure that the temp file remains empty
            self.assertEqual(os.stat(temp_file.name).st_size, 0)
            temp_file.close()

    def test_retries(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        # Check that retries will be performed for a retriable error.
        self.branin_experiment.runner = BrokenRunnerRuntimeError()
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=gs,
            options=SchedulerOptions(total_trials=1),
            db_settings=self.db_settings_if_always_needed,
        )
        # Should raise after 3 retries.
        with self.assertRaisesRegex(RuntimeError, ".* testing .*"):
            scheduler.run_all_trials()
            # pyre-fixme[16]: `Scheduler` has no attribute `run_trial_call_count`.
            self.assertEqual(scheduler.run_trial_call_count, 3)

    def test_retries_nonretriable_error(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        # Check that no retries will be performed for `ValueError`, since we
        # exclude it from the retriable errors.
        self.branin_experiment.runner = BrokenRunnerValueError()
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=gs,
            options=SchedulerOptions(total_trials=1),
            db_settings=self.db_settings_if_always_needed,
        )
        # Should raise right away since ValueError is non-retriable.
        with self.assertRaisesRegex(ValueError, ".* testing .*"):
            scheduler.run_all_trials()
            # pyre-fixme[16]: `Scheduler` has no attribute `run_trial_call_count`.
            self.assertEqual(scheduler.run_trial_call_count, 1)

    def test_set_ttl(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=gs,
            options=SchedulerOptions(
                total_trials=2,
                ttl_seconds_for_trials=1,
                init_seconds_between_polls=0,  # No wait between polls so test is fast.
                min_seconds_before_poll=0.0,
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        scheduler.run_all_trials()
        self.assertTrue(
            all(t.ttl_seconds == 1 for t in scheduler.experiment.trials.values())
        )

    def test_failure_rate(self) -> None:
        options = SchedulerOptions(
            total_trials=8,
            tolerated_trial_failure_rate=0.5,
            init_seconds_between_polls=0,  # No wait between polls so test is fast.
            min_failed_trials_for_failure_rate_check=2,
        )

        self.branin_experiment.runner = RunnerWithFrequentFailedTrials()
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GS_no_parallelism,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=gs,
            options=options,
            db_settings=self.db_settings_if_always_needed,
        )
        with self.assertRaises(FailureRateExceededError):
            scheduler.run_all_trials()
        # Trials will have statuses: 0, 2 - FAILED, 1 - COMPLETED. Failure rate
        # is 0.5, and so if 2 of the first 3 trials are failed, we can fail
        # immediately.
        self.assertEqual(len(scheduler.experiment.trials), 3)

        # If all trials fail, we can be certain that the sweep will
        # fail after only 2 trials.
        num_preexisting_trials = len(scheduler.experiment.trials)
        self.branin_experiment.runner = RunnerWithAllFailedTrials()
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GS_no_parallelism,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=gs,
            options=options,
            db_settings=self.db_settings_if_always_needed,
        )
        self.assertEqual(scheduler._num_preexisting_trials, num_preexisting_trials)
        with self.assertRaises(FailureRateExceededError):
            scheduler.run_all_trials()
        self.assertEqual(len(scheduler.experiment.trials), num_preexisting_trials + 2)

    def test_sqa_storage_without_experiment_name(self) -> None:
        init_test_engine_and_session_factory(force_init=True)
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        # Scheduler currently requires that the experiment be pre-saved.
        with self.assertRaisesRegex(ValueError, ".* must specify a name"):
            self.branin_experiment._name = None
            Scheduler(
                experiment=self.branin_experiment,
                generation_strategy=gs,
                options=SchedulerOptions(total_trials=1),
                db_settings=self.db_settings,
            )

    def test_sqa_storage_map_metric_experiment(self) -> None:
        init_test_engine_and_session_factory(force_init=True)
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_timestamp_map_metric_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        self.assertIsNotNone(self.branin_timestamp_map_metric_experiment)
        NUM_TRIALS = 5
        scheduler = Scheduler(
            experiment=self.branin_timestamp_map_metric_experiment,
            generation_strategy=gs,
            options=SchedulerOptions(
                total_trials=NUM_TRIALS,
                init_seconds_between_polls=0,  # No wait between polls so test is fast.
            ),
            db_settings=self.db_settings,
        )
        with patch.object(
            scheduler.experiment,
            "attach_data",
            Mock(wraps=scheduler.experiment.attach_data),
        ) as mock_experiment_attach_data:
            # Artificial timestamp logic so we can later check that it's the
            # last-timestamp data that was preserved after multiple `attach_
            # data` calls.
            with patch(
                f"{Experiment.__module__}.current_timestamp_in_millis",
                side_effect=lambda: len(
                    scheduler.experiment.trials_by_status[TrialStatus.COMPLETED]
                )
                * 1000
                + mock_experiment_attach_data.call_count,
            ):
                scheduler.run_all_trials()
        # Check that experiment and GS were saved and test reloading with reduced state.
        exp, loaded_gs = scheduler._load_experiment_and_generation_strategy(
            self.branin_timestamp_map_metric_experiment.name, reduced_state=True
        )
        exp = none_throws(exp)
        self.assertEqual(len(exp.trials), NUM_TRIALS)

        # There should only be one data object for each trial, since by default the
        # `Scheduler` should override previous data objects when it gets new ones in
        # a subsequent `fetch` call.
        for _, datas in exp.data_by_trial.items():
            self.assertEqual(len(datas), 1)

        # We also should have attempted the fetch more times
        # than there are trials because we have a `MapMetric` (many more since we are
        # waiting 3 seconds for each trial).
        self.assertGreater(mock_experiment_attach_data.call_count, NUM_TRIALS)

        # Check that it's the last-attached data that was kept, using
        # expected value based on logic in mocked "current_timestamp_in_millis"
        num_attach_calls = mock_experiment_attach_data.call_count
        expected_ts_last_trial = len(exp.trials) * 1000 + num_attach_calls
        self.assertEqual(
            next(iter(exp.data_by_trial[len(exp.trials) - 1])),
            expected_ts_last_trial,
        )

    def test_sqa_storage_with_experiment_name(self) -> None:
        init_test_engine_and_session_factory(force_init=True)
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        self.assertIsNotNone(self.branin_experiment)
        NUM_TRIALS = 5
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=gs,
            options=SchedulerOptions(
                total_trials=NUM_TRIALS,
                init_seconds_between_polls=0,  # No wait between polls so test is fast.
            ),
            db_settings=self.db_settings,
        )
        # Check that experiment and GS were saved.
        exp, loaded_gs = scheduler._load_experiment_and_generation_strategy(
            self.branin_experiment.name
        )
        self.assertEqual(exp, self.branin_experiment)
        exp = none_throws(exp)
        self.assertEqual(
            # pyre-fixme[16]: Add `_generator_runs` back to GSI interface or move
            # interface to node-level from strategy-level (the latter is likely the
            # better option) TODO
            len(gs._generator_runs),
            len(not_none(loaded_gs)._generator_runs),
        )
        scheduler.run_all_trials()
        # Check that experiment and GS were saved and test reloading with reduced state.
        exp, loaded_gs = scheduler._load_experiment_and_generation_strategy(
            self.branin_experiment.name, reduced_state=True
        )
        exp = none_throws(exp)
        self.assertEqual(len(exp.trials), NUM_TRIALS)
        # Because of RGS, gs has queued additional unused candidates
        self.assertGreaterEqual(len(gs._generator_runs), NUM_TRIALS)
        new_scheduler = Scheduler.from_stored_experiment(
            experiment_name=self.branin_experiment.name,
            options=SchedulerOptions(
                total_trials=NUM_TRIALS + 1,
                init_seconds_between_polls=0,  # No wait between polls so test is fast.
            ),
            db_settings=self.db_settings,
        )
        # Hack "resumed from storage timestamp" into `exp` to make sure all other fields
        # are equal, since difference in resumed from storage timestamps is expected.
        exp._properties[ExperimentStatusProperties.RESUMED_FROM_STORAGE_TIMESTAMPS] = (
            new_scheduler.experiment._properties[
                ExperimentStatusProperties.RESUMED_FROM_STORAGE_TIMESTAMPS
            ]
        )
        self.assertEqual(new_scheduler.experiment, exp)
        self.assertLessEqual(
            len(gs._generator_runs),
            len(new_scheduler.generation_strategy._generator_runs),
        )
        self.assertEqual(
            len(
                new_scheduler.experiment._properties[
                    ExperimentStatusProperties.RESUMED_FROM_STORAGE_TIMESTAMPS
                ]
            ),
            1,
        )

    def test_from_stored_experiment(self) -> None:
        init_test_engine_and_session_factory(force_init=True)
        save_experiment(self.branin_experiment, config=self.db_config)
        with self.subTest("it errors by default without a generation strategy"):
            with self.assertRaisesRegex(
                ValueError,
                "did not have a generation strategy",
            ):
                Scheduler.from_stored_experiment(
                    experiment_name=self.branin_experiment.name,
                    options=SchedulerOptions(),
                    db_settings=self.db_settings,
                )

    def test_unknown_generation_errors_eventually_exit(self) -> None:
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                total_trials=8,
                init_seconds_between_polls=0,  # No wait between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        scheduler.run_n_trials(max_trials=1)
        with patch.object(
            GenerationStrategy,
            "gen_for_multiple_trials_with_multiple_models",
            side_effect=AxGenerationException("model error"),
        ):
            with self.assertRaises(SchedulerInternalError):
                scheduler.run_n_trials(max_trials=3)

    def test_run_trials_and_yield_results(self) -> None:
        total_trials = 3
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        scheduler = TestScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                init_seconds_between_polls=0,
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        # `BaseBonesScheduler.poll_trial_status` is written to mark one
        # trial as `COMPLETED` at a time, so we should be obtaining results
        # at least as many times as `total_trials` and yielding from generator
        # after obtaining each new result. Note that
        # BraninMetric.is_available_while_running evaluates to True, so we may
        # generate more than `total_trials` results if any intermediate fetching
        # occurs.
        total_trials_completed_so_far = 0
        for res in scheduler.run_trials_and_yield_results(max_trials=total_trials):
            # The number of trials has either stayed the same or increased by 1.
            self.assertIn(
                len(res["trials_completed_so_far"]),
                [total_trials_completed_so_far, total_trials_completed_so_far + 1],
            )
            # If the number of trials has changed, increase our counter.
            if len(res["trials_completed_so_far"]) == total_trials_completed_so_far + 1:
                total_trials_completed_so_far += 1
        self.assertEqual(total_trials_completed_so_far, total_trials)

    def test_run_trials_and_yield_results_with_early_stopper(self) -> None:
        self._helper_for_run_trials_and_yield_results_with_early_stopper()

    def _helper_for_run_trials_and_yield_results_with_early_stopper(
        self,
        # Overridable for generation_strategy_interfaces that aren't
        # capable of respecting parallelism
        respect_parellelism: bool = True,
    ) -> None:
        total_trials = 3
        self.branin_experiment.runner = InfinitePollRunner()
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        scheduler = EarlyStopsInsteadOfNormalCompletionScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                init_seconds_between_polls=0,
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        # All trials should be marked complete after one run.
        with patch.object(
            scheduler,
            "should_stop_trials_early",
            wraps=scheduler.should_stop_trials_early,
        ) as mock_should_stop_trials_early, patch.object(
            InfinitePollRunner, "stop", return_value=None
        ) as mock_stop_trial_run:
            res_list = list(
                scheduler.run_trials_and_yield_results(max_trials=total_trials)
            )
            expected_num_polls = 2 if respect_parellelism else 1
            self.assertEqual(len(res_list), expected_num_polls + 1)
            # Both trials in first batch of parallelism will be early stopped
            self.assertEqual(
                len(res_list[0]["trials_early_stopped_so_far"]),
                (
                    self.two_sobol_steps_GS._steps[0].max_parallelism
                    if respect_parellelism
                    else total_trials
                ),
            )
            # Third trial in second batch of parallelism will be early stopped
            self.assertEqual(len(res_list[1]["trials_early_stopped_so_far"]), 3)
            self.assertEqual(
                mock_should_stop_trials_early.call_count, expected_num_polls
            )
            self.assertEqual(
                mock_stop_trial_run.call_count,
                len(res_list[1]["trials_early_stopped_so_far"]),
            )

    def test_scheduler_with_odd_index_early_stopping_strategy(self) -> None:
        total_trials = 3

        class OddIndexEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
            # Trials with odd indices will be early stopped
            # Thus, with 3 total trials, trial #1 will be early stopped
            def should_stop_trials_early(
                self,
                trial_indices: Set[int],
                experiment: Experiment,
                **kwargs: Dict[str, Any],
            ) -> Dict[int, Optional[str]]:
                # Make sure that we can lookup data for the trial,
                # even though we won't use it in this dummy strategy
                data = experiment.lookup_data(trial_indices=trial_indices)
                if data.df.empty:
                    raise Exception(
                        f"No data found for trials {trial_indices}; "
                        "can't determine whether or not to stop early."
                    )
                return {idx: None for idx in trial_indices if idx % 2 == 1}

        self.branin_timestamp_map_metric_experiment.runner = (
            RunnerWithEarlyStoppingStrategy()
        )
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_timestamp_map_metric_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        scheduler = TestScheduler(
            experiment=self.branin_timestamp_map_metric_experiment,
            generation_strategy=gs,
            options=SchedulerOptions(
                init_seconds_between_polls=0,
                early_stopping_strategy=OddIndexEarlyStoppingStrategy(
                    seconds_between_polls=1
                ),
                fetch_kwargs={
                    "overwrite_existing_data": False,
                },
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        with patch.object(
            RunnerWithEarlyStoppingStrategy, "stop", return_value=None
        ) as mock_stop_trial_run:
            res_list = list(
                scheduler.run_trials_and_yield_results(max_trials=total_trials)
            )
            expected_num_steps = 3
            self.assertEqual(len(res_list), expected_num_steps + 1)
            # Trial #1 early stopped in first step
            self.assertEqual(res_list[0]["trials_early_stopped_so_far"], {1})
            # All trials completed by end of second step
            self.assertEqual(res_list[1]["trials_early_stopped_so_far"], {1})
            self.assertEqual(res_list[1]["trials_completed_so_far"], {2})
            self.assertEqual(res_list[2]["trials_completed_so_far"], {0, 2})
            self.assertEqual(
                mock_stop_trial_run.call_count,
                len(res_list[1]["trials_early_stopped_so_far"]),
            )

        # There should be 3 dataframes for Trial 0 -- one from its *last* intermediate
        # poll and one from when the trial was completed.
        self.assertEqual(len(scheduler.experiment.data_by_trial[0]), 3)

        looked_up_data = scheduler.experiment.lookup_data()
        fetched_data = scheduler.experiment.fetch_data()

        # expect number of rows in regular df to equal num_metrics (2) * num_trials (3)
        num_metrics = 2
        expected_num_rows = num_metrics * total_trials
        self.assertEqual(len(looked_up_data.df), expected_num_rows)
        self.assertEqual(len(fetched_data.df), expected_num_rows)

        # expect number of rows in map df to equal:
        #   num_non_map_metrics * num_trials +
        #   num_map_metrics * num_trials + an extra row, since trial 0 runs
        #   longer and gets results for an extra timestamp
        expected_num_rows = (1 * total_trials) + (1 * total_trials + 1)
        # pyre-fixme[16]: `Data` has no attribute `map_df`.
        self.assertEqual(len(fetched_data.map_df), expected_num_rows)
        self.assertEqual(len(looked_up_data.map_df), expected_num_rows)

        self.assertAlmostEqual(scheduler.estimate_early_stopping_savings(), 0.5)

    def test_scheduler_with_metric_with_new_data_after_completion(self) -> None:
        init_test_engine_and_session_factory(force_init=True)
        branin_gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        self.branin_experiment.runner = SyntheticRunnerWithPredictableStatusPolling()
        # With runners & metrics, `Scheduler.run_all_trials` should run.
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=branin_gs,
            options=SchedulerOptions(
                # total_trials must be at least 2x generation strategy parallelism
                # to cause the possibility of multiple fetches on completed trials
                total_trials=5,
                init_seconds_between_polls=0,  # Short between polls so test is fast.
                # this is necessary to see how many times we fetched specific trials
                fetch_kwargs={"overwrite_existing_data": False},
            ),
            db_settings=self.db_settings,
        )
        with patch.object(
            BraninMetric,
            "period_of_new_data_after_trial_completion",
            return_value=timedelta(hours=1),
        ):
            scheduler.run_all_trials()
        # Expect multiple dataframes for Trial 0 -- it should complete on
        # the first iteration.
        # If it's 1 it means period_of_new_data_after_trial_completion is
        # being disregarded.
        self.assertGreater(len(scheduler.experiment.data_by_trial[0]), 1)

    def test_run_trials_in_batches(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        with patch.object(
            type(self.branin_experiment.runner),
            "poll_available_capacity",
            return_value=2,
        ):
            scheduler = Scheduler(
                experiment=self.branin_experiment,  # Has runner and metrics.
                generation_strategy=gs,
                options=SchedulerOptions(
                    init_seconds_between_polls=0,
                    run_trials_in_batches=True,
                ),
                db_settings=self.db_settings_if_always_needed,
            )

            with patch.object(
                scheduler, "run_trials", side_effect=scheduler.run_trials
            ) as mock_run_trials:
                scheduler.run_n_trials(max_trials=3)
                # Trials should be dispatched twice, as total of three trials
                # should be dispatched but capacity is limited to 2.
                self.assertEqual(mock_run_trials.call_count, ceil(3 / 2))

    def test_base_report_results(self) -> None:
        self.branin_experiment.runner = NoReportResultsRunner()
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                init_seconds_between_polls=0,
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        self.assertEqual(scheduler.run_n_trials(max_trials=3), OptimizationResult())

    def test_optimization_complete(self) -> None:
        # With runners & metrics, `Scheduler.run_all_trials` should run.
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                max_pending_trials=100,
                init_seconds_between_polls=0,  # Short between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        with patch.object(
            self.GENERATION_STRATEGY_INTERFACE_CLASS,
            "gen_for_multiple_trials_with_multiple_models",
            side_effect=OptimizationComplete("test error"),
        ):
            scheduler.run_n_trials(max_trials=1)
        # no trials should run if _gen_multiple throws an OptimizationComplete error
        self.assertEqual(len(scheduler.experiment.trials), 0)

    @patch(
        (
            f"{WithDBSettingsBase.__module__}.WithDBSettingsBase."
            "_save_generation_strategy_to_db_if_possible"
        )
    )
    @patch(
        f"{WithDBSettingsBase.__module__}._save_experiment", side_effect=StaleDataError
    )
    def test_suppress_all_storage_errors(self, mock_save_exp: Mock, _) -> None:
        init_test_engine_and_session_factory(force_init=True)
        config = SQAConfig()
        encoder = Encoder(config=config)
        decoder = Decoder(config=config)
        db_settings = DBSettings(encoder=encoder, decoder=decoder)
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                max_pending_trials=100,
                init_seconds_between_polls=0,  # Short between polls so test is fast.
                suppress_storage_errors_after_retries=True,
            ),
            db_settings=db_settings,
        )
        self.assertEqual(mock_save_exp.call_count, 3)

    def test_max_pending_trials(self) -> None:
        # With runners & metrics, `BareBonesTestScheduler.run_all_trials` should run.
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GPEI_GS,
        )
        scheduler = TestScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                max_pending_trials=1,
                init_seconds_between_polls=0,  # Short between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        last_n_completed = 0
        idx = 0
        for _res in scheduler.run_trials_and_yield_results(max_trials=3):
            curr_n_completed = len(
                self.branin_experiment.trial_indices_by_status[TrialStatus.COMPLETED]
            )
            # Skip if no new trials were completed.
            if last_n_completed == curr_n_completed:
                continue
            idx += 1
            # Trials should be scheduled one-at-a-time w/ parallelism limit of 1.
            self.assertEqual(len(self.branin_experiment.trials), idx)
            # Trials also should be getting completed one-at-a-time.
            self.assertEqual(
                len(
                    self.branin_experiment.trial_indices_by_status[
                        TrialStatus.COMPLETED
                    ]
                ),
                idx,
            )
            last_n_completed = curr_n_completed

    def test_get_best_trial(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                init_seconds_between_polls=0,  # Short between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )

        self.assertIsNone(scheduler.get_best_parameters())

        scheduler.run_n_trials(max_trials=1)

        trial, params, _arm = not_none(scheduler.get_best_trial())
        just_params, _just_arm = not_none(scheduler.get_best_parameters())
        just_params_unmodeled, _just_arm_unmodled = not_none(
            scheduler.get_best_parameters(use_model_predictions=False)
        )
        with self.assertRaisesRegex(
            NotImplementedError, "Please use `get_best_parameters`"
        ):
            scheduler.get_pareto_optimal_parameters()

        with self.assertRaisesRegex(
            NotImplementedError, "Please use `get_pareto_optimal_parameters`"
        ):
            scheduler.get_best_trial(
                optimization_config=get_branin_multi_objective_optimization_config()
            )

        # We override the optimization config but not objectives, so an error
        # results as expected, but only much deeper in the stack.
        with self.assertRaisesRegex(ValueError, "'branin_a' is not in list"):
            scheduler.get_pareto_optimal_parameters(
                optimization_config=get_branin_multi_objective_optimization_config(
                    has_objective_thresholds=True
                )
            )

        self.assertEqual(trial, 0)
        self.assertIn("x1", params)
        self.assertIn("x2", params)

        self.assertEqual(params, just_params)
        self.assertEqual(params, just_params_unmodeled)

    def test_get_best_trial_moo(self) -> None:
        experiment = get_branin_experiment_with_multi_objective()
        experiment.runner = self.runner

        gs = self._get_generation_strategy_strategy_for_test(
            experiment=experiment,
            generation_strategy=self.sobol_GPEI_GS,
        )

        scheduler = Scheduler(
            experiment=experiment,
            generation_strategy=gs,
            options=SchedulerOptions(init_seconds_between_polls=0),
            db_settings=self.db_settings_if_always_needed,
        )

        scheduler.run_n_trials(max_trials=1)

        with self.assertRaisesRegex(
            NotImplementedError, "Please use `get_pareto_optimal_parameters`"
        ):
            scheduler.get_best_trial()

        with self.assertRaisesRegex(
            NotImplementedError, "Please use `get_pareto_optimal_parameters`"
        ):
            scheduler.get_best_parameters()

        self.assertIsNotNone(scheduler.get_pareto_optimal_parameters())

    def test_batch_trial(self, status_quo_weight: float = 0.0) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                init_seconds_between_polls=0,  # Short between polls so test is fast.
                trial_type=TrialType.BATCH_TRIAL,
                batch_size=2,
                status_quo_weight=status_quo_weight,
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        self.branin_experiment.status_quo = Arm(parameters={"x1": 0.0, "x2": 0.0})
        scheduler.run_n_trials(max_trials=1)
        self.assertEqual(len(scheduler.experiment.trials), 1)
        trial = checked_cast(BatchTrial, scheduler.experiment.trials[0])
        self.assertEqual(
            len(trial.arms),
            2 if status_quo_weight == 0.0 else 3,
        )
        if status_quo_weight > 0:
            self.assertEqual(
                trial.arm_weights[self.branin_experiment.status_quo],
                1.0,
            )

    def test_batch_trial_with_status_quo(self) -> None:
        self.test_batch_trial(status_quo_weight=1.0)

    def test_poll_and_process_results_with_reasons(self) -> None:
        options = SchedulerOptions(
            total_trials=4,
            tolerated_trial_failure_rate=0.9,
            init_seconds_between_polls=0,
        )

        self.branin_experiment.runner = RunnerWithFailedAndAbandonedTrials()
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GS_no_parallelism,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=gs,
            options=options,
            db_settings=self.db_settings_if_always_needed,
        )

        with patch.object(
            scheduler.experiment.runner,
            "poll_exception",
            return_value=DUMMY_EXCEPTION,
        ):
            scheduler.run_all_trials()

        abandoned_idx = list(
            scheduler.experiment.trial_indices_by_status[TrialStatus.ABANDONED]
        )[0]
        failed_idx = list(
            scheduler.experiment.trial_indices_by_status[TrialStatus.FAILED]
        )[0]
        completed_idx = list(
            scheduler.experiment.trial_indices_by_status[TrialStatus.COMPLETED]
        )[0]

        self.assertEqual(
            scheduler.experiment.trials[failed_idx]._failed_reason,
            DUMMY_EXCEPTION,
        )
        self.assertEqual(
            scheduler.experiment.trials[abandoned_idx]._abandoned_reason,
            DUMMY_EXCEPTION,
        )
        self.assertIsNone(scheduler.experiment.trials[completed_idx]._failed_reason)

    def test_fetch_and_process_trials_data_results_failed_objective_available_while_running(  # noqa
        self,
    ) -> None:
        experiment = get_branin_experiment_with_timestamp_map_metric()
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        with patch(
            f"{BraninTimestampMapMetric.__module__}.BraninTimestampMapMetric.f",
            side_effect=[Exception("yikes!"), {"mean": 0, "timestamp": 12345}],
        ), patch(
            f"{BraninMetric.__module__}.BraninMetric.f",
            side_effect=[Exception("yikes!"), 0],
        ), patch(
            f"{SyntheticRunner.__module__}.SyntheticRunner.poll_trial_status",
            side_effect=[
                {TrialStatus.RUNNING: {0}},
                {TrialStatus.COMPLETED: {0}},
            ],
        ), self.assertLogs(
            logger="ax.service.scheduler", level="INFO"
        ) as lg:
            scheduler = Scheduler(
                experiment=experiment,
                generation_strategy=gs,
                options=SchedulerOptions(),
                db_settings=self.db_settings_if_always_needed,
            )
            scheduler.run_n_trials(max_trials=1)

            self.assertTrue(
                any(
                    "Failed to fetch branin_map for trial 0" in msg for msg in lg.output
                )
            )
            self.assertTrue(
                any("Waiting for completed trials" in msg for msg in lg.output)
            )
        self.assertEqual(scheduler.experiment.trials[0].status, TrialStatus.COMPLETED)

    def test_fetch_and_process_trials_data_results_failed_non_objective(
        self,
    ) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_timestamp_map_metric_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        with patch(
            f"{BraninMetric.__module__}.BraninMetric.f", side_effect=Exception("yikes!")
        ), self.assertLogs(logger="ax.service.scheduler") as lg:
            scheduler = Scheduler(
                experiment=self.branin_timestamp_map_metric_experiment,
                generation_strategy=gs,
                options=SchedulerOptions(),
                db_settings=self.db_settings_if_always_needed,
            )
            scheduler.run_n_trials(max_trials=1)

            self.assertTrue(
                any(
                    "Failed to fetch branin for trial 0" in warning
                    for warning in lg.output
                )
            )
            self.assertEqual(
                scheduler.experiment.trials[0].status, TrialStatus.COMPLETED
            )

    def test_fetch_and_process_trials_data_results_failed_objective(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        with patch(
            f"{BraninMetric.__module__}.BraninMetric.f", side_effect=Exception("yikes!")
        ), patch(
            f"{BraninMetric.__module__}.BraninMetric.is_available_while_running",
            return_value=False,
        ), self.assertLogs(
            logger="ax.service.scheduler"
        ) as lg:
            scheduler = Scheduler(
                experiment=self.branin_experiment,
                generation_strategy=gs,
                options=SchedulerOptions(),
                db_settings=self.db_settings_if_always_needed,
            )

            # This trial will fail
            with self.assertRaises(FailureRateExceededError):
                scheduler.run_n_trials(max_trials=1)

            self.assertTrue(
                any(
                    "Failed to fetch branin for trial 0" in warning
                    for warning in lg.output
                )
            )
            self.assertTrue(
                any(
                    "Because branin is an objective, marking trial 0 as "
                    "TrialStatus.FAILED" in warning
                    for warning in lg.output
                )
            )
            self.assertEqual(scheduler.experiment.trials[0].status, TrialStatus.FAILED)

    def test_completion_criterion(self) -> None:
        # Tests non-GSS parts of the completion criterion.
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GPEI_GS,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=gs,
            options=SchedulerOptions(
                total_trials=None,
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        # With total_trials=None.
        should_stop, message = scheduler.completion_criterion()
        self.assertFalse(should_stop)
        self.assertEqual(message, "")

        # With total_trials=5.
        scheduler.options = SchedulerOptions(total_trials=5)
        # Experiment has fewer trials.
        should_stop, message = scheduler.completion_criterion()
        self.assertFalse(should_stop)
        self.assertEqual(message, "")
        # Experiment has 5 trials.
        sobol_generator = get_sobol(search_space=self.branin_experiment.search_space)
        for _ in range(5):
            sobol_run = sobol_generator.gen(n=1)
            self.branin_experiment.new_trial(generator_run=sobol_run)
        self.assertEqual(len(self.branin_experiment.trials), 5)
        should_stop, message = scheduler.completion_criterion()
        self.assertTrue(should_stop)
        self.assertEqual(message, "Exceeding the total number of trials.")

    def test_get_fitted_model_bridge(self) -> None:
        self.branin_experiment._properties[Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF] = (
            True
        )
        # generation strategy
        NUM_SOBOL = 5
        generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL, num_trials=NUM_SOBOL, max_parallelism=NUM_SOBOL
                ),
                GenerationStep(model=Models.GPEI, num_trials=-1),
            ]
        )
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=generation_strategy,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=gs,
            options=SchedulerOptions(),
            db_settings=self.db_settings_if_always_needed,
        )
        # need to run some trials to initialize the ModelBridge
        scheduler.run_n_trials(max_trials=NUM_SOBOL + 1)
        self._helper_path_that_refits_the_model_if_it_is_not_already_initialized(
            scheduler=scheduler,
        )

    def _helper_path_that_refits_the_model_if_it_is_not_already_initialized(
        self,
        scheduler: Scheduler,
    ) -> None:
        with patch.object(
            self.GENERATION_STRATEGY_INTERFACE_CLASS,
            "model",
            new_callable=PropertyMock,
            return_value=None,
        ), patch.object(
            self.GENERATION_STRATEGY_INTERFACE_CLASS,
            "_fit_current_model",
            wraps=scheduler.standard_generation_strategy._fit_current_model,
        ) as fit_model:
            get_fitted_model_bridge(scheduler)
            fit_model.assert_called_once()

        # testing get_fitted_model_bridge
        model_bridge = get_fitted_model_bridge(scheduler)

        # testing compatibility with compute_model_fit_metrics_from_modelbridge
        fit_metrics = compute_model_fit_metrics_from_modelbridge(
            model_bridge=model_bridge,
            experiment=scheduler.experiment,
            untransform=False,
        )
        r2 = fit_metrics.get("coefficient_of_determination")
        self.assertIsInstance(r2, dict)
        r2 = cast(Dict[str, float], r2)
        self.assertTrue("branin" in r2)
        r2_branin = r2["branin"]
        self.assertIsInstance(r2_branin, float)

        std = fit_metrics.get("std_of_the_standardized_error")
        self.assertIsInstance(std, dict)
        std = cast(Dict[str, float], std)
        self.assertTrue("branin" in std)
        std_branin = std["branin"]
        self.assertIsInstance(std_branin, float)

        # testing with empty metrics dict
        empty_metrics = compute_model_fit_metrics_from_modelbridge(
            model_bridge=model_bridge,
            experiment=scheduler.experiment,
            fit_metrics_dict={},
            untransform=False,
        )
        self.assertIsInstance(empty_metrics, dict)
        self.assertTrue(len(empty_metrics) == 0)

    def test_standard_generation_strategy(self) -> None:
        with self.subTest("with a `GenerationStrategy"):
            # Tests standard GS creation.
            scheduler = Scheduler(
                experiment=self.branin_experiment,
                generation_strategy=self.sobol_GPEI_GS,
                options=SchedulerOptions(),
                db_settings=self.db_settings_if_always_needed,
            )
            self.assertEqual(scheduler.standard_generation_strategy, self.sobol_GPEI_GS)

        with self.subTest("with a `SpecialGenerationStrategy`"):
            scheduler = Scheduler(
                experiment=self.branin_experiment,
                generation_strategy=SpecialGenerationStrategy(),
                options=SchedulerOptions(),
                db_settings=self.db_settings_if_always_needed,
            )
            with self.assertRaisesRegex(
                NotImplementedError,
                "only supported with instances of `GenerationStrategy`",
            ):
                scheduler.standard_generation_strategy

    def test_get_improvement_over_baseline(self) -> None:
        n_total_trials = 8

        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )

        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                total_trials=n_total_trials,
                init_seconds_between_polls=0,  # Short between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )

        scheduler.run_all_trials()

        first_trial_name = (
            scheduler.experiment.trials[0].lookup_data().df["arm_name"].iloc[0]
        )
        percent_improvement = scheduler.get_improvement_over_baseline(
            baseline_arm_name=first_trial_name,
        )

        # Assert that the best trial improves, or
        # at least doesn't regress, over the first trial.
        self.assertGreaterEqual(percent_improvement, 0.0)

    def test_get_improvement_over_baseline_robustness_not_implemented(self) -> None:
        """Test edge cases for get_improvement_over_baseline"""
        experiment = get_branin_experiment_with_multi_objective()
        experiment.runner = self.runner

        gs = self._get_generation_strategy_strategy_for_test(
            experiment=experiment,
            generation_strategy=self.sobol_GPEI_GS,
        )

        scheduler = Scheduler(
            experiment=experiment,
            generation_strategy=gs,
            options=SchedulerOptions(init_seconds_between_polls=0),
            db_settings=self.db_settings_if_always_needed,
        )

        with self.assertRaises(NotImplementedError):
            scheduler.get_improvement_over_baseline(
                baseline_arm_name=None,
            )

    def test_get_improvement_over_baseline_robustness_user_input_error(self) -> None:
        """Test edge cases for get_improvement_over_baseline"""
        experiment = get_branin_experiment_with_multi_objective()
        experiment.name = f"{self.branin_experiment.name}_but_moo"
        experiment.runner = self.runner

        gs = self._get_generation_strategy_strategy_for_test(
            experiment=experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                total_trials=2,
                init_seconds_between_polls=0,  # Short between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )

        with self.assertRaises(UserInputError):
            scheduler.get_improvement_over_baseline(
                baseline_arm_name=None,
            )

        exp = scheduler.experiment
        exp_copy = Experiment(
            search_space=exp.search_space,
            name=exp.name,
            optimization_config=None,
            tracking_metrics=exp.tracking_metrics,
            runner=exp.runner,
        )
        scheduler.experiment = exp_copy

        with self.assertRaises(ValueError):
            scheduler.get_improvement_over_baseline(baseline_arm_name="baseline")

    def test_get_improvement_over_baseline_no_baseline(self) -> None:
        """Test that get_improvement_over_baseline returns UserInputError when
        baseline is not found in data."""
        n_total_trials = 8
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )

        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                total_trials=n_total_trials,
                init_seconds_between_polls=0,  # Short between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )

        scheduler.run_all_trials()

        with self.assertRaises(UserInputError):
            scheduler.get_improvement_over_baseline(
                baseline_arm_name="baseline_arm_not_in_data",
            )

    def test_it_can_skip_metric_validation(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )
        self.branin_experiment._optimization_config = None
        for metric in self.branin_experiment.metrics:
            self.branin_experiment.remove_tracking_metric(metric)

        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                validate_metrics=False,
                early_stopping_strategy=DummyEarlyStoppingStrategy(),
                # Avoids error because `seconds_between_polls`
                # is not defined on `DummyEarlyStoppingStrategy`
                # init_seconds_between_polls=0,
            ),
            db_settings=self.db_settings_if_always_needed,
        )

        scheduler.run_n_trials(max_trials=1)

        self.assertEqual(len(scheduler.experiment.completed_trials), 1)

    def test_it_does_not_overwrite_data_with_combine_fetch_kwarg(self) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
        )

        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                fetch_kwargs={
                    "combine_with_last_data": True,
                }
            ),
            db_settings=self.db_settings_if_always_needed,
        )

        scheduler.run_n_trials(max_trials=1)

        self.assertEqual(len(self.branin_experiment.completed_trials), 1)
        self.branin_experiment.attach_data(
            Data(
                df=pd.DataFrame(
                    {
                        "arm_name": ["0_0"],
                        "metric_name": ["foo"],
                        "mean": [1.0],
                        "sem": [0.1],
                        "trial_index": [0],
                    }
                )
            )
        )
        attached_metrics = (
            self.branin_experiment.lookup_data().df["metric_name"].unique()
        )
        # the attach has overwritten the data, so we can infer that
        # fetching happened in the next `run_n_trials()`
        self.assertNotIn("branin", attached_metrics)

        scheduler.run_n_trials(max_trials=1)
        attached_metrics = (
            self.branin_experiment.lookup_data().df["metric_name"].unique()
        )
        # it did fetch again, but kept "foo" because of the combine kwarg
        self.assertIn("foo", attached_metrics)
        self.assertIn("branin", attached_metrics)

    @fast_botorch_optimize
    def test_it_works_with_multitask_models(
        self,
    ) -> None:
        gs = self._get_generation_strategy_strategy_for_test(
            experiment=self.branin_experiment,
            generation_strategy=GenerationStrategy(
                steps=[
                    GenerationStep(
                        model=Models.SOBOL,
                        num_trials=1,
                    ),
                    GenerationStep(
                        model=Models.GPEI,
                        num_trials=1,
                    ),
                    GenerationStep(
                        model=Models.BOTORCH_MODULAR,
                        model_kwargs={
                            # this will cause and error if the model
                            # doesn't get fixed features
                            "transforms": ST_MTGP_trans,
                            "transform_configs": {
                                "TrialAsTask": {
                                    "trial_level_map": {
                                        "trial_index": {
                                            str(i): str(i) for i in range(3)
                                        }
                                    }
                                }
                            },
                        },
                        num_trials=1,
                    ),
                ]
            ),
        )

        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=gs,
            options=SchedulerOptions(
                total_trials=3,
                # pyre-fixme[6]: For 2nd param expected `Optional[int]` but got `float`.
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
            ),
            db_settings=self.db_settings_if_always_needed,
        )
        scheduler.run_n_trials(max_trials=3)

        # This is to ensure it generated from all nodes
        self.assertTrue(scheduler.standard_generation_strategy.optimization_complete)
        self.assertEqual(len(self.branin_experiment.trials), 3)

    def test_update_options_with_validate_metrics(self) -> None:
        experiment = self.branin_experiment_no_impl_runner_or_metrics
        experiment.runner = self.runner
        scheduler = Scheduler(
            experiment=experiment,
            generation_strategy=self._get_generation_strategy_strategy_for_test(
                experiment=self.branin_experiment_no_impl_runner_or_metrics,
                generation_strategy=self.sobol_GPEI_GS,
            ),
            options=SchedulerOptions(total_trials=10, validate_metrics=False),
            db_settings=self.db_settings_if_always_needed,
        )
        with self.assertRaisesRegex(
            UnsupportedError,
            ".*Metrics {'branin'} do not implement fetching logic.",
        ):
            scheduler.options = SchedulerOptions(total_trials=10, validate_metrics=True)

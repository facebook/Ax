#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from logging import WARNING
from math import ceil
from random import randint
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from unittest.mock import patch

from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.early_stopping.strategies import BaseEarlyStoppingStrategy
from ax.exceptions.core import OptimizationComplete, UnsupportedError, UserInputError
from ax.metrics.branin import BraninMetric
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.modelbridge_utils import (
    get_pending_observation_features_based_on_trial_status,
)
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from ax.service.scheduler import (
    ExperimentStatusProperties,
    FailureRateExceededError,
    OptimizationResult,
    Scheduler,
    SchedulerInternalError,
    SchedulerOptions,
)
from ax.service.utils.scheduler_options import TrialType
from ax.service.utils.with_db_settings_base import WithDBSettingsBase
from ax.storage.json_store.encoders import runner_to_dict
from ax.storage.json_store.registry import CORE_DECODER_REGISTRY, CORE_ENCODER_REGISTRY
from ax.storage.runner_registry import CORE_RUNNER_REGISTRY
from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.sqa_store.structs import DBSettings
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.common.timeutils import current_timestamp_in_millis
from ax.utils.testing.core_stubs import (
    DummyEarlyStoppingStrategy,
    DummyGlobalStoppingStrategy,
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_branin_experiment_with_timestamp_map_metric,
    get_branin_search_space,
    get_generator_run,
)
from sqlalchemy.orm.exc import StaleDataError


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


class TestScheduler(Scheduler):
    """Test scheduler that only implements ``report_results`` for convenience in
    testing.
    """

    # pyre-fixme[15]: `report_results` overrides method defined in `Scheduler`
    #  inconsistently.
    def report_results(
        self, force_refit: bool = False
    ) -> Tuple[bool, Dict[str, Set[int]]]:
        # pyre-fixme[7]: Expected `Tuple[bool, Dict[str, Set[int]]]` but got
        #  `Dict[str, Set[int]]`.
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

    # pyre-fixme[3]: Return type must be annotated.
    def should_stop_trials_early(self, trial_indices: Set[int]):
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
    # pyre-fixme[3]: Return type must be annotated.
    def poll_trial_status(self, trials: Iterable[BaseTrial]):
        return {}


class RunnerWithEarlyStoppingStrategy(SyntheticRunner):
    poll_trial_status_count = 0

    # pyre-fixme[3]: Return type must be annotated.
    def poll_trial_status(self, trials: Iterable[BaseTrial]):
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

    # pyre-fixme[14]: `run_multiple` overrides method defined in `Runner`
    #  inconsistently.
    # pyre-fixme[15]: `run_multiple` overrides method defined in `Runner`
    #  inconsistently.
    def run_multiple(self, trials: List[BaseTrial]) -> Dict[str, Any]:
        self.run_trial_call_count += 1
        raise ValueError("Failing for testing purposes.")


class BrokenRunnerRuntimeError(SyntheticRunnerWithStatusPolling):

    run_trial_call_count = 0

    # pyre-fixme[14]: `run_multiple` overrides method defined in `Runner`
    #  inconsistently.
    # pyre-fixme[15]: `run_multiple` overrides method defined in `Runner`
    #  inconsistently.
    def run_multiple(self, trials: List[BaseTrial]) -> Dict[str, Any]:
        self.run_trial_call_count += 1
        raise RuntimeError("Failing for testing purposes.")


class TestAxScheduler(TestCase):
    """Tests base `Scheduler` functionality."""

    def setUp(self) -> None:
        self.branin_experiment = get_branin_experiment()
        self.branin_timestamp_map_metric_experiment = (
            get_branin_experiment_with_timestamp_map_metric()
        )

        self.runner = SyntheticRunnerWithStatusPolling()
        self.branin_experiment.runner = self.runner
        self.branin_experiment._properties[
            Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF
        ] = True
        self.branin_experiment_no_impl_runner_or_metrics = Experiment(
            search_space=get_branin_search_space(),
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric(name="branin"))
            ),
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

    def test_init(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedError,
            "`Scheduler` requires that experiment specifies a `Runner`.",
        ):
            scheduler = Scheduler(
                experiment=self.branin_experiment_no_impl_runner_or_metrics,
                generation_strategy=self.sobol_GPEI_GS,
                options=SchedulerOptions(total_trials=10),
            )
        self.branin_experiment_no_impl_runner_or_metrics.runner = self.runner
        with self.assertRaisesRegex(
            UnsupportedError,
            ".*Metrics {'branin'} do not implement fetching logic.",
        ):
            scheduler = Scheduler(
                experiment=self.branin_experiment_no_impl_runner_or_metrics,
                generation_strategy=self.sobol_GPEI_GS,
                options=SchedulerOptions(total_trials=10),
            )
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GPEI_GS,
            options=SchedulerOptions(
                total_trials=0,
                tolerated_trial_failure_rate=0.2,
                init_seconds_between_polls=10,
            ),
        )
        self.assertEqual(scheduler.experiment, self.branin_experiment)
        self.assertEqual(scheduler.generation_strategy, self.sobol_GPEI_GS)
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
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GPEI_GS,
            options=SchedulerOptions(
                total_trials=0,
                tolerated_trial_failure_rate=0.2,
                init_seconds_between_polls=10,
            ),
        )
        self.assertEqual(
            f"{scheduler}",
            (
                "Scheduler(experiment=Experiment(branin_test_experiment), "
                "generation_strategy=GenerationStrategy(name='Sobol+GPEI', "
                "steps=[Sobol for 5 trials, GPEI for subsequent trials]), "
                "options=SchedulerOptions(max_pending_trials=10, "
                "trial_type=<TrialType.TRIAL: 0>, batch_size=None, "
                "total_trials=0, tolerated_trial_failure_rate=0.2, "
                "min_failed_trials_for_failure_rate_check=5, log_filepath=None, "
                "logging_level=20, ttl_seconds_for_trials=None, init_seconds_between_"
                "polls=10, min_seconds_before_poll=1.0, seconds_between_polls_backoff_"
                "factor=1.5, timeout_hours=None, run_trials_in_batches=False, "
                "debug_log_run_metadata=False, early_stopping_strategy=None, "
                "global_stopping_strategy=None, suppress_storage_errors_after_"
                "retries=False))"
            ),
        )

    def test_validate_early_stopping_strategy(self) -> None:
        with patch(
            f"{BraninMetric.__module__}.BraninMetric.is_available_while_running",
            return_value=False,
        ), self.assertRaises(ValueError):
            Scheduler(
                experiment=self.branin_experiment,
                generation_strategy=self.sobol_GPEI_GS,
                options=SchedulerOptions(
                    early_stopping_strategy=DummyEarlyStoppingStrategy()
                ),
            )

        with patch.object(
            OptimizationConfig, "is_moo_problem", return_value=True
        ), self.assertRaisesRegex(
            UnsupportedError,
            "Early stopping is not supported on multi-objective problems",
        ):
            Scheduler(
                experiment=self.branin_experiment,
                generation_strategy=self.sobol_GPEI_GS,
                options=SchedulerOptions(
                    early_stopping_strategy=DummyEarlyStoppingStrategy()
                ),
            )

        # should not error
        Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GPEI_GS,
            options=SchedulerOptions(
                early_stopping_strategy=DummyEarlyStoppingStrategy()
            ),
        )

    @patch(
        f"{GenerationStrategy.__module__}.GenerationStrategy._gen_multiple",
        return_value=[get_generator_run()],
    )
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def test_run_multi_arm_generator_run_error(self, mock_gen):
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GPEI_GS,
            options=SchedulerOptions(total_trials=1),
        )
        with self.assertRaisesRegex(SchedulerInternalError, ".* only one was expected"):
            scheduler.run_all_trials()

    @patch(
        # Record calls to function, but still execute it.
        (
            f"{Scheduler.__module__}."
            "get_pending_observation_features_based_on_trial_status"
        ),
        side_effect=get_pending_observation_features_based_on_trial_status,
    )
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def test_run_all_trials_using_runner_and_metrics(self, mock_get_pending):
        # With runners & metrics, `Scheduler.run_all_trials` should run.
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                total_trials=8,
                # pyre-fixme[6]: For 2nd param expected `Optional[int]` but got `float`.
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
            ),
        )
        scheduler.run_all_trials()
        # Check that we got pending feat. at least 8 times (1 for each new trial and
        # maybe more for cases where we tried to generate trials but ran into limit on
        # paralel., as polling trial statuses is randomized in Scheduler),
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

        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                total_trials=n_total_trials,
                # pyre-fixme[6]: For 2nd param expected `Optional[int]` but got `float`.
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
            ),
        )
        trials_info = {"n_completed": 0}

        # pyre-fixme[53]: Captured variable `trials_info` is not annotated.
        def write_n_trials(scheduler: Scheduler) -> None:
            trials_info["n_completed"] = len(scheduler.experiment.trials)

        self.assertTrue(trials_info["n_completed"] == 0)
        scheduler.run_all_trials(idle_callback=write_n_trials)
        self.assertTrue(trials_info["n_completed"] == n_total_trials)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    def base_run_n_trials(self, idle_callback: Optional[Callable[[Scheduler], Any]]):
        # With runners & metrics, `Scheduler.run_all_trials` should run.
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                # pyre-fixme[6]: For 1st param expected `Optional[int]` but got `float`.
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
            ),
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

    def test_run_preattached_trials_only(self) -> None:
        # assert that pre-attached trials run when max_trials = number of
        # pre-attached trials
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                # pyre-fixme[6]: For 1st param expected `Optional[int]` but got `float`.
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
            ),
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

    def test_inferring_reference_point(self) -> None:
        experiment = get_branin_experiment_with_multi_objective()
        experiment.runner = self.runner

        scheduler = Scheduler(
            experiment=experiment,
            generation_strategy=self.sobol_GS_no_parallelism,
            options=SchedulerOptions(
                # Stops the optimization after 5 trials.
                global_stopping_strategy=DummyGlobalStoppingStrategy(
                    min_trials=2, trial_to_stop=5
                ),
            ),
        )

        with patch(
            "ax.service.scheduler.infer_reference_point_from_experiment"
        ) as mock_infer_rp:
            scheduler.run_n_trials(max_trials=10)
            mock_infer_rp.assert_called_once()

    def test_global_stopping(self) -> None:
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.sobol_GS_no_parallelism,
            options=SchedulerOptions(
                # Stops the optimization after 5 trials.
                global_stopping_strategy=DummyGlobalStoppingStrategy(
                    min_trials=2, trial_to_stop=5
                ),
            ),
        )
        scheduler.run_n_trials(max_trials=10)
        self.assertEqual(len(scheduler.experiment.trials), 5)

    def test_ignore_global_stopping(self) -> None:
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.sobol_GS_no_parallelism,
            options=SchedulerOptions(
                # Stops the optimization after 5 trials.
                global_stopping_strategy=DummyGlobalStoppingStrategy(
                    min_trials=2, trial_to_stop=5
                ),
            ),
        )
        scheduler.run_n_trials(max_trials=10, ignore_global_stopping_strategy=True)
        self.assertEqual(len(scheduler.experiment.trials), 10)

    def test_stop_trial(self) -> None:
        # With runners & metrics, `Scheduler.run_all_trials` should run.
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                # pyre-fixme[6]: For 1st param expected `Optional[int]` but got `float`.
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
            ),
        )
        with patch.object(
            scheduler.experiment.runner, "stop", return_value=None
        ) as mock_runner_stop:
            scheduler.run_n_trials(max_trials=1)
            scheduler.stop_trial_runs(trials=[scheduler.experiment.trials[0]])
            mock_runner_stop.assert_called_once()

    @patch(f"{Scheduler.__module__}.MAX_SECONDS_BETWEEN_REPORTS", 2)
    def test_stop_at_MAX_SECONDS_BETWEEN_REPORTS(self) -> None:
        self.branin_experiment.runner = InfinitePollRunner()
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                total_trials=8,
                init_seconds_between_polls=1,  # No wait between polls so test is fast.
            ),
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
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                total_trials=8,
                init_seconds_between_polls=0,  # No wait between polls so test is fast.
            ),
        )
        scheduler.run_all_trials(timeout_hours=0)  # Forcing optimization to time out.
        self.assertEqual(len(scheduler.experiment.trials), 0)
        self.assertIn("aborted", scheduler.experiment._properties["run_trials_success"])

    def test_logging(self) -> None:
        with NamedTemporaryFile() as temp_file:
            Scheduler(
                experiment=self.branin_experiment,
                generation_strategy=self.sobol_GPEI_GS,
                options=SchedulerOptions(
                    total_trials=1,
                    init_seconds_between_polls=0,  # No wait bw polls so test is fast.
                    log_filepath=temp_file.name,
                ),
            ).run_all_trials()
            self.assertGreater(os.stat(temp_file.name).st_size, 0)
            self.assertIn("Running trials [0]", str(temp_file.readline()))
            temp_file.close()

    def test_logging_level(self) -> None:
        # We don't have any warnings yet, so warning level of logging shouldn't yield
        # any logs as of now.
        with NamedTemporaryFile() as temp_file:
            Scheduler(
                experiment=self.branin_experiment,
                generation_strategy=self.sobol_GPEI_GS,
                options=SchedulerOptions(
                    total_trials=3,
                    init_seconds_between_polls=0,  # No wait bw polls so test is fast.
                    log_filepath=temp_file.name,
                    logging_level=WARNING,
                ),
            ).run_all_trials()
            # Ensure that the temp file remains empty
            self.assertEqual(os.stat(temp_file.name).st_size, 0)
            temp_file.close()

    def test_retries(self) -> None:
        # Check that retries will be performed for a retriable error.
        self.branin_experiment.runner = BrokenRunnerRuntimeError()
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(total_trials=1),
        )
        # Should raise after 3 retries.
        with self.assertRaisesRegex(RuntimeError, ".* testing .*"):
            scheduler.run_all_trials()
            # pyre-fixme[16]: `Scheduler` has no attribute `run_trial_call_count`.
            self.assertEqual(scheduler.run_trial_call_count, 3)

    def test_retries_nonretriable_error(self) -> None:
        # Check that no retries will be performed for `ValueError`, since we
        # exclude it from the retriable errors.
        self.branin_experiment.runner = BrokenRunnerValueError()
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(total_trials=1),
        )
        # Should raise right away since ValueError is non-retriable.
        with self.assertRaisesRegex(ValueError, ".* testing .*"):
            scheduler.run_all_trials()
            # pyre-fixme[16]: `Scheduler` has no attribute `run_trial_call_count`.
            self.assertEqual(scheduler.run_trial_call_count, 1)

    def test_set_ttl(self) -> None:
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                total_trials=2,
                ttl_seconds_for_trials=1,
                init_seconds_between_polls=0,  # No wait between polls so test is fast.
                min_seconds_before_poll=0.0,
            ),
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
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GS_no_parallelism,
            options=options,
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
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GS_no_parallelism,
            options=options,
        )
        self.assertEqual(scheduler._num_preexisting_trials, num_preexisting_trials)
        with self.assertRaises(FailureRateExceededError):
            scheduler.run_all_trials()
        self.assertEqual(len(scheduler.experiment.trials), num_preexisting_trials + 2)

    def test_sqa_storage(self) -> None:
        init_test_engine_and_session_factory(force_init=True)
        encoder_registry = {
            SyntheticRunnerWithStatusPolling: runner_to_dict,
            **CORE_ENCODER_REGISTRY,
        }
        decoder_registry = {
            SyntheticRunnerWithStatusPolling.__name__: SyntheticRunnerWithStatusPolling,
            **CORE_DECODER_REGISTRY,
        }
        runner_registry = {
            SyntheticRunnerWithStatusPolling: 1998,
            **CORE_RUNNER_REGISTRY,
        }

        config = SQAConfig(
            json_encoder_registry=encoder_registry,
            json_decoder_registry=decoder_registry,
            runner_registry=runner_registry,
        )
        encoder = Encoder(config=config)
        decoder = Decoder(config=config)
        db_settings = DBSettings(encoder=encoder, decoder=decoder)
        experiment = self.branin_experiment
        # Scheduler currently requires that the experiment be pre-saved.
        with self.assertRaisesRegex(ValueError, ".* must specify a name"):
            experiment._name = None
            scheduler = Scheduler(
                experiment=experiment,
                generation_strategy=self.two_sobol_steps_GS,
                options=SchedulerOptions(total_trials=1),
                db_settings=db_settings,
            )
        experiment._name = "test_experiment"
        NUM_TRIALS = 5
        scheduler = Scheduler(
            experiment=experiment,
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                total_trials=NUM_TRIALS,
                init_seconds_between_polls=0,  # No wait between polls so test is fast.
            ),
            db_settings=db_settings,
        )
        # Check that experiment and GS were saved.
        exp, gs = scheduler._load_experiment_and_generation_strategy(experiment.name)
        self.assertEqual(exp, experiment)
        self.assertEqual(gs, self.two_sobol_steps_GS)
        scheduler.run_all_trials()
        # Check that experiment and GS were saved and test reloading with reduced state.
        exp, gs = scheduler._load_experiment_and_generation_strategy(
            experiment.name, reduced_state=True
        )
        # pyre-fixme[16]: `Optional` has no attribute `trials`.
        self.assertEqual(len(exp.trials), NUM_TRIALS)
        # pyre-fixme[16]: `Optional` has no attribute `_generator_runs`.
        self.assertEqual(len(gs._generator_runs), NUM_TRIALS)
        # Test `from_stored_experiment`.
        new_scheduler = Scheduler.from_stored_experiment(
            experiment_name=experiment.name,
            options=SchedulerOptions(
                total_trials=NUM_TRIALS + 1,
                init_seconds_between_polls=0,  # No wait between polls so test is fast.
            ),
            db_settings=db_settings,
        )
        # Hack "resumed from storage timestamp" into `exp` to make sure all other fields
        # are equal, since difference in resumed from storage timestamps is expected.
        # pyre-fixme[16]: `Optional` has no attribute `_properties`.
        exp._properties[
            ExperimentStatusProperties.RESUMED_FROM_STORAGE_TIMESTAMPS
        ] = new_scheduler.experiment._properties[
            ExperimentStatusProperties.RESUMED_FROM_STORAGE_TIMESTAMPS
        ]
        self.assertEqual(new_scheduler.experiment, exp)
        self.assertEqual(new_scheduler.generation_strategy, gs)
        self.assertEqual(
            len(
                new_scheduler.experiment._properties[
                    ExperimentStatusProperties.RESUMED_FROM_STORAGE_TIMESTAMPS
                ]
            ),
            1,
        )

    def test_run_trials_and_yield_results(self) -> None:
        total_trials = 3
        scheduler = TestScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                init_seconds_between_polls=0,
            ),
        )
        # `BaseBonesScheduler.poll_trial_status` is written to mark one
        # trial as `COMPLETED` at a time, so we should be obtaining results
        # as many times as `total_trials` and yielding from generator after
        # obtaining each new result.
        res_list = list(scheduler.run_trials_and_yield_results(max_trials=total_trials))
        self.assertEqual(len(res_list), total_trials + 1)
        self.assertEqual(len(res_list[0]["trials_completed_so_far"]), 1)
        self.assertEqual(len(res_list[1]["trials_completed_so_far"]), 2)
        self.assertEqual(len(res_list[2]["trials_completed_so_far"]), 3)

    def test_run_trials_and_yield_results_with_early_stopper(self) -> None:
        total_trials = 3
        self.branin_experiment.runner = InfinitePollRunner()
        scheduler = EarlyStopsInsteadOfNormalCompletionScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                # pyre-fixme[6]: For 1st param expected `Optional[int]` but got `float`.
                init_seconds_between_polls=0.1,
            ),
        )
        # All trials should be marked complete after one run.
        with patch.object(
            scheduler,
            "should_stop_trials_early",
            wraps=scheduler.should_stop_trials_early,
        ) as mock_should_stop_trials_early, patch.object(
            scheduler, "stop_trial_runs", return_value=None
        ) as mock_stop_trial_runs:
            res_list = list(
                scheduler.run_trials_and_yield_results(max_trials=total_trials)
            )
            # Two steps complete the experiment given parallelism.
            expected_num_polls = 2
            self.assertEqual(len(res_list), expected_num_polls + 1)
            # Both trials in first batch of parallelism will be early stopped
            self.assertEqual(len(res_list[0]["trials_early_stopped_so_far"]), 2)
            # Third trial in second batch of parallelism will be early stopped
            self.assertEqual(len(res_list[1]["trials_early_stopped_so_far"]), 3)
            self.assertEqual(
                mock_should_stop_trials_early.call_count, expected_num_polls
            )
            self.assertEqual(mock_stop_trial_runs.call_count, expected_num_polls)

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
        scheduler = TestScheduler(
            experiment=self.branin_timestamp_map_metric_experiment,
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                # pyre-fixme[6]: For 1st param expected `Optional[int]` but got `float`.
                init_seconds_between_polls=0.1,
                early_stopping_strategy=OddIndexEarlyStoppingStrategy(),
            ),
        )
        with patch.object(
            scheduler, "stop_trial_runs", return_value=None
        ) as mock_stop_trial_runs:
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
            self.assertEqual(mock_stop_trial_runs.call_count, expected_num_steps)

        # There should be 2 dataframes for Trial 0 -- one from its *last* intermediate
        # poll and one from when the trial was completed. If Scheduler.poll_and_process
        # results didn't specify overwrite_existing_results=True on the intermediate
        # polls, we'd have 3 dataframes instead -- one from *each* intermediate poll
        # and one from when the trial was completed.
        self.assertEqual(len(scheduler.experiment.data_by_trial[0]), 2)

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

    def test_run_trials_in_batches(self) -> None:
        # TODO[drfreund]: Use `Runner` instead when `poll_available_capacity`
        # is moved to `Runner`
        class PollAvailableCapacityScheduler(Scheduler):
            def poll_available_capacity(self) -> None:
                return 2

        scheduler = PollAvailableCapacityScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                init_seconds_between_polls=0,
                run_trials_in_batches=True,
            ),
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
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                init_seconds_between_polls=0,
            ),
        )
        self.assertEqual(scheduler.run_n_trials(max_trials=3), OptimizationResult())

    @patch(
        f"{GenerationStrategy.__module__}.GenerationStrategy._gen_multiple",
        side_effect=OptimizationComplete("test error"),
    )
    # pyre-fixme[3]: Return type must be annotated.
    def test_optimization_complete(self, _):
        # With runners & metrics, `Scheduler.run_all_trials` should run.
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                max_pending_trials=100,
                # pyre-fixme[6]: For 2nd param expected `Optional[int]` but got `float`.
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
            ),
        )
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
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def test_suppress_all_storage_errors(self, mock_save_exp, _):
        init_test_engine_and_session_factory(force_init=True)
        config = SQAConfig()
        encoder = Encoder(config=config)
        decoder = Decoder(config=config)
        db_settings = DBSettings(encoder=encoder, decoder=decoder)
        Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                max_pending_trials=100,
                # pyre-fixme[6]: For 2nd param expected `Optional[int]` but got `float`.
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
                suppress_storage_errors_after_retries=True,
            ),
            db_settings=db_settings,
        )
        self.assertEqual(mock_save_exp.call_count, 3)

    def test_max_pending_trials(self) -> None:
        # With runners & metrics, `BareBonesTestScheduler.run_all_trials` should run.
        scheduler = TestScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.sobol_GPEI_GS,
            options=SchedulerOptions(
                max_pending_trials=1,
                # pyre-fixme[6]: For 2nd param expected `Optional[int]` but got `float`.
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
            ),
        )
        for idx, _ in enumerate(scheduler.run_trials_and_yield_results(max_trials=3)):
            # Trials should be scheduled one-at-a-time w/ parallelism limit of 1.
            self.assertEqual(
                len(self.branin_experiment.trials), idx + 1 if idx < 3 else idx
            )
            # Trials also should be getting completed one-at-a-time.
            self.assertEqual(
                len(
                    self.branin_experiment.trial_indices_by_status[
                        TrialStatus.COMPLETED
                    ]
                ),
                idx + 1 if idx < 3 else idx,
            )

    def test_get_best_trial(self) -> None:
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                # pyre-fixme[6]: For 1st param expected `Optional[int]` but got `float`.
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
            ),
        )

        self.assertIsNone(scheduler.get_best_parameters())

        scheduler.run_n_trials(max_trials=1)

        # pyre-fixme[23]: Unable to unpack `Optional[Tuple[int, Dict[str,
        #  typing.Union[None, bool, float, int, str]], Optional[Tuple[Dict[str, float],
        #  Optional[Dict[str, typing.Dict[str, float]]]]]]]` into 3 values.
        trial, params, _arm = scheduler.get_best_trial()
        # pyre-fixme[23]: Unable to unpack `Optional[Tuple[Dict[str,
        #  typing.Union[None, bool, float, int, str]], Optional[Tuple[Dict[str, float],
        #  Optional[Dict[str, typing.Dict[str, float]]]]]]]` into 2 values.
        just_params, _just_arm = scheduler.get_best_parameters()
        # pyre-fixme[23]: Unable to unpack `Optional[Tuple[Dict[str,
        #  typing.Union[None, bool, float, int, str]], Optional[Tuple[Dict[str, float],
        #  Optional[Dict[str, typing.Dict[str, float]]]]]]]` into 2 values.
        just_params_unmodeled, _just_arm_unmodled = scheduler.get_best_parameters(
            use_model_predictions=False
        )
        with self.assertRaisesRegex(
            NotImplementedError, "Please use `get_best_parameters`"
        ):
            scheduler.get_pareto_optimal_parameters()

        self.assertEqual(trial, 0)
        self.assertIn("x1", params)
        self.assertIn("x2", params)

        self.assertEqual(params, just_params)
        self.assertEqual(params, just_params_unmodeled)

    def test_get_best_trial_moo(self) -> None:
        experiment = get_branin_experiment_with_multi_objective()
        experiment.runner = self.runner

        scheduler = Scheduler(
            experiment=experiment,
            generation_strategy=self.sobol_GPEI_GS,
            # pyre-fixme[6]: For 1st param expected `Optional[int]` but got `float`.
            options=SchedulerOptions(init_seconds_between_polls=0.1),
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

    def test_batch_trial(self) -> None:
        scheduler = Scheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                # pyre-fixme[6]: For 1st param expected `Optional[int]` but got `float`.
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
                trial_type=TrialType.BATCH_TRIAL,
                batch_size=2,
            ),
        )

        scheduler.run_n_trials(max_trials=1)
        self.assertEqual(len(scheduler.experiment.trials), 1)
        self.assertEqual(len(scheduler.experiment.trials[0].arms), 2)

    def test_fetch_and_process_trials_data_results_failed_objective_available_while_running(  # noqa
        self,
    ) -> None:
        with patch(
            f"{BraninMetric.__module__}.BraninMetric.f", side_effect=Exception("yikes!")
        ), self.assertLogs(logger="ax.service.scheduler") as lg:
            scheduler = Scheduler(
                experiment=get_branin_experiment_with_timestamp_map_metric(),
                generation_strategy=self.two_sobol_steps_GS,
                options=SchedulerOptions(),
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
        with patch(
            f"{BraninMetric.__module__}.BraninMetric.f", side_effect=Exception("yikes!")
        ), patch(
            f"{BraninMetric.__module__}.BraninMetric.is_available_while_running",
            return_value=False,
        ), self.assertLogs(
            logger="ax.service.scheduler"
        ) as lg:
            scheduler = Scheduler(
                experiment=get_branin_experiment(),
                generation_strategy=self.two_sobol_steps_GS,
                options=SchedulerOptions(),
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

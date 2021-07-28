#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from logging import WARNING
from math import ceil
from random import randint
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, Optional, Set, Tuple
from unittest.mock import patch

from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.early_stopping.strategies import BaseEarlyStoppingStrategy
from ax.exceptions.core import OptimizationComplete, UnsupportedError
from ax.metrics.branin import BraninMetric
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.modelbridge_utils import (
    get_pending_observation_features_based_on_trial_status,
)
from ax.modelbridge.registry import Models
from ax.service.scheduler import (
    FailureRateExceededError,
    Scheduler,
    SchedulerInternalError,
    SchedulerOptions,
    ExperimentStatusProperties,
    OptimizationResult,
)
from ax.service.utils.with_db_settings_base import (
    WithDBSettingsBase,
)
from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.sqa_store.structs import DBSettings
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.common.timeutils import current_timestamp_in_millis
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_search_space,
    get_generator_run,
)
from sqlalchemy.orm.exc import StaleDataError


class BareBonesTestScheduler(Scheduler):
    """Test scheduler that only implements the required `poll_trial_status` and
    therefore requires full-fleshed runners and metrics to be set on the experiment.
    """

    def poll_trial_status(self) -> Dict[TrialStatus, Set[int]]:
        # Pretend that sometimes trials take a few seconds to complete and that they
        # might get completed out of order.
        if randint(0, 3) > 0:
            running = [t.index for t in self.running_trials]
            return {TrialStatus.COMPLETED: {running[randint(0, len(running) - 1)]}}
        return {}

    def report_results(self) -> Tuple[bool, Dict[str, Set[int]]]:
        return {
            # use `set` constructor to copy the set, else the value
            # will be a pointer and all will be the same
            "trials_completed_so_far": set(
                self.experiment.trial_indices_by_status[TrialStatus.COMPLETED]
            ),
            "trials_early_stopped_so_far": set(
                self.experiment.trial_indices_by_status[TrialStatus.EARLY_STOPPED]
            ),
        }


class TestScheduler(BareBonesTestScheduler):
    """Test scheduler that extends the `BareBonesTestScheduler` with logic for running
    trials and fetching trial data –– and therefore does not require implemented runners
    or metrics.
    """

    def run_trial(self, trial: BaseTrial) -> Dict[str, Any]:
        return {"name": f"depl_{trial.index}", "run_timestamp": time.time()}

    def run_trials(self, trials: Iterable[BaseTrial]) -> Dict[int, Dict[str, Any]]:
        return {
            t.index: {"name": f"depl_{t.index}", "run_timestamp": time.time()}
            for t in trials
        }


class TestAxScheduler(TestCase):
    """Tests base `Scheduler` functionality."""

    def setUp(self):
        self.branin_experiment = get_branin_experiment()
        self.branin_experiment._properties[
            Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF
        ] = True
        self.branin_experiment_no_impl_metrics = Experiment(
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

    def test_init(self):
        with self.assertRaisesRegex(
            UnsupportedError, ".* metrics .* implemented fetching"
        ):
            scheduler = BareBonesTestScheduler(
                experiment=self.branin_experiment_no_impl_metrics,
                generation_strategy=self.sobol_GPEI_GS,
                options=SchedulerOptions(total_trials=10),
            )
        scheduler = BareBonesTestScheduler(
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
            current_timestamp_in_millis(),
        )

    def test_repr(self):
        scheduler = BareBonesTestScheduler(
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
                "BareBonesTestScheduler(experiment=Experiment(branin_test_experiment), "
                "generation_strategy=GenerationStrategy(name='Sobol+GPEI', "
                "steps=[Sobol for 5 trials, GPEI for subsequent trials]), "
                "options=SchedulerOptions(trial_type=<class 'ax.core.trial.Trial'>, "
                "total_trials=0, tolerated_trial_failure_rate=0.2, "
                "min_failed_trials_for_failure_rate_check=5, log_filepath=None, "
                "logging_level=20, ttl_seconds_for_trials=None, init_seconds_between_"
                "polls=10, min_seconds_before_poll=1.0, seconds_between_polls_backoff_"
                "factor=1.5, run_trials_in_batches=False, "
                "debug_log_run_metadata=False, early_stopping_strategy=None, "
                "suppress_storage_errors_after_retries=False))"
            ),
        )

    def test_validate_runners_if_required(self):
        # `BareBonesTestScheduler` does not have runner and metrics, so it cannot
        # run on experiment that does not specify those (or specifies base Metric,
        # which do not implement data-fetching logic).
        scheduler = BareBonesTestScheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GPEI_GS,
            options=SchedulerOptions(total_trials=10),
        )
        self.branin_experiment.runner = None
        with self.assertRaisesRegex(NotImplementedError, ".* runner is required"):
            scheduler.run_all_trials()

    def test_validate_early_stopping_strategy(self):
        class DummyEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
            def should_stop_trials_early(
                self,
                trial_indices: Set[int],
                experiment: Experiment,
                **kwargs: Dict[str, Any],
            ) -> Set[int]:
                return {}

        with patch(
            f"{BraninMetric.__module__}.BraninMetric.is_available_while_running",
            return_value=False,
        ), self.assertRaises(ValueError):
            BareBonesTestScheduler(
                experiment=self.branin_experiment,
                generation_strategy=self.sobol_GPEI_GS,
                options=SchedulerOptions(
                    early_stopping_strategy=DummyEarlyStoppingStrategy()
                ),
            )

        # should not error
        BareBonesTestScheduler(
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
    def test_run_multi_arm_generator_run_error(self, mock_gen):
        scheduler = TestScheduler(
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
    def test_run_all_trials_using_runner_and_metrics(self, mock_get_pending):
        # With runners & metrics, `BareBonesTestScheduler.run_all_trials` should run.
        scheduler = BareBonesTestScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                total_trials=8,
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
            ),
        )
        scheduler.run_all_trials()
        # Check that we got pending feat. at least 8 times (1 for each new trial and
        # maybe more for cases where we tried to generate trials but ran into limit on
        # paralel., as polling trial statuses is randomized in BareBonesTestScheduler),
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

    def test_run_n_trials(self):
        # With runners & metrics, `BareBonesTestScheduler.run_all_trials` should run.
        scheduler = BareBonesTestScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
            ),
        )
        scheduler.run_n_trials(max_trials=1)
        self.assertEqual(len(scheduler.experiment.trials), 1)
        scheduler.run_n_trials(max_trials=10)
        self.assertTrue(  # Make sure all trials got to complete.
            all(t.completed_successfully for t in scheduler.experiment.trials.values())
        )
        # Check that all the data, fetched during optimization, was attached to the
        # experiment.
        dat = scheduler.experiment.fetch_data().df
        self.assertEqual(set(dat["trial_index"].values), set(range(11)))

    def test_run_preattached_trials_only(self):
        # assert that pre-attached trials run when max_trials = 0
        scheduler = BareBonesTestScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
            ),
        )
        trial = scheduler.experiment.new_trial()
        trial.add_arm(Arm(parameters={"x1": 5, "x2": 5}))
        scheduler.run_n_trials(max_trials=0)
        self.assertEqual(len(scheduler.experiment.trials), 1)
        self.assertTrue(  # Make sure all trials got to complete.
            all(t.completed_successfully for t in scheduler.experiment.trials.values())
        )

    def test_stop_trial(self):
        # With runners & metrics, `BareBonesTestScheduler.run_all_trials` should run.
        scheduler = BareBonesTestScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
            ),
        )
        with patch.object(
            scheduler.experiment.runner, "stop", return_value=None
        ) as mock_runner_stop:
            scheduler.run_n_trials(max_trials=1)
            scheduler.stop_trial_run(scheduler.experiment.trials[0])
            mock_runner_stop.assert_called_once()

    def test_run_all_trials_not_using_runner(self):
        # `TestScheduler` has `run_trial` and `fetch_trial_data` logic, so runner &
        # implemented metrics are not required.
        scheduler = TestScheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                total_trials=8,
                init_seconds_between_polls=0,  # No wait between polls so test is fast.
            ),
        )
        self.branin_experiment.runner = None
        scheduler.run_all_trials()
        self.assertTrue(  # Make sure all trials got to complete.
            all(t.completed_successfully for t in scheduler.experiment.trials.values())
        )
        self.assertEqual(len(scheduler.experiment.trials), 8)
        # Check that all the data, fetched during optimization, was attached to the
        # experiment.
        dat = scheduler.experiment.fetch_data().df
        self.assertEqual(set(dat["trial_index"].values), set(range(8)))

    @patch(f"{Scheduler.__module__}.MAX_SECONDS_BETWEEN_POLLS", 2)
    def test_stop_at_MAX_SECONDS_BETWEEN_POLLS(self):
        class InfinitePollScheduler(BareBonesTestScheduler):
            def poll_trial_status(self):
                return {}

        scheduler = InfinitePollScheduler(
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
            scheduler.run_all_trials(timeout_hours=1 / 60 / 15)  # 4 second timeout.
            # We should be calling `wait_for_completed_trials_and_report_results`
            # N = total runtime / `MAX_SECONDS_BETWEEN_POLLS` times.
            self.assertEqual(
                len(mock_await_trials.call_args),
                2,  # MAX_SECONDS_BETWEEN_POLLS as patched in decorator
            )

    def test_timeout(self):
        # `TestScheduler` has `run_trial` and `fetch_trial_data` logic, so runner &
        # implemented metrics are not required.
        scheduler = TestScheduler(
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

    def test_logging(self):
        with NamedTemporaryFile() as temp_file:
            BareBonesTestScheduler(
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

    def test_logging_level(self):
        # We don't have any warnings yet, so warning level of logging shouldn't yield
        # any logs as of now.
        with NamedTemporaryFile() as temp_file:
            BareBonesTestScheduler(
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

    def test_retries(self):
        # Check that retries will be performed for a retriable error.
        class BrokenSchedulerRuntimeError(BareBonesTestScheduler):

            run_trial_call_count = 0

            def run_trial(self, trial: BaseTrial) -> Dict[str, Any]:
                self.run_trial_call_count += 1
                raise RuntimeError("Failing for testing purposes.")

        scheduler = BrokenSchedulerRuntimeError(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(total_trials=1),
        )
        # Should raise after 3 retries.
        with self.assertRaisesRegex(RuntimeError, ".* testing .*"):
            scheduler.run_all_trials()
            self.assertEqual(scheduler.run_trial_call_count, 3)

    def test_retries_nonretriable_error(self):
        # Check that no retries will be performed for `ValueError`, since we
        # exclude it from the retriable errors.
        class BrokenSchedulerValueError(BareBonesTestScheduler):

            run_trial_call_count = 0

            def run_trial(self, trial: BaseTrial) -> Dict[str, Any]:
                self.run_trial_call_count += 1
                raise ValueError("Failing for testing purposes.")

        scheduler = BrokenSchedulerValueError(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(total_trials=1),
        )
        # Should raise right away since ValueError is non-retriable.
        with self.assertRaisesRegex(ValueError, ".* testing .*"):
            scheduler.run_all_trials()
            self.assertEqual(scheduler.run_trial_call_count, 1)

    def test_set_ttl(self):
        scheduler = TestScheduler(
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

    def test_failure_rate(self):
        class SchedulerWithFrequentFailedTrials(TestScheduler):

            poll_failed_next_time = True

            def poll_trial_status(self) -> Dict[TrialStatus, Set[int]]:
                running = [t.index for t in self.running_trials]
                status = (
                    TrialStatus.FAILED
                    if self.poll_failed_next_time
                    else TrialStatus.COMPLETED
                )
                # Poll different status next time.
                self.poll_failed_next_time = not self.poll_failed_next_time
                return {status: {running[randint(0, len(running) - 1)]}}

        class SchedulerWithAllFailedTrials(TestScheduler):
            def poll_trial_status(self) -> Dict[TrialStatus, Set[int]]:
                running = [t.index for t in self.running_trials]
                return {TrialStatus.FAILED: {running[randint(0, len(running) - 1)]}}

        options = SchedulerOptions(
            total_trials=8,
            tolerated_trial_failure_rate=0.5,
            init_seconds_between_polls=0,  # No wait between polls so test is fast.
            min_failed_trials_for_failure_rate_check=2,
        )

        scheduler = SchedulerWithFrequentFailedTrials(
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
        scheduler = SchedulerWithAllFailedTrials(
            experiment=self.branin_experiment,
            generation_strategy=self.sobol_GS_no_parallelism,
            options=options,
        )
        self.assertEqual(scheduler._num_preexisting_trials, num_preexisting_trials)
        with self.assertRaises(FailureRateExceededError):
            scheduler.run_all_trials()
        self.assertEqual(len(scheduler.experiment.trials), num_preexisting_trials + 2)

    def test_sqa_storage(self):
        init_test_engine_and_session_factory(force_init=True)
        config = SQAConfig()
        encoder = Encoder(config=config)
        decoder = Decoder(config=config)
        db_settings = DBSettings(encoder=encoder, decoder=decoder)
        experiment = self.branin_experiment
        # Scheduler currently requires that the experiment be pre-saved.
        with self.assertRaisesRegex(ValueError, ".* must specify a name"):
            experiment._name = None
            scheduler = TestScheduler(
                experiment=experiment,
                generation_strategy=self.two_sobol_steps_GS,
                options=SchedulerOptions(total_trials=1),
                db_settings=db_settings,
            )
        experiment._name = "test_experiment"
        NUM_TRIALS = 5
        scheduler = TestScheduler(
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
        self.assertEqual(len(exp.trials), NUM_TRIALS)
        self.assertEqual(len(gs._generator_runs), NUM_TRIALS)
        # Test `from_stored_experiment`.
        new_scheduler = TestScheduler.from_stored_experiment(
            experiment_name=experiment.name,
            options=SchedulerOptions(
                total_trials=NUM_TRIALS + 1,
                init_seconds_between_polls=0,  # No wait between polls so test is fast.
            ),
            db_settings=db_settings,
        )
        # Hack "resumed from storage timestamp" into `exp` to make sure all other fields
        # are equal, since difference in resumed from storage timestamps is expected.
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

    def test_run_trials_and_yield_results(self):
        total_trials = 3
        scheduler = BareBonesTestScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                init_seconds_between_polls=0,
            ),
        )
        # `BaseBonesTestScheduler.poll_trial_status` is written to mark one
        # trial as `COMPLETED` at a time, so we should be obtaining results
        # as many times as `total_trials` and yielding from generator after
        # obtaining each new result.
        res_list = list(scheduler.run_trials_and_yield_results(max_trials=total_trials))
        self.assertEqual(len(res_list), total_trials + 1)
        self.assertEqual(len(res_list[0]["trials_completed_so_far"]), 1)
        self.assertEqual(len(res_list[1]["trials_completed_so_far"]), 2)
        self.assertEqual(len(res_list[2]["trials_completed_so_far"]), 3)

    def test_run_trials_and_yield_results_with_early_stopper(self):
        class EarlyStopsInsteadOfNormalCompletionScheduler(BareBonesTestScheduler):
            def poll_trial_status(self):
                return {}

            def should_stop_trials_early(self, trial_indices: Set[int]):
                return {i: None for i in trial_indices}

        total_trials = 3
        scheduler = EarlyStopsInsteadOfNormalCompletionScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
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

    def test_scheduler_with_odd_index_early_stopping_strategy(self):
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

        class SchedulerWithEarlyStoppingStrategy(BareBonesTestScheduler):
            poll_trial_status_count = 0

            def poll_trial_status(self):
                # In the first step, don't complete any trials
                # Trial #1 will be early stopped
                if self.poll_trial_status_count == 0:
                    self.poll_trial_status_count += 1
                    return {}

                # In the second step, complete trials 0 and 2
                self.poll_trial_status_count += 1
                return {TrialStatus.COMPLETED: {0, 2}}

        scheduler = SchedulerWithEarlyStoppingStrategy(
            experiment=self.branin_experiment,
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
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
            expected_num_steps = 2
            self.assertEqual(len(res_list), expected_num_steps + 1)
            # Trial #1 early stopped in first step
            self.assertEqual(res_list[0]["trials_early_stopped_so_far"], {1})
            # All trials completed by end of second step
            self.assertEqual(res_list[1]["trials_early_stopped_so_far"], {1})
            self.assertEqual(res_list[1]["trials_completed_so_far"], {0, 2})
            self.assertEqual(mock_stop_trial_runs.call_count, expected_num_steps)

    def test_run_trials_in_batches(self):
        with self.assertRaisesRegex(
            UnsupportedError, "only if `poll_available_capacity`"
        ):
            scheduler = BareBonesTestScheduler(
                experiment=self.branin_experiment,  # Has runner and metrics.
                generation_strategy=self.two_sobol_steps_GS,
                options=SchedulerOptions(
                    init_seconds_between_polls=0,
                    run_trials_in_batches=True,
                ),
            )
            scheduler.run_n_trials(max_trials=3)

        class PollAvailableCapacityScheduler(BareBonesTestScheduler):
            def poll_available_capacity(self):
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

    def test_base_report_results(self):
        class NoReportResultsScheduler(Scheduler):
            def poll_trial_status(self) -> Dict[TrialStatus, Set[int]]:
                if randint(0, 3) > 0:
                    running = [t.index for t in self.running_trials]
                    return {
                        TrialStatus.COMPLETED: {running[randint(0, len(running) - 1)]}
                    }
                return {}

        scheduler = NoReportResultsScheduler(
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
    def test_optimization_complete(self, _):
        # With runners & metrics, `BareBonesTestScheduler.run_all_trials` should run.
        scheduler = BareBonesTestScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
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
    def test_suppress_all_storage_errors(self, mock_save_exp, _):
        init_test_engine_and_session_factory(force_init=True)
        config = SQAConfig()
        encoder = Encoder(config=config)
        decoder = Decoder(config=config)
        db_settings = DBSettings(encoder=encoder, decoder=decoder)
        BareBonesTestScheduler(
            experiment=self.branin_experiment,  # Has runner and metrics.
            generation_strategy=self.two_sobol_steps_GS,
            options=SchedulerOptions(
                init_seconds_between_polls=0.1,  # Short between polls so test is fast.
                suppress_storage_errors_after_retries=True,
            ),
            db_settings=db_settings,
        )
        self.assertEqual(mock_save_exp.call_count, 3)

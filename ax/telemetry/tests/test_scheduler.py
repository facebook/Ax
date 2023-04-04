#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.service.scheduler import Scheduler, SchedulerOptions
from ax.telemetry.experiment import ExperimentCompletedRecord, ExperimentCreatedRecord
from ax.telemetry.generation_strategy import GenerationStrategyCreatedRecord
from ax.telemetry.scheduler import SchedulerCompletedRecord, SchedulerCreatedRecord
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.modeling_stubs import get_generation_strategy


class TestScheduler(TestCase):
    def test_scheduler_created_record_from_scheduler(self) -> None:
        scheduler = Scheduler(
            experiment=get_branin_experiment(),
            generation_strategy=get_generation_strategy(),
            options=SchedulerOptions(
                total_trials=0,
                tolerated_trial_failure_rate=0.2,
                init_seconds_between_polls=10,
            ),
        )

        record = SchedulerCreatedRecord.from_scheduler(scheduler=scheduler)

        expected = SchedulerCreatedRecord(
            experiment_created_record=ExperimentCreatedRecord.from_experiment(
                experiment=scheduler.experiment
            ),
            generation_strategy_created_record=(
                GenerationStrategyCreatedRecord.from_generation_strategy(
                    generation_strategy=scheduler.generation_strategy
                )
            ),
            scheduler_total_trials=0,
            scheduler_max_pending_trials=10,
            arms_per_trial=1,
            early_stopping_strategy_cls=None,
            global_stopping_strategy_cls=None,
            transformed_dimensionality=2,
        )
        self.assertEqual(record, expected)

        flat = record.flatten()
        expected_dict = {
            **ExperimentCreatedRecord.from_experiment(
                experiment=scheduler.experiment
            ).__dict__,
            **GenerationStrategyCreatedRecord.from_generation_strategy(
                generation_strategy=scheduler.generation_strategy
            ).__dict__,
            "scheduler_total_trials": 0,
            "scheduler_max_pending_trials": 10,
            "arms_per_trial": 1,
            "early_stopping_strategy_cls": None,
            "global_stopping_strategy_cls": None,
            "transformed_dimensionality": 2,
        }
        self.assertEqual(flat, expected_dict)

    def test_scheduler_completed_record_from_scheduler(self) -> None:
        scheduler = Scheduler(
            experiment=get_branin_experiment(),
            generation_strategy=get_generation_strategy(),
            options=SchedulerOptions(
                total_trials=0,
                tolerated_trial_failure_rate=0.2,
                init_seconds_between_polls=10,
            ),
        )

        record = SchedulerCompletedRecord.from_scheduler(scheduler=scheduler)
        expected = SchedulerCompletedRecord(
            experiment_completed_record=ExperimentCompletedRecord.from_experiment(
                experiment=scheduler.experiment
            ),
            best_point_quality=-1,
            model_fit_quality=-1,
            num_metric_fetch_e_encountered=0,
            num_trials_bad_due_to_err=0,
        )
        self.assertEqual(record, expected)

        flat = record.flatten()
        expected_dict = {
            **ExperimentCompletedRecord.from_experiment(
                experiment=scheduler.experiment
            ).__dict__,
            "best_point_quality": -1,
            "model_fit_quality": -1,
            "num_metric_fetch_e_encountered": 0,
            "num_trials_bad_due_to_err": 0,
        }
        self.assertEqual(flat, expected_dict)

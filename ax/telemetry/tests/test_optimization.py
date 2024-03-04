#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import asdict
from datetime import datetime

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.scheduler import Scheduler, SchedulerOptions
from ax.telemetry.ax_client import AxClientCompletedRecord, AxClientCreatedRecord
from ax.telemetry.common import get_unique_identifier
from ax.telemetry.experiment import ExperimentCreatedRecord
from ax.telemetry.generation_strategy import GenerationStrategyCreatedRecord
from ax.telemetry.optimization import (
    OptimizationCompletedRecord,
    OptimizationCreatedRecord,
)
from ax.telemetry.scheduler import SchedulerCompletedRecord, SchedulerCreatedRecord
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.modeling_stubs import get_generation_strategy


class TestOptimization(TestCase):
    def test_optimization_created_record_from_scheduler(self) -> None:
        scheduler = Scheduler(
            experiment=get_branin_experiment(),
            generation_strategy=get_generation_strategy(),
            options=SchedulerOptions(
                total_trials=0,
                tolerated_trial_failure_rate=0.2,
                init_seconds_between_polls=10,
            ),
        )

        record = OptimizationCreatedRecord.from_scheduler(
            scheduler=scheduler,
            unique_identifier="foo",
            owner="bar",
            product_surface="Axolotl",
            launch_surface="web",
            deployed_job_id=1118,
            trial_evaluation_identifier="train",
            is_manual_generation_strategy=True,
            warm_started_from=None,
            num_custom_trials=0,
            support_tier="good!",
        )

        expected_dict = {
            **SchedulerCreatedRecord.from_scheduler(scheduler=scheduler).flatten(),
            "unique_identifier": "foo",
            "owner": "bar",
            "product_surface": "Axolotl",
            "launch_surface": "web",
            "deployed_job_id": 1118,
            "trial_evaluation_identifier": "train",
            "is_manual_generation_strategy": True,
            "warm_started_from": None,
            "num_custom_trials": 0,
            "support_tier": "good!",
        }

        self.assertEqual(asdict(record), expected_dict)

    def test_optimization_created_record_from_ax_client(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objectives={"branin": ObjectiveProperties(minimize=True)},
            is_test=True,
        )

        record = OptimizationCreatedRecord.from_ax_client(
            ax_client=ax_client,
            unique_identifier="foo",
            owner="bar",
            product_surface="Axolotl",
            launch_surface="web",
            deployed_job_id=1118,
            trial_evaluation_identifier="train",
            is_manual_generation_strategy=True,
            warm_started_from=None,
            num_custom_trials=0,
        )

        expected_dict = {
            **AxClientCreatedRecord.from_ax_client(ax_client=ax_client).flatten(),
            "unique_identifier": "foo",
            "owner": "bar",
            "product_surface": "Axolotl",
            "launch_surface": "web",
            "deployed_job_id": 1118,
            "trial_evaluation_identifier": "train",
            "is_manual_generation_strategy": True,
            "warm_started_from": None,
            "num_custom_trials": 0,
            # Extra fields
            "scheduler_max_pending_trials": -1,
            "scheduler_total_trials": None,
            "support_tier": "",
        }

        self.assertEqual(asdict(record), expected_dict)

    def test_optimization_created_record_from_experiment_without_generation_strategy(
        self,
    ) -> None:
        experiment = get_branin_experiment()
        experiment.experiment_type = "NOTEBOOK"
        time_created_str = "2023-04-07T16:01:23.45"
        experiment._time_created = datetime.strptime(
            time_created_str, "%Y-%m-%dT%H:%M:%S.%f"
        )

        record = OptimizationCreatedRecord.from_experiment(
            experiment=experiment,
            generation_strategy=None,
            unique_identifier=get_unique_identifier(experiment=experiment),
            owner="bar",
            product_surface="foo",
            launch_surface="web",
            deployed_job_id=1118,
            trial_evaluation_identifier="train",
            is_manual_generation_strategy=True,
            warm_started_from=None,
            num_custom_trials=0,
        )

        expected_dict = {
            **asdict(ExperimentCreatedRecord.from_experiment(experiment=experiment)),
            "unique_identifier": "branin_test_experiment_2023-04-07 16:01:23",
            "owner": "bar",
            "product_surface": "foo",
            "launch_surface": "web",
            "deployed_job_id": 1118,
            "trial_evaluation_identifier": "train",
            "is_manual_generation_strategy": True,
            "warm_started_from": None,
            "num_custom_trials": 0,
            "arms_per_trial": -1,
            "early_stopping_strategy_cls": None,
            "global_stopping_strategy_cls": None,
            "scheduler_max_pending_trials": -1,
            "scheduler_total_trials": None,
            "support_tier": "",
            "transformed_dimensionality": None,
            "num_requested_bayesopt_trials": None,
            "num_requested_initialization_trials": None,
            "num_requested_other_trials": None,
            "max_parallelism": None,
            "generation_strategy_name": None,
        }
        self.maxDiff = None
        self.assertEqual(asdict(record), expected_dict)

    def test_optimization_created_record_from_experiment_with_generation_strategy(
        self,
    ) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objectives={"branin": ObjectiveProperties(minimize=True)},
            experiment_type="NOTEBOOK",
            is_test=True,
        )
        experiment = ax_client.experiment
        generation_strategy = ax_client.generation_strategy
        time_created_str = "2023-04-07T16:01:23.45"
        experiment._time_created = datetime.strptime(
            time_created_str, "%Y-%m-%dT%H:%M:%S.%f"
        )

        record = OptimizationCreatedRecord.from_experiment(
            experiment=experiment,
            generation_strategy=generation_strategy,
            unique_identifier=get_unique_identifier(experiment=experiment),
            owner="bar",
            product_surface="foo",
            launch_surface="web",
            deployed_job_id=1118,
            trial_evaluation_identifier="train",
            is_manual_generation_strategy=True,
            warm_started_from=None,
            num_custom_trials=0,
        )

        expected_dict = {
            **asdict(ExperimentCreatedRecord.from_experiment(experiment=experiment)),
            **asdict(
                GenerationStrategyCreatedRecord.from_generation_strategy(
                    generation_strategy=generation_strategy
                )
            ),
            "unique_identifier": "test_experiment_2023-04-07 16:01:23",
            "owner": "bar",
            "product_surface": "foo",
            "launch_surface": "web",
            "deployed_job_id": 1118,
            "trial_evaluation_identifier": "train",
            "is_manual_generation_strategy": True,
            "warm_started_from": None,
            "num_custom_trials": 0,
            "arms_per_trial": -1,
            "early_stopping_strategy_cls": None,
            "global_stopping_strategy_cls": None,
            "scheduler_max_pending_trials": -1,
            "scheduler_total_trials": None,
            "support_tier": "",
            "transformed_dimensionality": 2,
        }

        self.maxDiff = None
        self.assertEqual(asdict(record), expected_dict)

    def test_optimization_completed_record_from_scheduler(self) -> None:
        scheduler = Scheduler(
            experiment=get_branin_experiment(),
            generation_strategy=get_generation_strategy(),
            options=SchedulerOptions(
                total_trials=0,
                tolerated_trial_failure_rate=0.2,
                init_seconds_between_polls=10,
            ),
        )

        record = OptimizationCompletedRecord.from_scheduler(
            scheduler=scheduler,
            unique_identifier="foo",
            deployed_job_id=1118,
            estimated_early_stopping_savings=19,
            estimated_global_stopping_savings=98,
        )

        expected_dict = {
            **SchedulerCompletedRecord.from_scheduler(scheduler=scheduler).flatten(),
            "unique_identifier": "foo",
            "deployed_job_id": 1118,
            "estimated_early_stopping_savings": 19,
            "estimated_global_stopping_savings": 98,
        }

        self.assertDictsAlmostEqual(
            asdict(record), expected_dict, consider_nans_equal=True
        )

    def test_optimization_completed_record_from_ax_client(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objectives={"branin": ObjectiveProperties(minimize=True)},
            is_test=True,
        )

        record = OptimizationCompletedRecord.from_ax_client(
            ax_client=ax_client,
            unique_identifier="foo",
            deployed_job_id=1118,
            estimated_early_stopping_savings=19,
            estimated_global_stopping_savings=98,
        )
        expected_dict = {
            **AxClientCompletedRecord.from_ax_client(ax_client=ax_client).flatten(),
            "unique_identifier": "foo",
            "deployed_job_id": 1118,
            "estimated_early_stopping_savings": 19,
            "estimated_global_stopping_savings": 98,
            # Extra fields
            "improvement_over_baseline": float("nan"),
            "num_metric_fetch_e_encountered": -1,
            "num_trials_bad_due_to_err": -1,
        }

        self.assertDictsAlmostEqual(
            asdict(record), expected_dict, consider_nans_equal=True
        )

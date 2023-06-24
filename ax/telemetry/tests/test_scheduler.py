#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

from ax.core.experiment import Experiment
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.branin import BraninMetric
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from ax.service.scheduler import get_fitted_model_bridge, Scheduler, SchedulerOptions
from ax.telemetry.experiment import ExperimentCompletedRecord, ExperimentCreatedRecord
from ax.telemetry.generation_strategy import GenerationStrategyCreatedRecord
from ax.telemetry.scheduler import SchedulerCompletedRecord, SchedulerCreatedRecord
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment, get_branin_search_space
from ax.utils.testing.modeling_stubs import get_generation_strategy

NUM_SOBOL = 5


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
            best_point_quality=float("-inf"),
            model_fit_quality=float("-inf"),  # -inf because no model has been fit
            num_metric_fetch_e_encountered=0,
            num_trials_bad_due_to_err=0,
        )
        self.assertEqual(record, expected)

        flat = record.flatten()
        expected_dict = {
            **ExperimentCompletedRecord.from_experiment(
                experiment=scheduler.experiment
            ).__dict__,
            "best_point_quality": float("-inf"),
            "model_fit_quality": float("-inf"),
            "num_metric_fetch_e_encountered": 0,
            "num_trials_bad_due_to_err": 0,
        }
        self.assertEqual(flat, expected_dict)

    def test_scheduler_model_fit_metrics_logging(self) -> None:
        # set up for model fit metrics
        branin_experiment = Experiment(
            name="branin_test_experiment",
            search_space=get_branin_search_space(),
            runner=SyntheticRunner(),
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=BraninMetric(name="branin", param_names=["x1", "x2"]),
                    minimize=True,
                ),
            ),
            is_test=True,
        )
        branin_experiment._properties[Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF] = True
        generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL, num_trials=NUM_SOBOL, max_parallelism=NUM_SOBOL
                ),
                GenerationStep(model=Models.GPEI, num_trials=-1),
            ]
        )

        # starting proper tests
        scheduler = Scheduler(
            experiment=branin_experiment,
            generation_strategy=generation_strategy,
            options=SchedulerOptions(),
        )
        # Trying to attain a record without any trials yields an error in ModelFitRecord
        # and a warning in SchedulerCompletedRecord.

        scheduler.run_n_trials(max_trials=NUM_SOBOL + 1)

        # end-to-end test with Scheduler
        record = SchedulerCompletedRecord.from_scheduler(scheduler=scheduler)
        model_bridge = get_fitted_model_bridge(scheduler)
        fit_metrics = model_bridge.compute_model_fit_metrics(
            experiment=scheduler.experiment
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
        # log_std_branin = math.log10(std_branin)
        # model_std_quality = -log_std_branin  # align positivity with over-estimation

        expected = SchedulerCompletedRecord(
            experiment_completed_record=ExperimentCompletedRecord.from_experiment(
                experiment=scheduler.experiment
            ),
            best_point_quality=float("-inf"),
            model_fit_quality=r2_branin,
            num_metric_fetch_e_encountered=0,
            num_trials_bad_due_to_err=0,
        )
        self.assertEqual(record, expected)

        flat = record.flatten()
        expected_dict = {
            **ExperimentCompletedRecord.from_experiment(
                experiment=scheduler.experiment
            ).__dict__,
            "best_point_quality": float("-inf"),
            "model_fit_quality": r2_branin,
            "num_metric_fetch_e_encountered": 0,
            "num_trials_bad_due_to_err": 0,
        }
        self.assertEqual(flat, expected_dict)

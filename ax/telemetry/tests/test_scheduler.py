#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import cast, Dict
from unittest import mock

import numpy as np

from ax.core.experiment import Experiment
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.branin import BraninMetric
from ax.modelbridge.cross_validation import compute_model_fit_metrics_from_modelbridge
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
                    generation_strategy=scheduler.standard_generation_strategy
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
                generation_strategy=scheduler.standard_generation_strategy
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

        with mock.patch.object(
            scheduler, "get_improvement_over_baseline", return_value=5.0
        ):
            record = SchedulerCompletedRecord.from_scheduler(scheduler=scheduler)
        expected = SchedulerCompletedRecord(
            experiment_completed_record=ExperimentCompletedRecord.from_experiment(
                experiment=scheduler.experiment
            ),
            best_point_quality=float("nan"),
            model_fit_quality=float("nan"),  # nan because no model has been fit
            model_std_quality=float("nan"),
            model_fit_generalization=float("nan"),
            model_std_generalization=float("nan"),
            improvement_over_baseline=5.0,
            num_metric_fetch_e_encountered=0,
            num_trials_bad_due_to_err=0,
        )
        self._compare_scheduler_completed_records(record, expected)

        flat = record.flatten()
        expected_dict = {
            **ExperimentCompletedRecord.from_experiment(
                experiment=scheduler.experiment
            ).__dict__,
            "best_point_quality": float("nan"),
            "model_fit_quality": float("nan"),
            "model_std_quality": float("nan"),
            "model_fit_generalization": float("nan"),
            "model_std_generalization": float("nan"),
            "improvement_over_baseline": 5.0,
            "num_metric_fetch_e_encountered": 0,
            "num_trials_bad_due_to_err": 0,
        }
        self.assertDictsAlmostEqual(flat, expected_dict, consider_nans_equal=True)

    def test_scheduler_raise_exceptions(self) -> None:
        scheduler = Scheduler(
            experiment=get_branin_experiment(),
            generation_strategy=get_generation_strategy(),
            options=SchedulerOptions(
                total_trials=0,
                tolerated_trial_failure_rate=0.2,
                init_seconds_between_polls=10,
            ),
        )

        with mock.patch.object(
            scheduler,
            "get_improvement_over_baseline",
            side_effect=Exception("test_exception"),
        ):
            record = SchedulerCompletedRecord.from_scheduler(scheduler=scheduler)
        flat = record.flatten()
        self.assertTrue(np.isnan(flat["improvement_over_baseline"]))

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

        fit_metrics = compute_model_fit_metrics_from_modelbridge(
            model_bridge=model_bridge,
            experiment=scheduler.experiment,
            generalization=False,
            untransform=False,
        )
        # checking fit metrics
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

        model_std_quality = 1 / std_branin

        # check generalization metrics
        gen_metrics = compute_model_fit_metrics_from_modelbridge(
            model_bridge=model_bridge,
            experiment=scheduler.experiment,
            generalization=True,
            untransform=False,
        )
        r2_gen = gen_metrics.get("coefficient_of_determination")
        r2_gen = cast(Dict[str, float], r2_gen)
        r2_gen_branin = r2_gen["branin"]
        gen_std = gen_metrics.get("std_of_the_standardized_error")
        gen_std = cast(Dict[str, float], gen_std)
        gen_std_branin = gen_std["branin"]
        model_std_generalization = 1 / gen_std_branin

        expected = SchedulerCompletedRecord(
            experiment_completed_record=ExperimentCompletedRecord.from_experiment(
                experiment=scheduler.experiment
            ),
            best_point_quality=float("nan"),
            model_fit_quality=r2_branin,
            model_std_quality=model_std_quality,
            model_fit_generalization=r2_gen_branin,
            model_std_generalization=model_std_generalization,
            improvement_over_baseline=float("nan"),
            num_metric_fetch_e_encountered=0,
            num_trials_bad_due_to_err=0,
        )
        self._compare_scheduler_completed_records(record, expected)

        flat = record.flatten()
        expected_dict = {
            **ExperimentCompletedRecord.from_experiment(
                experiment=scheduler.experiment
            ).__dict__,
            "best_point_quality": float("nan"),
            "model_fit_quality": r2_branin,
            "model_std_quality": model_std_quality,
            "model_fit_generalization": r2_gen_branin,
            "model_std_generalization": model_std_generalization,
            "improvement_over_baseline": float("nan"),
            "num_metric_fetch_e_encountered": 0,
            "num_trials_bad_due_to_err": 0,
        }
        self.assertDictsAlmostEqual(flat, expected_dict, consider_nans_equal=True)

    def _compare_scheduler_completed_records(
        self, record: SchedulerCompletedRecord, expected: SchedulerCompletedRecord
    ) -> None:
        self.assertEqual(
            record.experiment_completed_record, expected.experiment_completed_record
        )
        numeric_fields = [
            "best_point_quality",
            "model_fit_quality",
            "model_std_quality",
            "model_fit_generalization",
            "model_std_generalization",
            "improvement_over_baseline",
            "num_metric_fetch_e_encountered",
            "num_trials_bad_due_to_err",
        ]
        for field in numeric_fields:
            rec_field = getattr(record, field)
            exp_field = getattr(expected, field)
            if np.isnan(rec_field):
                self.assertTrue(np.isnan(exp_field))
            else:
                self.assertAlmostEqual(rec_field, exp_field)

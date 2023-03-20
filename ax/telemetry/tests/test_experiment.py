#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.utils import get_model_times
from ax.telemetry.experiment import ExperimentCompletedRecord, ExperimentCreatedRecord
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_custom_runner_and_metric


class TestExperiment(TestCase):
    def test_experiment_created_record_from_experiment(self) -> None:
        experiment = get_experiment_with_custom_runner_and_metric()

        record = ExperimentCreatedRecord.from_experiment(experiment=experiment)
        expected = ExperimentCreatedRecord(
            experiment_name="test",
            experiment_type=None,
            num_continuous_range_parameters=1,
            num_int_range_parameters_small=0,
            num_int_range_parameters_medium=0,
            num_int_range_parameters_large=1,
            num_log_scale_range_parameters=0,
            num_unordered_choice_parameters_small=1,
            num_unordered_choice_parameters_medium=0,
            num_unordered_choice_parameters_large=0,
            num_fixed_parameters=1,
            dimensionality=3,
            hierarchical_tree_height=1,
            num_parameter_constraints=3,
            num_objectives=1,
            num_tracking_metrics=1,
            num_outcome_constraints=1,
            num_map_metrics=0,
            metric_cls_to_quantity={"Metric": 2, "CustomTestMetric": 1},
            runner_cls="CustomTestRunner",
        )
        self.assertEqual(record, expected)

    def test_experiment_completed_record_from_experiment(self) -> None:
        experiment = get_experiment_with_custom_runner_and_metric()

        record = ExperimentCompletedRecord.from_experiment(experiment=experiment)

        # Calculate these here, may change from run to run
        fit_time, gen_time = get_model_times(experiment=experiment)
        expected = ExperimentCompletedRecord(
            num_initialization_trials=1,
            num_bayesopt_trials=0,
            num_other_trials=0,
            num_completed_trials=1,
            num_failed_trials=0,
            num_abandoned_trials=0,
            num_early_stopped_trials=0,
            total_fit_time=int(fit_time),
            total_gen_time=int(gen_time),
        )
        self.assertEqual(record, expected)

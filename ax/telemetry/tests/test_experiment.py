#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.utils import get_model_times
from ax.modelbridge.registry import Models
from ax.telemetry.experiment import ExperimentCompletedRecord, ExperimentCreatedRecord
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_experiment_with_custom_runner_and_metric,
)
from ax.utils.testing.mock import fast_botorch_optimize


class TestExperiment(TestCase):
    def test_experiment_created_record_from_experiment(self) -> None:
        experiment = get_experiment_with_custom_runner_and_metric(
            has_outcome_constraint=True
        )

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
        experiment = get_experiment_with_custom_runner_and_metric(
            has_outcome_constraint=True, num_trials=1
        )
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

    @fast_botorch_optimize
    def test_bayesopt_trials_are_trials_containing_bayesopt(self) -> None:
        experiment = get_branin_experiment()
        sobol = Models.SOBOL(search_space=experiment.search_space)
        trial = experiment.new_batch_trial().add_generator_run(
            generator_run=sobol.gen(5)
        )
        trial.mark_completed(unsafe=True)

        # create a trial that among other things does bayesopt
        data = experiment.fetch_data()
        botorch = Models.BOTORCH_MODULAR(
            experiment=experiment,
            data=data,
        )
        trial = (
            experiment.new_batch_trial()
            .add_generator_run(generator_run=sobol.gen(2))
            .add_generator_run(generator_run=botorch.gen(5))
        )
        trial.add_arm(experiment.arms_by_name["0_0"])
        trial.mark_completed(unsafe=True)

        # create another BO trial but leave it as a candidate
        botorch = Models.BOTORCH_MODULAR(
            experiment=experiment,
            data=data,
        )
        trial = experiment.new_batch_trial().add_generator_run(
            generator_run=botorch.gen(5)
        )

        record = ExperimentCompletedRecord.from_experiment(experiment=experiment)
        self.assertEqual(record.num_initialization_trials, 1)
        self.assertEqual(record.num_bayesopt_trials, 1)
        self.assertEqual(record.num_other_trials, 0)

    def test_other_trials_are_trials_with_no_models(self) -> None:
        experiment = get_branin_experiment()
        sobol = Models.SOBOL(search_space=experiment.search_space)
        trial = experiment.new_batch_trial().add_generator_run(
            generator_run=sobol.gen(5)
        )
        trial.mark_completed(unsafe=True)

        # create a trial that has no GRs that used models
        trial = experiment.new_batch_trial()
        trial.add_arm(experiment.arms_by_name["0_0"])
        trial.mark_completed(unsafe=True)

        record = ExperimentCompletedRecord.from_experiment(experiment=experiment)
        self.assertEqual(record.num_initialization_trials, 1)
        self.assertEqual(record.num_bayesopt_trials, 0)
        self.assertEqual(record.num_other_trials, 1)

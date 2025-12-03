# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import Mock

import pandas as pd
from ax.adapter.registry import Generators
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.trial import Trial
from ax.exceptions.core import DataRequiredError
from ax.service.utils.best_point import get_trace
from ax.service.utils.best_point_mixin import BestPointMixin
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment_with_batch_trial,
    get_experiment_with_observations,
    get_experiment_with_trial,
)
from pyre_extensions import assert_is_instance, none_throws


class TestBestPointMixin(TestCase):
    def test_get_trace(self) -> None:
        # Single objective, minimize.
        exp = get_experiment_with_observations(
            observations=[[11], [10], [9], [15], [5]], minimize=True
        )
        self.assertEqual(get_trace(exp), [11, 10, 9, 9, 5])

        # Same experiment with maximize via new optimization config.
        opt_conf = none_throws(exp.optimization_config).clone()
        opt_conf.objective.minimize = False
        self.assertEqual(get_trace(exp, opt_conf), [11, 11, 11, 15, 15])

        with self.subTest("Single objective with constraints"):
            # The second metric is the constraint and needs to be >= 0
            exp = get_experiment_with_observations(
                observations=[[11, -1], [10, 1], [9, 1], [15, -1], [11, 1]],
                minimize=False,
                constrained=True,
            )
            self.assertEqual(get_trace(exp), [float("-inf"), 10, 10, 10, 11])

            exp = get_experiment_with_observations(
                observations=[[11, -1], [10, 1], [9, 1], [15, -1], [11, 1]],
                minimize=True,
                constrained=True,
            )
            self.assertEqual(get_trace(exp), [float("inf"), 10, 9, 9, 9])

        # Scalarized.
        exp = get_experiment_with_observations(
            observations=[[1, 1], [2, 2], [3, 3]],
            scalarized=True,
        )
        self.assertEqual(get_trace(exp), [2, 4, 6])

        # Multi objective.
        exp = get_experiment_with_observations(
            observations=[[1, 1], [-1, 100], [1, 2], [3, 3], [2, 4], [2, 1]],
        )
        self.assertEqual(get_trace(exp), [1, 1, 2, 9, 11, 11])

        # W/o ObjectiveThresholds (infering ObjectiveThresholds from nadir point)
        assert_is_instance(
            exp.optimization_config, MultiObjectiveOptimizationConfig
        ).objective_thresholds = []
        self.assertEqual(get_trace(exp), [0.0, 0.0, 2.0, 8.0, 11.0, 11.0])

        # Multi-objective w/ constraints.
        exp = get_experiment_with_observations(
            observations=[[-1, 1, 1], [1, 2, 1], [3, 3, -1], [2, 4, 1], [2, 1, 1]],
            constrained=True,
        )
        self.assertEqual(get_trace(exp), [0, 2, 2, 8, 8])

        # W/ relative constraints & status quo.
        exp.status_quo = Arm(parameters={"x": 0.5, "y": 0.5}, name="status_quo")
        exp.optimization_config.outcome_constraints[0].bound = 1.0
        exp.optimization_config.outcome_constraints[0].relative = True
        # Fails if there's no data for status quo.
        with self.assertRaisesRegex(DataRequiredError, "relative constraint"):
            get_trace(exp)
        # Add data for status quo.
        trial = Trial(experiment=exp).add_arm(arm=exp.status_quo).run().mark_completed()
        df_dict = [
            {
                "trial_index": trial.index,
                "metric_name": m,
                "arm_name": "status_quo",
                "mean": 0.0,
                "sem": 0.0,
                "metric_signature": m,
            }
            for m in ["m1", "m2", "m3"]
        ]
        status_quo_data = Data(df=pd.DataFrame.from_records(df_dict))
        exp.attach_data(data=status_quo_data)
        self.assertEqual(get_trace(exp), [0, 2, 2, 8, 8])

        # W/ first objective being minimized.
        exp = get_experiment_with_observations(
            observations=[[1, 1], [-1, 2], [3, 3], [-2, 4], [2, 1]], minimize=True
        )
        self.assertEqual(get_trace(exp), [0, 2, 2, 8, 8])

        # W/ empty data.
        exp = get_experiment_with_trial()
        self.assertEqual(get_trace(exp), [])

        # test batch trial
        exp = get_experiment_with_batch_trial(with_status_quo=False)
        trial = exp.trials[0]
        exp.optimization_config.outcome_constraints[0].relative = False
        trial.mark_running(no_runner_required=True).mark_completed()
        df_dict = []
        for i, arm in enumerate(trial.arms):
            df_dict.extend(
                [
                    {
                        "trial_index": 0,
                        "metric_name": m,
                        "arm_name": arm.name,
                        "mean": float(i),
                        "sem": 0.0,
                        "metric_signature": m,
                    }
                    for m in exp.metrics.keys()
                ]
            )
        exp.attach_data(Data(df=pd.DataFrame.from_records(df_dict)))
        self.assertEqual(get_trace(exp), [2.0])
        # test that there is performance metric in the trace for each
        # completed/early-stopped trial
        trial1 = assert_is_instance(trial, BatchTrial).clone_to(include_sq=False)
        trial1.mark_abandoned(unsafe=True)
        trial2 = exp.new_batch_trial(Generators.SOBOL(experiment=exp).gen(n=3))
        trial2.mark_running(no_runner_required=True).mark_completed()
        df_dict2 = []
        for i, arm in enumerate(trial2.arms):
            df_dict2.extend(
                [
                    {
                        "trial_index": 2,
                        "metric_name": m,
                        "arm_name": arm.name,
                        "mean": 10 * float(i),
                        "sem": 0.0,
                        "metric_signature": m,
                    }
                    for m in exp.metrics.keys()
                ]
            )
        exp.attach_data(Data(df=pd.DataFrame.from_records(df_dict2)))
        self.assertEqual(get_trace(exp), [2.0, 20.0])

    def test_get_trace_with_include_status_quo(self) -> None:
        with self.subTest("Multi-objective: status quo dominates in some trials"):
            # Create experiment with multi-objective optimization where status quo
            # is deliberately the best arm in some trials to test include_status_quo.
            exp = get_experiment_with_observations(
                observations=[[1, 1], [-1, 2], [3, 3]], minimize=True
            )

            # Set up status quo
            exp.status_quo = Arm(parameters={"x": 0.5, "y": 0.5}, name="status_quo")

            # Create batch trial where status quo DOMINATES other arms
            batch_trial1 = exp.new_batch_trial(should_add_status_quo_arm=True)
            batch_trial1.add_arm(
                Arm(parameters={"x": 0.1, "y": 0.1}, name="poor_arm_1")
            )
            batch_trial1.add_arm(
                Arm(parameters={"x": 0.2, "y": 0.2}, name="poor_arm_2")
            )

            # Data: Status quo has excellent values, other arms are poor
            df_dict1 = [
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": "m1",
                    "arm_name": "poor_arm_1",
                    "mean": 10.0,
                    "sem": 0.0,
                    "metric_signature": "m1",
                },
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": "m2",
                    "arm_name": "poor_arm_1",
                    "mean": -5.0,
                    "sem": 0.0,
                    "metric_signature": "m2",
                },
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": "m1",
                    "arm_name": "poor_arm_2",
                    "mean": 12.0,
                    "sem": 0.0,
                    "metric_signature": "m1",
                },
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": "m2",
                    "arm_name": "poor_arm_2",
                    "mean": -3.0,
                    "sem": 0.0,
                    "metric_signature": "m2",
                },
                # Status quo: excellent in both objectives
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": "m1",
                    "arm_name": "status_quo",
                    "mean": -10.0,
                    "sem": 0.0,
                    "metric_signature": "m1",
                },
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": "m2",
                    "arm_name": "status_quo",
                    "mean": 10.0,
                    "sem": 0.0,
                    "metric_signature": "m2",
                },
            ]
            exp.attach_data(Data(df=pd.DataFrame.from_records(df_dict1)))
            batch_trial1.mark_running(no_runner_required=True).mark_completed()

            # Get trace without status quo
            trace_without_sq = get_trace(exp, include_status_quo=False)

            # Get trace with status quo
            trace_with_sq = get_trace(exp, include_status_quo=True)

            # Both should have 4 trace values (3 initial + 1 batch trial)
            self.assertEqual(len(trace_without_sq), 4)
            self.assertEqual(len(trace_with_sq), 4)

            # The last value MUST differ because status quo dominates
            # Without status quo, only poor arms contribute (low hypervolume)
            # With status quo, excellent values contribute (high hypervolume)
            self.assertGreater(
                trace_with_sq[-1],
                trace_without_sq[-1],
                f"Status quo dominates in trial 3, so trace with SQ should be higher. "
                f"Without SQ: {trace_without_sq}, With SQ: {trace_with_sq}",
            )

        with self.subTest("Single-objective: status quo is best in some trials"):
            # Create single-objective experiment where status quo is deliberately
            # the best arm in some trials.
            exp = get_experiment_with_observations(
                observations=[[11], [10], [9]], minimize=True
            )

            # Get the actual metric name from the experiment
            metric_name = list(exp.metrics.keys())[0]

            exp.status_quo = Arm(parameters={"x": 0.5, "y": 0.5}, name="status_quo")

            batch_trial1 = exp.new_batch_trial(should_add_status_quo_arm=True)
            batch_trial1.add_arm(
                Arm(parameters={"x": 0.1, "y": 0.1}, name="mediocre_arm_1")
            )
            batch_trial1.add_arm(
                Arm(parameters={"x": 0.2, "y": 0.2}, name="mediocre_arm_2")
            )

            df_dict1 = [
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": metric_name,
                    "arm_name": "mediocre_arm_1",
                    "mean": 15.0,
                    "sem": 0.0,
                    "metric_signature": metric_name,
                },
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": metric_name,
                    "arm_name": "mediocre_arm_2",
                    "mean": 20.0,
                    "sem": 0.0,
                    "metric_signature": metric_name,
                },
                # Status quo: best value (lowest)
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": metric_name,
                    "arm_name": "status_quo",
                    "mean": 5.0,
                    "sem": 0.0,
                    "metric_signature": metric_name,
                },
            ]
            exp.attach_data(Data(df=pd.DataFrame.from_records(df_dict1)))
            batch_trial1.mark_running(no_runner_required=True).mark_completed()

            # Get trace without status quo
            trace_without_sq = get_trace(exp, include_status_quo=False)

        with self.subTest("Include status quo: status quo included in trace"):
            # Get trace with status quo
            trace_with_sq = get_trace(exp, include_status_quo=True)

            # Both should have 4 values (3 initial + 1 batch trial)
            self.assertEqual(len(trace_without_sq), 4)
            self.assertEqual(len(trace_with_sq), 4)

            # The last value MUST differ because status quo is best
            # Without status quo: best in trial 3 is 15.0, cumulative min is 9
            # With status quo: best in trial 3 is 5.0, cumulative min is 5
            self.assertLess(
                trace_with_sq[-1],
                trace_without_sq[-1],
                f"Status quo is best in trial 3, so trace with SQ should be "
                f"lower (minimize). Without SQ: {trace_without_sq}, "
                f"With SQ: {trace_with_sq}",
            )

    def test_get_hypervolume(self) -> None:
        # W/ empty data.
        exp = get_experiment_with_trial()
        self.assertEqual(BestPointMixin._get_hypervolume(exp, Mock()), 0.0)

    def test_get_best_observed_value(self) -> None:
        # Alias for easier access.
        get_best = BestPointMixin._get_best_observed_value

        # Single objective, minimize.
        exp = get_experiment_with_observations(
            observations=[[11], [10], [9], [15], [5]], minimize=True
        )
        self.assertEqual(get_best(exp), 5)
        # Same experiment with maximize via new optimization config.
        opt_conf = none_throws(exp.optimization_config).clone()
        opt_conf.objective.minimize = False
        self.assertEqual(get_best(exp, opt_conf), 15)

        # Scalarized.
        exp = get_experiment_with_observations(
            observations=[[1, 1], [2, 2], [3, 3]],
            scalarized=True,
        )
        self.assertEqual(get_best(exp), 6)

        # Exclude out of design arms
        exp = get_experiment_with_observations(
            observations=[[11], [10], [9], [15], [5]],
            parameterizations=[
                {"x": 0.0, "y": 0.0},
                {"x": 0.1, "y": 0.0},
                {"x": 10.0, "y": 10.0},  # out of design
                {"x": 0.2, "y": 0.0},
                {"x": 10.1, "y": 10.0},  # out of design
            ],
            minimize=True,
        )
        self.assertEqual(get_best(exp), 10)  # 5 and 9 are out of design

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
        trial = Trial(experiment=exp).add_arm(arm=exp.status_quo)
        df_dict = [
            {
                "trial_index": trial.index,
                "metric_name": m,
                "arm_name": "status_quo",
                "mean": 0.0,
                "sem": 0.0,
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
        exp = get_experiment_with_batch_trial()
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
                    }
                    for m in exp.metrics.keys()
                ]
            )
        exp.attach_data(Data(df=pd.DataFrame.from_records(df_dict)))
        self.assertEqual(get_trace(exp), [len(trial.arms) - 1])
        # test that there is performance metric in the trace for each
        # completed/early-stopped trial
        trial1 = assert_is_instance(trial, BatchTrial).clone_to()
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
                    }
                    for m in exp.metrics.keys()
                ]
            )
        exp.attach_data(Data(df=pd.DataFrame.from_records(df_dict2)))
        self.assertEqual(get_trace(exp), [2, 20.0])

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

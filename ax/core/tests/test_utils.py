#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from ax.core.data import Data
from ax.core.metric import Metric
from ax.core.objective import Objective, MultiObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint, ObjectiveThreshold
from ax.core.types import ComparisonOp
from ax.core.utils import (
    MissingMetrics,
    best_feasible_objective,
    get_missing_metrics,
    get_missing_metrics_by_name,
    feasible_hypervolume,
)
from ax.utils.common.testutils import TestCase


class UtilsTest(TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            [
                {
                    "arm_name": "0_0",
                    "mean": 2.0,
                    "sem": 0.2,
                    "trial_index": 1,
                    "metric_name": "a",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_0",
                    "mean": 1.8,
                    "sem": 0.3,
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_1",
                    "mean": float("nan"),
                    "sem": float("nan"),
                    "trial_index": 1,
                    "metric_name": "a",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_1",
                    "mean": 3.7,
                    "sem": 0.5,
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_2",
                    "mean": 0.5,
                    "sem": None,
                    "trial_index": 1,
                    "metric_name": "a",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_2",
                    "mean": float("nan"),
                    "sem": float("nan"),
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_2",
                    "mean": float("nan"),
                    "sem": float("nan"),
                    "trial_index": 1,
                    "metric_name": "c",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
            ]
        )

        self.data = Data(df=self.df)

        self.optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric(name="a")),
            outcome_constraints=[
                OutcomeConstraint(
                    metric=Metric(name="b"),
                    op=ComparisonOp.GEQ,
                    bound=0,
                    relative=False,
                )
            ],
        )

    def test_get_missing_metrics_by_name(self):
        expected = {"a": {("0_1", 1)}, "b": {("0_2", 1)}}
        actual = get_missing_metrics_by_name(self.data, ["a", "b"])
        self.assertEqual(actual, expected)

    def test_get_missing_metrics(self):
        expected = MissingMetrics(
            {"a": {("0_1", 1)}},
            {"b": {("0_2", 1)}},
            {"c": {("0_0", 1), ("0_1", 1), ("0_2", 1)}},
        )
        actual = get_missing_metrics(self.data, self.optimization_config)
        self.assertEqual(actual, expected)

    def test_best_feasible_objective(self):
        bfo = best_feasible_objective(
            self.optimization_config,
            values={"a": np.array([1.0, 3.0, 2.0]), "b": np.array([0.0, -1.0, 0.0])},
        )
        self.assertEqual(list(bfo), [1.0, 1.0, 2.0])

    def test_feasible_hypervolume(self):
        ma = Metric(name="a", lower_is_better=False)
        mb = Metric(name="b", lower_is_better=True)
        mc = Metric(name="c", lower_is_better=False)
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(metrics=[ma, mb]),
            outcome_constraints=[
                OutcomeConstraint(
                    mc,
                    op=ComparisonOp.GEQ,
                    bound=0,
                    relative=False,
                )
            ],
            objective_thresholds=[
                ObjectiveThreshold(
                    ma,
                    bound=1.0,
                ),
                ObjectiveThreshold(
                    mb,
                    bound=1.0,
                ),
            ],
        )
        feas_hv = feasible_hypervolume(
            optimization_config,
            values={
                "a": np.array(
                    [
                        1.0,
                        3.0,
                        2.0,
                        2.0,
                    ]
                ),
                "b": np.array(
                    [
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                    ]
                ),
                "c": np.array(
                    [
                        0.0,
                        -0.0,
                        1.0,
                        -2.0,
                    ]
                ),
            },
        )
        self.assertEqual(list(feas_hv), [0.0, 0.0, 1.0, 1.0])

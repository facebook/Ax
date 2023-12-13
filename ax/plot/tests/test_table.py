#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import (
    ComparisonOp,
    ObjectiveThreshold,
    OutcomeConstraint,
)
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.plot.table_view import metric_summary_df, search_space_summary_df
from ax.utils.common.testutils import TestCase


class TestMetric(Metric):
    pass


class TracesTest(TestCase):
    def setUp(self) -> None:
        self.min_search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x1", parameter_type=ParameterType.INT, lower=0, upper=2
                ),
                RangeParameter(
                    "x2",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.1,
                    upper=10,
                ),
            ]
        )
        self.max_search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x1", parameter_type=ParameterType.INT, lower=0, upper=2
                ),
                RangeParameter(
                    "x2",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.1,
                    upper=10,
                    log_scale=True,
                    is_fidelity=True,
                    target_value=10,
                ),
                FixedParameter("x3", parameter_type=ParameterType.BOOL, value=True),
                ChoiceParameter(
                    "x4",
                    parameter_type=ParameterType.STRING,
                    values=["a", "b", "c"],
                    is_ordered=False,
                    dependents={"a": ["x1", "x2"], "b": ["x4", "x5"], "c": ["x6"]},
                ),
                ChoiceParameter(
                    "x5",
                    parameter_type=ParameterType.STRING,
                    values=["d", "e", "f"],
                    is_ordered=True,
                ),
                RangeParameter(
                    name="x6",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=1.0,
                ),
            ]
        )

    def test_search_space_summary_df(self) -> None:
        df = search_space_summary_df(self.max_search_space)
        expected_df = pd.DataFrame(
            data={
                "Name": ["x1", "x2", "x3", "x4", "x5", "x6"],
                "Type": ["Range", "Range", "Fixed", "Choice", "Choice", "Range"],
                "Domain": [
                    "range=[0, 2]",
                    "range=[0.1, 10.0]",
                    "value=True",
                    "values=['a', 'b', 'c']",
                    "values=['d', 'e', 'f']",
                    "range=[0.0, 1.0]",
                ],
                "Datatype": ["int", "float", "bool", "string", "string", "float"],
                "Flags": [
                    "None",
                    "fidelity, log_scale",
                    "None",
                    "unordered, hierarchical, unsorted",
                    "ordered, unsorted",
                    "None",
                ],
                "Target Value": ["None", 10.0, "None", "None", "None", "None"],
                "Dependent Parameters": [
                    "None",
                    "None",
                    "None",
                    {"a": ["x1", "x2"], "b": ["x4", "x5"], "c": ["x6"]},
                    "None",
                    "None",
                ],
            }
        )

        pd.testing.assert_frame_equal(df, expected_df)

        df = search_space_summary_df(self.min_search_space)
        expected_df = pd.DataFrame(
            data={
                "Name": ["x1", "x2"],
                "Type": ["Range", "Range"],
                "Domain": ["range=[0, 2]", "range=[0.1, 10.0]"],
                "Datatype": ["int", "float"],
            }
        )
        pd.testing.assert_frame_equal(df, expected_df)

    def test_metric_summary_df(self) -> None:
        experiment = Experiment(
            name="test_experiment",
            search_space=SearchSpace(parameters=[]),
            optimization_config=MultiObjectiveOptimizationConfig(
                objective=MultiObjective(
                    objectives=[
                        Objective(
                            metric=Metric(name="my_objective_1", lower_is_better=True),
                            minimize=True,
                        ),
                        Objective(
                            metric=TestMetric(name="my_objective_2"), minimize=False
                        ),
                    ]
                ),
                objective_thresholds=[
                    ObjectiveThreshold(
                        metric=TestMetric(name="my_objective_2"),
                        bound=5.1,
                        relative=False,
                        op=ComparisonOp.GEQ,
                    )
                ],
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=Metric(name="my_constraint_1", lower_is_better=False),
                        bound=1,
                        relative=True,
                        op=ComparisonOp.GEQ,
                    ),
                    OutcomeConstraint(
                        metric=TestMetric(name="my_constraint_2"),
                        bound=-7.8,
                        relative=False,
                        op=ComparisonOp.LEQ,
                    ),
                ],
            ),
            tracking_metrics=[
                Metric(name="my_tracking_metric_1", lower_is_better=True),
                TestMetric(name="my_tracking_metric_2", lower_is_better=False),
                Metric(name="my_tracking_metric_3"),
            ],
        )
        df = metric_summary_df(experiment)
        expected_df = pd.DataFrame(
            data={
                "Name": [
                    "my_objective_1",
                    "my_objective_2",
                    "my_constraint_1",
                    "my_constraint_2",
                    "my_tracking_metric_1",
                    "my_tracking_metric_2",
                    "my_tracking_metric_3",
                ],
                "Type": [
                    "Metric",
                    "TestMetric",
                    "Metric",
                    "TestMetric",
                    "Metric",
                    "TestMetric",
                    "Metric",
                ],
                "Goal": [
                    "minimize",
                    "maximize",
                    "constrain",
                    "constrain",
                    "track",
                    "track",
                    "track",
                ],
                "Bound": ["None", ">= 5.1", ">= 1%", "<= -7.8", "None", "None", "None"],
                "Lower is Better": [True, "None", False, "None", True, False, "None"],
            }
        )
        expected_df["Goal"] = pd.Categorical(
            df["Goal"],
            categories=["minimize", "maximize", "constrain", "track", "None"],
            ordered=True,
        )
        pd.testing.assert_frame_equal(df, expected_df)

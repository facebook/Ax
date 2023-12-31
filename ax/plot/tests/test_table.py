#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    Parameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.plot.table_view import search_space_summary_df
from ax.utils.common.testutils import TestCase


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
                    "bounds: [0, 2]",
                    "bounds: [0.1, 10.0]",
                    "value: True",
                    "values: ['a', 'b', 'c']",
                    "values: ['d', 'e', 'f']",
                    "bounds: [0.0, 1.0]",
                ],
                "Datatype": ["int", "float", "bool", "string", "string", "float"],
                "Flags": [
                    "None",
                    "fidelity, log_scale",
                    "None",
                    "unordered, hierarchical",
                    "ordered",
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
                "Domain": ["bounds: [0, 2]", "bounds: [0.1, 10.0]"],
                "Datatype": ["int", "float"],
                "Flags": ["None", "None"],
            }
        )
        pd.testing.assert_frame_equal(df, expected_df)

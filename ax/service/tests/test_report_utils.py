#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
from unittest.mock import patch

import pandas as pd
from ax.core.arm import Arm
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.types import ComparisonOp
from ax.modelbridge.registry import Models
from ax.service.utils.report_utils import (
    _get_shortest_unique_suffix_dict,
    exp_to_df,
    Experiment,
    get_standard_plots,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_branin_experiment_with_timestamp_map_metric,
    get_multi_type_experiment,
)
from ax.utils.testing.mock import fast_botorch_optimize
from ax.utils.testing.modeling_stubs import get_generation_strategy
from plotly import graph_objects as go

OBJECTIVE_NAME = "branin"
PARAMETER_COLUMNS = ["x1", "x2"]
FLOAT_COLUMNS = [OBJECTIVE_NAME] + PARAMETER_COLUMNS
EXPECTED_COLUMNS = [
    "trial_index",
    "arm_name",
    "trial_status",
    "generation_method",
] + FLOAT_COLUMNS
DUMMY_OBJECTIVE_MEAN = 1.2345
DUMMY_SOURCE = "test_source"
DUMMY_MAP_KEY = "test_map_key"
TRUE_OBJECTIVE_NAME = "other_metric"
TRUE_OBJECTIVE_MEAN = 2.3456


class ReportUtilsTest(TestCase):
    def test_exp_to_df(self):
        # MultiTypeExperiment should fail
        exp = get_multi_type_experiment()
        with self.assertRaisesRegex(ValueError, "MultiTypeExperiment"):
            exp_to_df(exp=exp)

        # exp with no trials should return empty results
        exp = get_branin_experiment()
        df = exp_to_df(exp=exp)
        self.assertEqual(len(df), 0)

        # set up experiment
        exp = get_branin_experiment(with_batch=True)

        # check that pre-run experiment returns all columns except objective
        df = exp_to_df(exp)
        self.assertEqual(set(EXPECTED_COLUMNS) - set(df.columns), {OBJECTIVE_NAME})
        self.assertEqual(len(df.index), len(exp.arms_by_name))

        exp.trials[0].run()
        exp.fetch_data()

        # assert result is df with expected columns and length
        df = exp_to_df(exp=exp)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(sorted(df.columns), sorted(EXPECTED_COLUMNS))
        self.assertEqual(len(df.index), len(exp.arms_by_name))

        # test with run_metadata_fields and trial_properties_fields not empty
        # add source to properties
        for _, trial in exp.trials.items():
            trial._properties["source"] = DUMMY_SOURCE
        df = exp_to_df(
            exp, run_metadata_fields=["name"], trial_properties_fields=["source"]
        )
        self.assertIn("name", df.columns)
        self.assertIn("trial_properties_source", df.columns)

        # test column values or types
        self.assertTrue(all(x == 0 for x in df.trial_index))
        self.assertTrue(all(x == "RUNNING" for x in df.trial_status))
        self.assertTrue(all(x == "Sobol" for x in df.generation_method))
        self.assertTrue(all(x == DUMMY_SOURCE for x in df.trial_properties_source))
        self.assertTrue(all(x == "branin_test_experiment_0" for x in df.name))
        for float_column in FLOAT_COLUMNS:
            self.assertTrue(all(isinstance(x, float) for x in df[float_column]))

        # works correctly for failed trials (will need to mock)
        dummy_struct = namedtuple("dummy_struct", "df")
        mock_results = dummy_struct(
            df=pd.DataFrame(
                {
                    "arm_name": ["0_0"],
                    "metric_name": [OBJECTIVE_NAME],
                    "mean": [DUMMY_OBJECTIVE_MEAN],
                    "sem": [0],
                    "trial_index": [0],
                    "n": [123],
                    "frac_nonnull": [1],
                }
            )
        )
        with patch.object(Experiment, "lookup_data", lambda self: mock_results):
            df = exp_to_df(exp=exp)

        # all but one row should have a metric value of NaN
        self.assertEqual(pd.isna(df[OBJECTIVE_NAME]).sum(), len(df.index) - 1)

        # an experiment with more results than arms raises an error
        with patch.object(
            Experiment, "lookup_data", lambda self: mock_results
        ), self.assertRaisesRegex(ValueError, "inconsistent experimental state"):
            exp_to_df(exp=get_branin_experiment())

        # custom added trial has a generation_method of Manual
        custom_arm = Arm(name="custom", parameters={"x1": 0, "x2": 0})
        exp.new_trial().add_arm(custom_arm)
        df = exp_to_df(exp)
        self.assertEqual(
            df[df.arm_name == "custom"].iloc[0].generation_method, "Manual"
        )

    def test_get_shortest_unique_suffix_dict(self):
        expected_output = {
            "abc.123": "abc.123",
            "asdf.abc.123": "asdf.abc.123",
            "def.123": "def.123",
            "abc.456": "456",
            "": "",
            "no_delimiter": "no_delimiter",
        }
        actual_output = _get_shortest_unique_suffix_dict(
            ["abc.123", "abc.456", "def.123", "asdf.abc.123", "", "no_delimiter"]
        )
        self.assertDictEqual(expected_output, actual_output)

    @fast_botorch_optimize
    def test_get_standard_plots(self):
        exp = get_branin_experiment()
        self.assertEqual(
            len(
                get_standard_plots(
                    experiment=exp, model=get_generation_strategy().model
                )
            ),
            0,
        )
        exp = get_branin_experiment(with_batch=True, minimize=True)
        exp.trials[0].run()
        plots = get_standard_plots(
            experiment=exp,
            model=Models.BOTORCH(experiment=exp, data=exp.fetch_data()),
        )
        self.assertEqual(len(plots), 6)
        self.assertTrue(all(isinstance(plot, go.Figure) for plot in plots))
        exp = get_branin_experiment_with_multi_objective(with_batch=True)
        exp.optimization_config.objective.objectives[0].minimize = False
        exp.optimization_config.objective.objectives[1].minimize = True
        exp.optimization_config._objective_thresholds = [
            ObjectiveThreshold(
                metric=exp.metrics["branin_a"], op=ComparisonOp.GEQ, bound=-100.0
            ),
            ObjectiveThreshold(
                metric=exp.metrics["branin_b"], op=ComparisonOp.LEQ, bound=100.0
            ),
        ]
        exp.trials[0].run()
        plots = get_standard_plots(
            experiment=exp, model=Models.MOO(experiment=exp, data=exp.fetch_data())
        )
        self.assertEqual(len(plots), 7)

        # All plots are successfully created when objective thresholds are absent
        exp.optimization_config._objective_thresholds = []
        plots = get_standard_plots(
            experiment=exp, model=Models.MOO(experiment=exp, data=exp.fetch_data())
        )
        self.assertEqual(len(plots), 7)

        exp = get_branin_experiment_with_timestamp_map_metric(with_status_quo=True)
        exp.new_trial().add_arm(exp.status_quo)
        exp.trials[0].run()
        exp.new_trial(
            generator_run=Models.SOBOL(search_space=exp.search_space).gen(n=1)
        )
        exp.trials[1].run()
        plots = get_standard_plots(
            experiment=exp,
            model=Models.BOTORCH(experiment=exp, data=exp.fetch_data()),
            true_objective_metric_name="branin",
        )

        self.assertEqual(len(plots), 9)
        self.assertTrue(all(isinstance(plot, go.Figure) for plot in plots))
        self.assertIn(
            "Objective branin_map vs. True Objective Metric branin",
            [p.layout.title.text for p in plots],
        )

        with self.assertRaisesRegex(
            ValueError, "Please add a valid true_objective_metric_name"
        ):
            plots = get_standard_plots(
                experiment=exp,
                model=Models.BOTORCH(experiment=exp, data=exp.fetch_data()),
                true_objective_metric_name="not_present",
            )

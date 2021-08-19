#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
from unittest.mock import patch

import pandas as pd
from ax.core.arm import Arm
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.registry import Models
from ax.service.utils.report_utils import (
    _get_shortest_unique_suffix_dict,
    exp_to_df,
    get_best_trial,
    get_standard_plots,
    Experiment,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment, get_multi_type_experiment
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
        with patch.object(Experiment, "fetch_data", lambda self, metrics: mock_results):
            df = exp_to_df(exp=exp)

        # all but one row should have a metric value of NaN
        self.assertEqual(pd.isna(df[OBJECTIVE_NAME]).sum(), len(df.index) - 1)

        # an experiment with more results than arms raises an error
        with patch.object(
            Experiment, "fetch_data", lambda self, metrics: mock_results
        ), self.assertRaisesRegex(ValueError, "inconsistent experimental state"):
            exp_to_df(exp=get_branin_experiment())

        # custom added trial has a generation_method of Manual
        custom_arm = Arm(name="custom", parameters={"x1": 0, "x2": 0})
        exp.new_trial().add_arm(custom_arm)
        df = exp_to_df(exp)
        self.assertEqual(
            df[df.arm_name == "custom"].iloc[0].generation_method, "Manual"
        )

    def test_get_best_trial(self):
        exp = get_branin_experiment(with_batch=True, minimize=True)

        # exp with no completed trials should return None
        self.assertIsNone(get_best_trial(exp))

        # exp with completed trials should return optimal row
        # Hack in `noise_sd` value to ensure full reproducibility.
        exp.metrics[OBJECTIVE_NAME].noise_sd = 0.0
        exp.trials[0].run()
        df = exp_to_df(exp)
        best_trial = get_best_trial(exp)
        pd.testing.assert_frame_equal(
            df.sort_values(OBJECTIVE_NAME).head(1), best_trial
        )

        # exp with missing rows should return optimal row
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
        with patch.object(Experiment, "fetch_data", lambda self, metrics: mock_results):
            best_trial = get_best_trial(exp=exp)
        self.assertEqual(best_trial[OBJECTIVE_NAME][0], DUMMY_OBJECTIVE_MEAN)

        # when optimal objective is shared across multiple trials,
        # arbitrarily return a single optimal row
        mock_results = dummy_struct(
            df=pd.DataFrame(
                {
                    "arm_name": ["0_0", "0_1"],
                    "metric_name": [OBJECTIVE_NAME] * 2,
                    "mean": [DUMMY_OBJECTIVE_MEAN] * 2,
                    "sem": [0] * 2,
                    "trial_index": [0, 1],
                    "n": [123] * 2,
                    "frac_nonnull": [1] * 2,
                }
            )
        )
        with patch.object(Experiment, "fetch_data", lambda self, metrics: mock_results):
            best_trial = get_best_trial(exp=exp)
        self.assertEqual(len(best_trial.index), 1)
        self.assertEqual(best_trial[OBJECTIVE_NAME][0], DUMMY_OBJECTIVE_MEAN)

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
        gs = choose_generation_strategy(search_space=exp.search_space)
        gs._model = Models.BOTORCH(
            experiment=exp,
            data=exp.fetch_data(),
        )
        plots = get_standard_plots(experiment=exp, model=gs.model)
        self.assertEqual(len(plots), 3)
        self.assertTrue(all(isinstance(plot, go.Figure) for plot in plots))

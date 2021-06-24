#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
from unittest.mock import patch

import pandas as pd
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

EXPECTED_COLUMNS = [
    "branin",
    "trial_index",
    "arm_name",
    "x1",
    "x2",
    "trial_status",
    "generator_model",
]


class ReportUtilsTest(TestCase):
    def test_exp_to_df(self):
        # MultiTypeExperiment should fail
        exp = get_multi_type_experiment()
        with self.assertRaisesRegex(ValueError, "MultiTypeExperiment"):
            exp_to_df(exp=exp)
        # empty case - no trials, should return empty results
        exp = get_branin_experiment()
        df = exp_to_df(exp=exp)
        self.assertEqual(len(df), 0)
        # set up working experiment
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()
        # run_metadata_fields not List[str] should fail
        with self.assertRaisesRegex(
            ValueError, r"run_metadata_fields.*List\[str\] or None"
        ):
            exp_to_df(exp=exp, run_metadata_fields=[1, "asdf"])
        with self.assertRaisesRegex(
            ValueError, r"run_metadata_fields.*List\[str\] or None"
        ):
            exp_to_df(exp=exp, run_metadata_fields="asdf")

        # assert result is df with expected columns and length
        df = exp_to_df(exp=exp)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(list(df.columns), EXPECTED_COLUMNS)
        self.assertEqual(len(df.index), len(exp.arms_by_name))

        # test with run_metadata_fields not empty
        df = exp_to_df(exp, run_metadata_fields=["name"])
        self.assertIn("name", df.columns)

        # test column values
        self.assertTrue(all(x == 0 for x in df.trial_index))
        self.assertTrue(all(x == "RUNNING" for x in df.trial_status))
        self.assertTrue(all(x == "Sobol" for x in df.generator_model))
        self.assertTrue(all(x == "branin_test_experiment_0" for x in df.name))
        # works correctly for failed trials (will need to mock)
        dummy_struct = namedtuple("dummy_struct", "df")
        mock_results = dummy_struct(
            df=pd.DataFrame(
                {
                    "arm_name": ["0_0"],
                    "metric_name": ["branin"],
                    "mean": [0],
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
        self.assertEqual(pd.isna(df["branin"]).sum(), len(df.index) - 1)

        # an experiment with more results than arms raises an error
        with patch.object(
            Experiment, "fetch_data", lambda self, metrics: mock_results
        ), self.assertRaisesRegex(ValueError, "inconsistent experimental state"):
            exp_to_df(exp=get_branin_experiment())

    def test_get_best_trial(self):
        exp = get_branin_experiment(with_batch=True, minimize=True)
        # Hack in `noise_sd` value to ensure full reproducibility.
        exp.metrics["branin"].noise_sd = 0.0
        exp.trials[0].run()
        df = exp_to_df(exp)
        best_trial = get_best_trial(exp)
        pd.testing.assert_frame_equal(df.sort_values("branin").head(1), best_trial)

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
                    experiment=exp, generation_strategy=get_generation_strategy()
                )
            ),
            0,
        )

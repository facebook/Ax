#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from ax.service.utils.report_utils import (
    _get_shortest_unique_suffix_dict,
    exp_to_df,
    get_best_trial,
    get_standard_plots,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
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
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()
        df = exp_to_df(exp)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(list(df.columns), EXPECTED_COLUMNS)
        df = exp_to_df(exp, run_metadata_fields=["name"])
        self.assertIn("name", df.columns)

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

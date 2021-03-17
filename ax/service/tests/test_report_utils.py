#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from ax.service.utils.report_utils import exp_to_df, get_best_trial
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class ReportUtilsTest(TestCase):
    def test_exp_to_df(self):
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()
        df = exp_to_df(exp)
        self.assertIsInstance(df, pd.DataFrame)
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

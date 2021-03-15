#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from ax.service.utils.report_utils import exp_to_df
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

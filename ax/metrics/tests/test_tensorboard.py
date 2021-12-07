#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import numpy as np
import pandas as pd
from ax.metrics.tensorboard import TensorboardCurveMetric
from ax.utils.common.testutils import TestCase


class TensorboardCurveMetricTest(TestCase):
    def test_GetCurvesFromIds(self):
        def mock_get_tb_from_posix(path):
            if path == "None":
                return None
            return pd.Series([int(path)] * 2)

        mock_path = "ax.metrics.tensorboard.get_tb_from_posix"
        with mock.patch(mock_path, side_effect=mock_get_tb_from_posix) as mgtbfp:
            out = TensorboardCurveMetric.get_curves_from_ids(["1", "None", "2"])
            mgtbfp.assert_has_calls([mock.call("1"), mock.call("None"), mock.call("2")])
        self.assertEqual(len(out), 2)
        self.assertTrue(np.array_equal(out["1"].values, np.array([1, 1])))
        self.assertTrue(np.array_equal(out["2"].values, np.array([2, 2])))

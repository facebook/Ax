#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from unittest import mock

import numpy as np
import pandas as pd
from ax.metrics.tensorboard import TensorboardCurveMetric, get_tb_from_posix
from ax.utils.common.testutils import TestCase
from torch.utils.tensorboard import SummaryWriter


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

    def test_GetTbFromPosix(self):
        time = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = SummaryWriter(log_dir=temp_dir)
            # first training curve with 3 points
            for i in range(3):
                writer.add_scalar(
                    tag="train/loss", scalar_value=i * 10, global_step=i, walltime=time
                )
                time += 1
            result = get_tb_from_posix(path=temp_dir)
            self.assertTrue(
                np.allclose(result["train/loss"].index.values, np.array([0, 1, 2]))
            )
            self.assertTrue(
                np.allclose(result["train/loss"].values, np.array([0, 10, 20]))
            )
            # second training curve (like a restarted run) with later walltimes
            for i in range(5):
                writer.add_scalar(
                    tag="train/loss", scalar_value=i * 5, global_step=i, walltime=time
                )
                time += 1
            result = get_tb_from_posix(path=temp_dir)
            # expected behavior is to return the second training curve and ignore first
            self.assertTrue(
                np.allclose(
                    result["train/loss"].index.values, np.array([0, 1, 2, 3, 4])
                )
            )
            self.assertTrue(
                np.allclose(result["train/loss"].values, np.array([0, 5, 10, 15, 20]))
            )

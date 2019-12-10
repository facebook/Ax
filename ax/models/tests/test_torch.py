#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.models.torch_base import TorchModel
from ax.utils.common.testutils import TestCase


class TorchModelTest(TestCase):
    def setUp(self):
        pass

    def testTorchModelFit(self):
        torch_model = TorchModel()
        torch_model.fit(
            Xs=[np.array(0)],
            Ys=[np.array(0)],
            Yvars=[np.array(1)],
            bounds=[(0, 1)],
            task_features=[],
            feature_names=["x1"],
            metric_names=["y"],
            fidelity_features=[],
        )

    def testTorchModelPredict(self):
        torch_model = TorchModel()
        with self.assertRaises(NotImplementedError):
            torch_model.predict(np.array([0]))

    def testTorchModelGen(self):
        torch_model = TorchModel()
        with self.assertRaises(NotImplementedError):
            torch_model.gen(n=1, bounds=[(0, 1)], objective_weights=np.array([1]))

    def testNumpyTorchBestPoint(self):
        torch_model = TorchModel()
        x = torch_model.best_point(bounds=[(0, 1)], objective_weights=np.array([1]))
        self.assertIsNone(x)

    def testTorchModelCrossValidate(self):
        torch_model = TorchModel()
        with self.assertRaises(NotImplementedError):
            torch_model.cross_validate(
                Xs_train=[np.array([1])],
                Ys_train=[np.array([1])],
                Yvars_train=[np.array([1])],
                X_test=np.array([1]),
            )

    def testTorchModelUpdate(self):
        numpy_model = TorchModel()
        with self.assertRaises(NotImplementedError):
            numpy_model.update(Xs=[np.array(0)], Ys=[np.array(0)], Yvars=[np.array(1)])

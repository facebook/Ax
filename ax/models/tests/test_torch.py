#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
from ax.models.torch.utils import normalize_indices
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


class TorchUtilsTest(TestCase):
    def testNormalizeIndices(self):
        indices = [0, 2]
        nlzd_indices = normalize_indices(indices, 3)
        self.assertEqual(nlzd_indices, indices)
        nlzd_indices = normalize_indices(indices, 4)
        self.assertEqual(nlzd_indices, indices)
        indices = [0, -1]
        nlzd_indices = normalize_indices(indices, 3)
        self.assertEqual(nlzd_indices, [0, 2])
        with self.assertRaises(ValueError):
            nlzd_indices = normalize_indices([3], 3)
        with self.assertRaises(ValueError):
            nlzd_indices = normalize_indices([-4], 3)

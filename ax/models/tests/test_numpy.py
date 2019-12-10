#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.models.numpy_base import NumpyModel
from ax.utils.common.testutils import TestCase


class NumpyModelTest(TestCase):
    def setUp(self):
        pass

    def testNumpyModelFit(self):
        numpy_model = NumpyModel()
        numpy_model.fit(
            Xs=[np.array(0)],
            Ys=[np.array(0)],
            Yvars=[np.array(1)],
            bounds=[(0, 1)],
            task_features=[],
            feature_names=["x"],
            metric_names=["y"],
            fidelity_features=[],
        )

    def testNumpyModelFeatureImportances(self):
        numpy_model = NumpyModel()
        with self.assertRaises(NotImplementedError):
            numpy_model.feature_importances()

    def testNumpyModelPredict(self):
        numpy_model = NumpyModel()
        with self.assertRaises(NotImplementedError):
            numpy_model.predict(np.array([0]))

    def testNumpyModelGen(self):
        numpy_model = NumpyModel()
        with self.assertRaises(NotImplementedError):
            numpy_model.gen(n=1, bounds=[(0, 1)], objective_weights=np.array([1]))

    def testNumpyModelBestPoint(self):
        numpy_model = NumpyModel()
        x = numpy_model.best_point(bounds=[(0, 1)], objective_weights=np.array([1]))
        self.assertIsNone(x)

    def testNumpyModelCrossValidate(self):
        numpy_model = NumpyModel()
        with self.assertRaises(NotImplementedError):
            numpy_model.cross_validate(
                Xs_train=[np.array([1])],
                Ys_train=[np.array([1])],
                Yvars_train=[np.array([1])],
                X_test=np.array([1]),
            )

    def testNumpyModelUpdate(self):
        numpy_model = NumpyModel()
        with self.assertRaises(NotImplementedError):
            numpy_model.update(Xs=[np.array(0)], Ys=[np.array(0)], Yvars=[np.array(1)])

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.models.numpy.randomforest import RandomForest
from ax.utils.common.testutils import TestCase


class RandomForestTest(TestCase):
    def testRFModel(self):
        Xs = [np.random.rand(10, 2) for i in range(2)]
        Ys = [np.random.rand(10, 1) for i in range(2)]
        Yvars = [np.random.rand(10, 1) for i in range(2)]

        m = RandomForest(num_trees=5)
        m.fit(
            Xs=Xs,
            Ys=Ys,
            Yvars=Yvars,
            bounds=[(0, 1)] * 2,
            task_features=[],
            feature_names=["x1", "x2"],
            metric_names=["y"],
            fidelity_features=[],
        )
        self.assertEqual(len(m.models), 2)
        self.assertEqual(len(m.models[0].estimators_), 5)

        f, cov = m.predict(np.random.rand(5, 2))
        self.assertEqual(f.shape, (5, 2))
        self.assertEqual(cov.shape, (5, 2, 2))

        f, cov = m.cross_validate(
            Xs_train=Xs, Ys_train=Ys, Yvars_train=Yvars, X_test=np.random.rand(3, 2)
        )
        self.assertEqual(f.shape, (3, 2))
        self.assertEqual(cov.shape, (3, 2, 2))

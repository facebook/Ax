#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.models.random.rembo_initializer import REMBOInitializer
from ax.utils.common.testutils import TestCase


class REMBOInitializerTest(TestCase):
    def testREMBOInitializerModel(self):
        A = np.vstack((np.eye(2, 2), -(np.eye(2, 2))))
        # Test setting attributes
        m = REMBOInitializer(A=A, bounds_d=[(-2, 2)] * 2)
        self.assertTrue(np.allclose(A, m.A))
        self.assertEqual(m.bounds_d, [(-2, 2), (-2, 2)])

        # Test project up
        Z = m.project_up(5 * np.random.rand(3, 2))
        self.assertEqual(Z.shape, (3, 4))
        self.assertTrue(Z.min() >= -1.0)
        self.assertTrue(Z.max() <= 1.0)

        # Test gen
        Z, w = m.gen(3, bounds=[(-1.0, 1.0)] * 4)
        self.assertEqual(Z.shape, (3, 4))
        self.assertTrue(Z.min() >= -1.0)
        self.assertTrue(Z.max() <= 1.0)

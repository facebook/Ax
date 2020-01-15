#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.models.random.alebo_initializer import ALEBOInitializer
from ax.utils.common.testutils import TestCase


class ALEBOSobolTest(TestCase):
    def testALEBOSobolModel(self):
        B = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        Q = np.linalg.pinv(B) @ B
        # Test setting attributes
        m = ALEBOInitializer(B=B)
        self.assertTrue(np.allclose(Q, m.Q))

        # Test gen
        Z, w = m.gen(5, bounds=[(-1.0, 1.0)] * 3)
        self.assertEqual(Z.shape, (5, 3))
        self.assertTrue(Z.min() >= -1.0)
        self.assertTrue(Z.max() <= 1.0)
        # Verify that it is in the subspace
        self.assertTrue(np.allclose(Q @ Z.transpose(), Z.transpose()))

        m = ALEBOInitializer(B=B, nsamp=1)
        with self.assertRaises(ValueError):
            m.gen(2, bounds=[(-1.0, 1.0)] * 3)

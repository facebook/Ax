#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.models.random.base import RandomModel
from ax.utils.common.testutils import TestCase


class RandomModelTest(TestCase):
    def setUp(self):
        self.random_model = RandomModel()

    def testRandomModelGenSamples(self):
        with self.assertRaises(NotImplementedError):
            self.random_model._gen_samples(n=1, tunable_d=1)

    def testRandomModelGenUnconstrained(self):
        with self.assertRaises(NotImplementedError):
            self.random_model._gen_unconstrained(
                n=1, d=2, tunable_feature_indices=np.array([])
            )

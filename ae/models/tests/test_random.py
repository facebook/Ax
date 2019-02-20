#!/usr/bin/env python3

import numpy as np
from ae.lazarus.ae.models.random.base import RandomModel
from ae.lazarus.ae.utils.common.testutils import TestCase


class RandomModelTest(TestCase):
    def setUp(self):
        pass

    def testNumpyModelFit(self):
        random_model = RandomModel()
        random_model.fit(
            Xs=[np.array(0)],
            Ys=[np.array(0)],
            Yvars=[np.array(1)],
            bounds=[(0, 1)],
            task_features=[],
            feature_names=["x"],
        )

    def testRandomModelGen(self):
        random_model = RandomModel()
        with self.assertRaises(NotImplementedError):
            random_model.gen(n=1, bounds=[(0, 1)], objective_weights=np.array([1]))

    def testRandomModelGenUnconstrained(self):
        random_model = RandomModel()
        with self.assertRaises(NotImplementedError):
            random_model._gen_unconstrained(
                n=1, d=2, tunable_feature_indices=np.array([])
            )

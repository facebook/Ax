#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.types import merge_model_predict
from ax.utils.common.testutils import TestCase


class TypesTest(TestCase):
    def setUp(self):
        self.num_arms = 2
        mu = {"m1": [0.0, 0.5], "m2": [0.1, 0.6]}
        cov = {
            "m1": {"m1": [0.0, 0.0], "m2": [0.0, 0.0]},
            "m2": {"m1": [0.0, 0.0], "m2": [0.0, 0.0]},
        }
        self.predict = (mu, cov)

    def testMergeModelPredict(self):
        mu_append = {"m1": [0.6], "m2": [0.7]}
        cov_append = {
            "m1": {"m1": [0.0], "m2": [0.0]},
            "m2": {"m1": [0.0], "m2": [0.0]},
        }
        merged_predicts = merge_model_predict(self.predict, (mu_append, cov_append))
        self.assertEqual(len(merged_predicts[0]["m1"]), 3)

    def testMergeModelPredictFail(self):
        mu_append = {"m1": [0.6]}
        cov_append = {
            "m1": {"m1": [0.0], "m2": [0.0]},
            "m2": {"m1": [0.0], "m2": [0.0]},
        }
        with self.assertRaises(ValueError):
            merge_model_predict(self.predict, (mu_append, cov_append))

        mu_append = {"m1": [0.6], "m2": [0.7]}
        cov_append = {"m1": {"m1": [0.0], "m2": [0.0]}}
        with self.assertRaises(ValueError):
            merge_model_predict(self.predict, (mu_append, cov_append))

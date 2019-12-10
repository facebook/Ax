#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock

from ax.modelbridge.transforms.base import Transform
from ax.utils.common.testutils import TestCase


class TransformsTest(TestCase):
    def testIdentityTransform(self):
        # Test that the identity transform does not mutate anything
        t = Transform(MagicMock(), MagicMock(), MagicMock())
        x = MagicMock()
        ys = []
        ys.append(t.transform_search_space(x))
        ys.append(t.transform_optimization_config(x, x, x))
        ys.append(t.transform_observation_features(x))
        ys.append(t.transform_observation_data(x, x))
        ys.append(t.untransform_observation_features(x))
        ys.append(t.untransform_observation_data(x, x))
        self.assertEqual(len(x.mock_calls), 0)
        for y in ys:
            self.assertEqual(y, x)

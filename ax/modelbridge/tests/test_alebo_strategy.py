#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import numpy as np
import pandas as pd
import torch
from ax.core.data import Data
from ax.modelbridge.strategies.alebo import (
    ALEBOStrategy,
    get_ALEBO,
    get_ALEBOInitializer,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class ALEBOStrategyTest(TestCase):
    def test_factory_functions(self):
        experiment = get_branin_experiment(with_batch=True)
        B = np.array([[1.0, 2.0]])
        m1 = get_ALEBOInitializer(search_space=experiment.search_space, B=B)
        self.assertTrue(np.allclose(m1.model.Q, np.linalg.pinv(B) @ B))
        data = Data(
            pd.DataFrame(
                {
                    "arm_name": ["0_0", "0_1", "0_2"],
                    "metric_name": "y",
                    "mean": [-1.0, 0.0, 1.0],
                    "sem": 0.1,
                }
            )
        )
        with mock.patch("ax.modelbridge.strategies.alebo.ALEBO.fit", autospec=True):
            m2 = get_ALEBO(
                experiment=experiment, search_space=None, data=data, B=torch.tensor(B)
            )

        self.assertTrue(np.array_equal(m2.model.B.numpy(), B))

    def test_ALEBOStrategy(self):
        D = 20
        d = 3
        init_size = 5
        s = ALEBOStrategy(D=D, d=d, init_size=init_size)
        self.assertEqual(s._steps[0].num_trials, init_size)
        random_B = s._steps[0].model_kwargs["B"]
        gp_B = s._steps[1].model_kwargs["B"]
        # Check that random and GP have the same projection
        self.assertTrue(np.allclose(random_B, gp_B.numpy()))
        # And that the projection has correct properties
        self.assertEqual(random_B.shape, (d, D))
        self.assertTrue(
            torch.allclose(
                torch.sqrt((gp_B ** 2).sum(dim=0)), torch.ones(D, dtype=torch.double)
            )
        )

        s2 = s.clone_reset()
        # Check that attributes copied, but not B
        self.assertEqual(s2.d, d)
        self.assertEqual(s2.D, D)
        self.assertEqual(s2._steps[0].num_trials, init_size)
        random_B2 = s2._steps[0].model_kwargs["B"]
        self.assertEqual(random_B2.shape, (d, D))
        self.assertFalse(np.allclose(random_B, random_B2))

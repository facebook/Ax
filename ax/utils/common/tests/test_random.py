#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random

import numpy as np
import torch
from ax.utils.common.random import set_rng_seed, with_rng_seed
from ax.utils.common.testutils import TestCase


class TestRandom(TestCase):
    def test_set_rng_seed(self) -> None:
        # Set the seeds manually & using the helper, and compares the random numbers.
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        native_rand = random.random()
        np_rand = np.random.rand(5)
        torch_rand = torch.rand(5)

        set_rng_seed(seed)
        self.assertEqual(random.random(), native_rand)
        self.assertTrue(np.allclose(np_rand, np.random.rand(5)))
        self.assertTrue(torch.allclose(torch_rand, torch.rand(5)))

    def test_with_rng_seed(self, with_none_seed: bool = False) -> None:
        # Test that the context manager sets the seed and restores the state.
        native_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        with with_rng_seed(None if with_none_seed else 0):
            native_rand = random.random()
            numpy_rand = np.random.rand(5)
            torch_rand = torch.rand(5)
        self.assertEqual(random.getstate(), native_state)
        for first, second in zip(np_state, np.random.get_state()):
            if isinstance(first, np.ndarray):
                self.assertTrue(np.equal(first, second).all())
            else:
                self.assertEqual(first, second)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

        set_rng_seed(0)
        if with_none_seed:
            # The seed is None, so the random numbers should be different.
            self.assertNotEqual(random.random(), native_rand)
            self.assertFalse(np.allclose(numpy_rand, np.random.rand(5)))
            self.assertFalse(torch.allclose(torch_rand, torch.rand(5)))
        else:
            # The seed was set, so the random numbers should be the same.
            self.assertEqual(random.random(), native_rand)
            self.assertTrue(np.allclose(numpy_rand, np.random.rand(5)))
            self.assertTrue(torch.allclose(torch_rand, torch.rand(5)))

    def test_with_rng_seed_with_none_seed(self) -> None:
        self.test_with_rng_seed(with_none_seed=True)

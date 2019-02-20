#!/usr/bin/env python3

import numpy as np
from ae.lazarus.ae.generator.transforms.rounding import (
    randomized_onehot_round,
    strict_onehot_round,
)
from ae.lazarus.ae.utils.common.testutils import TestCase


class RoundingTest(TestCase):
    def setUp(self):
        pass

    def testOneHotRound(self):
        self.assertTrue(
            np.allclose(
                strict_onehot_round(np.array([0.1, 0.5, 0.3])), np.array([0, 1, 0])
            )
        )
        # One item should be set to one at random.
        self.assertEqual(
            np.count_nonzero(
                np.isclose(
                    randomized_onehot_round(np.array([0.0, 0.0, 0.0])),
                    np.array([1, 1, 1]),
                )
            ),
            1,
        )

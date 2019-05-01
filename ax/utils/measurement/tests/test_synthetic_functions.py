#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from ax.utils.common.testutils import TestCase
from ax.utils.measurement.synthetic_functions import branin, hartmann6


class TestSyntheticFunctions(TestCase):
    def test_branin(self):
        self.assertEqual(branin.name, "Branin")
        self.assertEqual(branin(1, 2), 21.62763539206238)
        self.assertEqual(branin(x1=1, x2=2), 21.62763539206238)
        self.assertEqual(branin(np.array([1, 2])), 21.62763539206238)
        self.assertEqual(branin(np.array([[1, 2], [1, 2]]))[0], 21.62763539206238)
        self.assertEqual(branin.minimums[0][0], -np.pi)
        self.assertEqual(branin.fmin, 0.397887)
        self.assertEqual(branin.fmax, 294.0)
        self.assertEqual(branin.domain[0], (-5, 10))
        self.assertEqual(branin.required_dimensionality, 2)
        with self.assertRaisesRegex(NotImplementedError, "Branin does not specify"):
            branin.maximums
        with self.assertRaisesRegex(ValueError, "Synthetic function call"):
            branin(np.array([[[1, 3]]]))

    def test_hartmann6(self):
        self.assertEqual(hartmann6.name, "Hartmann6")
        self.assertAlmostEqual(hartmann6(1, 2, 3, 4, 5, 6), 0.0)
        self.assertAlmostEqual(hartmann6(x1=1, x2=2, x3=3, x4=4, x5=5, x6=6), 0.0)
        self.assertAlmostEqual(hartmann6(np.array([1, 2, 3, 4, 5, 6])), 0.0)
        self.assertAlmostEqual(
            hartmann6(np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]))[0], 0.0
        )
        self.assertEqual(hartmann6.minimums[0][0], 0.20169)
        self.assertEqual(hartmann6.fmin, -3.32237)
        self.assertEqual(hartmann6.fmax, 0.0)
        self.assertEqual(hartmann6.domain[0], (0, 1))
        self.assertEqual(hartmann6.required_dimensionality, 6)
        with self.assertRaisesRegex(NotImplementedError, "Hartmann6 does not specify"):
            hartmann6.maximums

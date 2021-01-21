#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.utils.common.testutils import TestCase
from ax.utils.measurement.synthetic_functions import (
    FromBotorch,
    aug_branin,
    aug_hartmann6,
    branin,
    hartmann6,
)
from botorch.test_functions import synthetic as botorch_synthetic


class TestSyntheticFunctions(TestCase):
    def test_branin(self):
        self.assertEqual(branin.name, "Branin")
        self.assertAlmostEqual(branin(1, 2), 21.62763539206238)
        self.assertAlmostEqual(branin(x1=1, x2=2), 21.62763539206238)
        self.assertAlmostEqual(branin(np.array([1, 2])), 21.62763539206238)
        self.assertAlmostEqual(branin(np.array([[1, 2], [1, 2]]))[0], 21.62763539206238)
        self.assertAlmostEqual(branin.minimums[0][0], -np.pi)
        self.assertAlmostEqual(branin.fmin, 0.397887, places=6)
        self.assertAlmostEqual(branin.fmax, 308.129, places=3)
        self.assertAlmostEqual(branin.fmax, branin(-5, 0), places=3)
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
        self.assertAlmostEqual(hartmann6.minimums[0][0], 0.20169, places=5)
        self.assertAlmostEqual(hartmann6.fmin, -3.32237, places=5)
        self.assertEqual(hartmann6.fmax, 0.0)
        self.assertEqual(hartmann6.domain[0], (0, 1))
        self.assertEqual(hartmann6.required_dimensionality, 6)
        with self.assertRaisesRegex(NotImplementedError, "Hartmann6 does not specify"):
            hartmann6.maximums

    def test_aug_hartmann6(self):
        self.assertEqual(aug_hartmann6.name, "Aug_Hartmann6")
        self.assertAlmostEqual(aug_hartmann6(1, 2, 3, 4, 5, 6, 1), 0.0)
        self.assertAlmostEqual(
            aug_hartmann6(x1=1, x2=2, x3=3, x4=4, x5=5, x6=6, x7=1), 0.0
        )
        self.assertAlmostEqual(aug_hartmann6(np.array([1, 2, 3, 4, 5, 6, 1])), 0.0)
        self.assertAlmostEqual(
            aug_hartmann6(np.array([[1, 2, 3, 4, 5, 6, 1], [1, 2, 3, 4, 5, 6, 1]]))[0],
            0.0,
        )
        self.assertAlmostEqual(aug_hartmann6.minimums[0][0], 0.20169, places=5)
        self.assertAlmostEqual(aug_hartmann6.fmin, -3.32237, places=5)
        self.assertEqual(aug_hartmann6.fmax, 0.0)
        self.assertEqual(aug_hartmann6.domain[0], (0, 1))
        self.assertEqual(aug_hartmann6.required_dimensionality, 7)
        with self.assertRaisesRegex(
            NotImplementedError, "Aug_Hartmann6 does not specify"
        ):
            aug_hartmann6.maximums

    def test_aug_branin(self):
        self.assertEqual(aug_branin.name, "Aug_Branin")
        self.assertAlmostEqual(aug_branin(1, 2, 1), 21.62763539206238)
        self.assertAlmostEqual(aug_branin(x1=1, x2=2, x3=1), 21.62763539206238)
        self.assertAlmostEqual(aug_branin(np.array([1, 2, 1])), 21.62763539206238)
        self.assertAlmostEqual(
            aug_branin(np.array([[1, 2, 1], [1, 2, 1]]))[0], 21.62763539206238
        )
        self.assertAlmostEqual(aug_branin.minimums[0][0], -np.pi)
        self.assertAlmostEqual(aug_branin.fmin, branin.fmin)
        self.assertAlmostEqual(aug_branin.fmax, branin.fmax)
        self.assertEqual(aug_branin.domain[2], (0, 1))
        self.assertEqual(aug_branin.domain[0], (-5, 10))
        self.assertEqual(aug_branin.required_dimensionality, 3)
        with self.assertRaisesRegex(NotImplementedError, "Aug_Branin does not specify"):
            aug_branin.maximums
        with self.assertRaisesRegex(ValueError, "Synthetic function call"):
            aug_branin(np.array([[[1, 3, 1]]]))

    def test_botorch_ackley(self):
        ackley = FromBotorch(botorch_synthetic_function=botorch_synthetic.Ackley())
        self.assertEqual(ackley.name, "FromBotorch_Ackley")
        self.assertAlmostEqual(ackley(1.0, 2.0), 5.422131717799505)
        self.assertAlmostEqual(ackley(x1=1.0, x2=2.0), 5.422131717799505)
        self.assertAlmostEqual(ackley(np.array([1, 2])), 5.422131717799505)
        self.assertAlmostEqual(ackley(np.array([[1, 2], [1, 2]]))[0], 5.422131717799505)
        self.assertAlmostEqual(ackley.domain[0], (-32.768, 32.768), places=3)
        self.assertEqual(ackley.required_dimensionality, 2)
        self.assertEqual(ackley.fmin, 0.0)
        with self.assertRaisesRegex(NotImplementedError, "Ackley does not specify"):
            ackley.maximums
        with self.assertRaisesRegex(NotImplementedError, "Ackley does not specify"):
            ackley.minimums
        with self.assertRaisesRegex(NotImplementedError, "Ackley does not specify"):
            ackley.fmax

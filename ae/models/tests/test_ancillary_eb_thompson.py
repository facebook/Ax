#!/usr/bin/env python3

import numpy as np
from ae.lazarus.ae.models.discrete.ancillary_eb_thompson import (
    AncillaryEBThompsonSampler,
)
from ae.lazarus.ae.utils.common.testutils import TestCase


class AncillaryEBThompsonSamplerTest(TestCase):
    def setUp(self):
        self.Xs = [
            [[1, 1], [2, 2], [3, 3], [4, 4]],
            [[1, 1], [2, 2], [3, 3], [4, 4]],
        ]  # 2 metrics, 4 arms, each of dimensionality 2
        self.Ys = [[0.1, 0.2, 0.3, 0.4], [0, 0, 0, 0.1]]
        self.Yvars = [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]
        self.parameter_values = [[1, 2, 3, 4], [1, 2, 3, 4]]

    def testAncillaryEBThompsonSamplerFit(self):
        generator = AncillaryEBThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
        )
        self.assertEqual(generator.X, self.Xs[0])
        self.assertTrue(
            np.allclose(
                np.array(generator.Ys),
                np.array([[0.06, 0.123, 0.184, 0.227], [0.025, 0.025, 0.025, 0.025]]),
                atol=1e-2,
            )
        )
        self.assertTrue(
            np.allclose(
                np.array(generator.Yvars),
                np.array([[0.267, 0.226, 0.280, 0.330], [0.026, 0.026, 0.026, 0.036]]),
                atol=1e-2,
            )
        )

    def testAncillaryEBThompsonSamplerGen(self):
        generator = AncillaryEBThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
        )
        arms, weights = generator.gen(
            n=5,
            parameter_values=self.parameter_values,
            objective_weights=np.array([1, 0]),
        )
        self.assertEqual(arms, [[4, 4], [3, 3], [2, 2], [1, 1]])
        for weight, expected_weight in zip(weights, [0.31, 0.27, 0.22, 0.20]):
            self.assertAlmostEqual(weight, expected_weight, 1)

    def testAncillaryEBThompsonSamplerError(self):
        generator = AncillaryEBThompsonSampler(min_weight=0.0)
        with self.assertRaises(ValueError):
            generator.fit(
                Xs=[x[:-1] for x in self.Xs],
                Ys=[y[:-1] for y in self.Ys],
                Yvars=[y[:-1] for y in self.Yvars],
                parameter_values=self.parameter_values,
            )

    def testAncillaryEBThompsonSamplerValidation(self):

        with self.assertRaises(ValueError):
            generator = AncillaryEBThompsonSampler(
                min_weight=0.01, primary_outcome=0, secondary_outcome=0
            )
            generator.fit(
                Xs=self.Xs,
                Ys=self.Ys,
                Yvars=self.Yvars,
                parameter_values=self.parameter_values,
            )
        generator = AncillaryEBThompsonSampler(min_weight=0.01)
        with self.assertRaises(ValueError):
            generator.fit(
                Xs=self.Xs,
                Ys=[[1.1, 0.2, 0.3, 0.4], [0, 0, 0, 0.1]],
                Yvars=self.Yvars,
                parameter_values=self.parameter_values,
            )
        with self.assertRaises(ValueError):
            generator.fit(
                Xs=self.Xs,
                Ys=self.Ys,
                Yvars=[[0.1, 0.1, 0.3, 0.1], [0, 0, 0, 0.1]],
                parameter_values=self.parameter_values,
            )

    def testAncillaryEBThompsonSamplerPredict(self):
        generator = AncillaryEBThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
        )
        f, cov = generator.predict([[1, 1], [3, 3]])
        self.assertTrue(
            np.allclose(f, np.array([[0.06, 0.025], [0.18, 0.025]]), atol=1e-2)
        )

        # first element of cov is the cov matrix for  the first prediction
        # the element at 0,0 is for the first outcome
        # the element at 1,1 is for the second outcome
        self.assertTrue(
            np.allclose(
                cov,
                np.array([[[0.267, 0.0], [0.0, 0.026]], [[0.279, 0.0], [0.0, 0.0262]]]),
                atol=1e-2,
            )
        )

        with self.assertRaises(ValueError):
            generator.predict([[1, 2]])

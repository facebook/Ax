#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

import numpy as np
from ax.exceptions.core import UnsupportedError
from ax.generators.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.utils.common.random import set_rng_seed
from ax.utils.common.testutils import TestCase


class EmpiricalBayesThompsonSamplerTest(TestCase):
    def setUp(self) -> None:
        self.Xs = [
            [[1, 1], [2, 2], [3, 3], [4, 4]],
            [[1, 1], [2, 2], [3, 3], [4, 4]],
        ]  # 2 metrics, 4 arms, each of dimensionality 2
        self.Ys = [[1, 2, 3, 4], [0, 0, 0, 1]]
        self.Yvars = [[1, 1, 1, 1], [1, 1, 1, 1]]
        self.parameter_values = [[1, 2, 3, 4], [1, 2, 3, 4]]
        self.outcome_names = ["x", "y"]  # not used for regular EB

    def test_EmpiricalBayesThompsonSamplerFit(self) -> None:
        generator = EmpiricalBayesThompsonSampler(min_weight=0.0)
        self.assertFalse(generator.can_predict)
        self.assertTrue(generator.can_model_in_sample)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        self.assertEqual(generator.X, self.Xs[0])
        self.assertTrue(
            np.allclose(
                np.array(generator.Ys),
                np.array([[1.3, 2.1, 2.9, 3.7], [0.25, 0.25, 0.25, 0.25]]),
            )
        )
        self.assertTrue(
            np.allclose(
                np.array(generator.Yvars),
                np.array([[1.03, 0.87, 0.87, 1.03], [0.375, 0.375, 0.375, 1.375]]),
            )
        )

    def test_EmpiricalBayesThompsonSamplerGen(self) -> None:
        generator = EmpiricalBayesThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        with patch(
            "ax.generators.discrete.eb_thompson.EmpiricalBayesThompsonSampler",
            return_value=(
                [[4, 4], [3, 3], [2, 2], [1, 1]],
                [
                    2.67,
                    0,
                    0.25,
                    0.07,
                ],
            ),
        ):
            arms, weights, _ = generator.gen(
                n=5,
                parameter_values=self.parameter_values,
                objective_weights=np.array([1, 0]),
            )
        self.assertEqual(arms, [[4, 4], [3, 3], [2, 2], [1, 1]])
        for weight, expected_weight in zip(
            weights, [5 * i for i in [0.66, 0.25, 0.07, 0.02]]
        ):
            self.assertAlmostEqual(weight, expected_weight, delta=0.1)

    def test_EmpiricalBayesThompsonSamplerWarning(self) -> None:
        set_rng_seed(0)
        generator = EmpiricalBayesThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=[x[:-1] for x in self.Xs],
            Ys=[y[:-1] for y in self.Ys],
            Yvars=[y[:-1] for y in self.Yvars],
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        arms, weights, _ = generator.gen(
            n=5,
            parameter_values=self.parameter_values,
            objective_weights=np.array([1, 0]),
        )
        self.assertEqual(arms, [[3, 3], [2, 2], [1, 1]])
        for weight, expected_weight in zip(
            weights, [5 * i for i in [0.74, 0.21, 0.05]]
        ):
            self.assertAlmostEqual(weight, expected_weight, delta=0.1)

    def test_EmpiricalBayesThompsonSamplerValidation(self) -> None:
        generator = EmpiricalBayesThompsonSampler(min_weight=0.01)
        with self.assertRaises(ValueError):
            generator.fit(
                Xs=[[[1, 1], [2, 2], [3, 3], [4, 4]], [[1, 1], [2, 2], [4, 4]]],
                Ys=self.Ys,
                Yvars=self.Yvars,
                parameter_values=self.parameter_values,
                outcome_names=self.outcome_names,
            )

    def test_EmpiricalBayesThompsonSamplerPredict(self) -> None:
        generator = EmpiricalBayesThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        f, cov = generator.predict([[1, 1], [3, 3]])
        self.assertTrue(np.allclose(f, np.array([[1.3, 0.25], [2.9, 0.25]])))

        # first element of cov is the cov matrix for  the first prediction
        # the element at 0,0 is for the first outcome
        # the element at 1,1 is for the second outcome
        self.assertTrue(
            np.allclose(
                cov,
                np.array([[[1.03, 0.0], [0.0, 0.375]], [[0.87, 0.0], [0.0, 0.375]]]),
            )
        )

        with self.assertRaisesRegex(UnsupportedError, "out-of-sample"):
            generator.predict([[1, 2]])

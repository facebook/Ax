#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from ax.exceptions.model import ModelError
from ax.models.discrete.thompson import ThompsonSampler
from ax.utils.common.testutils import TestCase


class ThompsonSamplerTest(TestCase):
    def setUp(self):
        self.Xs = [[[1, 1], [2, 2], [3, 3], [4, 4]]]  # 4 arms, each of dimensionality 2
        self.Ys = [[1, 2, 3, 4]]
        self.Yvars = [[1, 1, 1, 1]]
        self.parameter_values = [[1, 2, 3, 4], [1, 2, 3, 4]]
        self.outcome_names = ["x", "y"]  # not used for regular TS

        self.multiple_metrics_Xs = [
            [[1, 1], [2, 2], [3, 3], [4, 4]],
            [[1, 1], [2, 2], [3, 3], [4, 4]],
        ]  # 2 metrics, 4 arms, each of dimensionality 2
        self.multiple_metrics_Ys = [[1, 2, 3, 4], [0, 0, 0, 1]]
        self.multiple_metrics_Yvars = [[1, 1, 1, 1], [1, 1, 1, 1]]

    def testThompsonSampler(self):
        generator = ThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        arms, weights, _ = generator.gen(
            n=3, parameter_values=self.parameter_values, objective_weights=np.ones(1)
        )
        self.assertEqual(arms, [[4, 4], [3, 3], [2, 2]])
        for weight, expected_weight in zip(weights, [0.725, 0.225, 0.05]):
            self.assertAlmostEqual(weight, expected_weight, 1)

    def testThompsonSamplerValidation(self):
        generator = ThompsonSampler(min_weight=0.01)

        # all Xs are not the same
        with self.assertRaises(ValueError):
            generator.fit(
                Xs=[[[1, 1], [2, 2], [3, 3], [4, 4]], [[1, 1], [2, 2], [4, 4]]],
                Ys=self.Ys,
                Yvars=self.Yvars,
                parameter_values=self.parameter_values,
                outcome_names=self.outcome_names,
            )

        # multiple observations per parameterization
        with self.assertRaises(ValueError):
            generator.fit(
                Xs=[[[1, 1], [2, 2], [2, 2]]],
                Ys=self.Ys,
                Yvars=self.Yvars,
                parameter_values=self.parameter_values,
                outcome_names=self.outcome_names,
            )

        # these are not the same observations, so should not error
        generator.fit(
            Xs=[[[1, 1], [2.0, 2], [2, 2]]],
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )

        # requires objective weights
        with self.assertRaises(ValueError):
            generator.gen(5, self.parameter_values, objective_weights=None)

    def testThompsonSamplerMinWeight(self):
        generator = ThompsonSampler(min_weight=0.01)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        arms, weights, _ = generator.gen(
            n=5, parameter_values=self.parameter_values, objective_weights=np.ones(1)
        )
        self.assertEqual(arms, [[4, 4], [3, 3], [2, 2]])
        for weight, expected_weight in zip(weights, [0.725, 0.225, 0.05]):
            self.assertAlmostEqual(weight, expected_weight, 1)

    def testThompsonSamplerUniformWeights(self):
        generator = ThompsonSampler(min_weight=0.0, uniform_weights=True)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        arms, weights, _ = generator.gen(
            n=3, parameter_values=self.parameter_values, objective_weights=np.ones(1)
        )
        self.assertEqual(arms, [[4, 4], [3, 3], [2, 2]])
        for weight, expected_weight in zip(weights, [0.33, 0.33, 0.33]):
            self.assertAlmostEqual(weight, expected_weight, 1)

    def testThompsonSamplerInfeasible(self):
        generator = ThompsonSampler(min_weight=0.9)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        with self.assertRaises(ModelError):
            generator.gen(
                n=3,
                parameter_values=self.parameter_values,
                objective_weights=np.ones(1),
            )

    def testThompsonSamplerOutcomeConstraints(self):
        generator = ThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=self.multiple_metrics_Xs,
            Ys=self.multiple_metrics_Ys,
            Yvars=self.multiple_metrics_Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        arms, weights, _ = generator.gen(
            n=4,
            parameter_values=self.parameter_values,
            objective_weights=np.array([1, 0]),
            outcome_constraints=(
                # pass in multiples of the same constraint
                # to ensure that shapes are correct for multiple constraints
                np.array([[0, 1], [0, 1], [0, 1]]),
                np.array([[1], [1], [1]]),
            ),
        )
        self.assertEqual(arms, [[3, 3], [4, 4], [2, 2], [1, 1]])
        for weight, expected_weight in zip(weights, [0.4, 0.4, 0.15, 0.05]):
            self.assertAlmostEqual(weight, expected_weight, 1)

    def testThompsonSamplerOutcomeConstraintsInfeasible(self):
        generator = ThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=self.multiple_metrics_Xs,
            Ys=self.multiple_metrics_Ys,
            Yvars=self.multiple_metrics_Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        with self.assertRaises(ValueError):
            generator.gen(
                n=3,
                parameter_values=self.parameter_values,
                objective_weights=np.ones(2),
                outcome_constraints=(np.array([[0, 1]]), np.array([[-10]])),
            )

    def testThompsonSamplerPredict(self):
        generator = ThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        f, cov = generator.predict([[1, 1], [3, 3]])
        self.assertTrue(np.array_equal(f, np.array([[1], [3]])))
        self.assertTrue(np.array_equal(cov, np.ones((2, 1, 1))))

        with self.assertRaises(ValueError):
            generator.predict([[1, 2]])

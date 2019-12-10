#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from ax.models.random.uniform import UniformGenerator
from ax.utils.common.testutils import TestCase


class UniformGeneratorTest(TestCase):
    def setUp(self):
        self.tunable_param_bounds = (0, 1)
        self.fixed_param_bounds = (1, 100)

    def _create_bounds(self, n_tunable, n_fixed):
        tunable_bounds = [self.tunable_param_bounds] * n_tunable
        fixed_bounds = [self.fixed_param_bounds] * n_fixed
        return tunable_bounds + fixed_bounds

    def testUniformGeneratorAllTunable(self):
        generator = UniformGenerator(seed=0)
        bounds = self._create_bounds(n_tunable=3, n_fixed=0)
        generated_points, weights = generator.gen(n=3, bounds=bounds)

        expected_points = np.array(
            [
                [0.5488135, 0.71518937, 0.60276338],
                [0.54488318, 0.4236548, 0.64589411],
                [0.43758721, 0.891773, 0.96366276],
            ]
        )
        self.assertTrue(np.shape(expected_points) == np.shape(generated_points))
        self.assertTrue(np.allclose(expected_points, generated_points))
        self.assertTrue(np.all(weights == 1.0))

    def testUniformGeneratorFixedSpace(self):
        generator = UniformGenerator(seed=0)
        bounds = self._create_bounds(n_tunable=0, n_fixed=2)
        n = 3
        generated_points, _ = generator.gen(
            n=3, bounds=bounds, fixed_features={0: 1, 1: 2}
        )
        expected_points = np.tile(np.array([[1, 2]]), (n, 1))
        self.assertTrue(np.shape(expected_points) == np.shape(generated_points))
        self.assertTrue(np.allclose(expected_points, generated_points))

    def testUniformGeneratorOnline(self):
        # Verify that the generator will return the expected arms if called
        # one at a time.
        generator = UniformGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)

        n = 3
        expected_points = np.array(
            [
                [0.5488135, 0.71518937, 0.60276338, 1],
                [0.54488318, 0.4236548, 0.64589411, 1],
                [0.43758721, 0.891773, 0.96366276, 1],
            ]
        )
        for i in range(n):
            generated_points, weights = generator.gen(
                n=1, bounds=bounds, fixed_features={fixed_param_index: 1}
            )
            self.assertEqual(weights, [1])
            self.assertTrue(np.allclose(generated_points, expected_points[i, :]))

    def testUniformGeneratorReseed(self):
        # Verify that the generator will return the expected arms if called
        # one at a time.
        generator = UniformGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)

        n = 3
        expected_points = np.array(
            [
                [0.5488135, 0.71518937, 0.60276338, 1],
                [0.54488318, 0.4236548, 0.64589411, 1],
                [0.43758721, 0.891773, 0.96366276, 1],
            ]
        )
        for i in range(n):
            generated_points, weights = generator.gen(
                n=1, bounds=bounds, fixed_features={fixed_param_index: 1}
            )
            self.assertEqual(weights, [1])
            self.assertTrue(np.allclose(generated_points, expected_points[i, :]))

    def testUniformGeneratorWithOrderConstraints(self):
        # Enforce dim_0 <= dim_1 <= dim_2 <= dim_3.
        # Enforce both fixed and tunable constraints.
        generator = UniformGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        generated_points, weights = generator.gen(
            n=3,
            bounds=bounds,
            linear_constraints=(
                np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]]),
                np.array([0, 0, 0]),
            ),
            fixed_features={fixed_param_index: 0.5},
        )

        expected_points = np.array(
            [
                [0.13818295, 0.19658236, 0.36872517, 0.5],
                [0.0486903, 0.25364252, 0.44613551, 0.5],
                [0.09088573, 0.2277595, 0.41030156, 0.5],
            ]
        )
        self.assertTrue(np.shape(expected_points) == np.shape(generated_points))
        self.assertTrue(np.allclose(expected_points, generated_points))

    def testUniformGeneratorWithLinearConstraints(self):
        # Enforce dim_0 <= dim_1 <= dim_2 <= dim_3.
        # Enforce both fixed and tunable constraints.
        generator = UniformGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        generated_points, weights = generator.gen(
            n=3,
            bounds=bounds,
            linear_constraints=(
                np.array([[1, 1, 0, 0], [0, 1, 1, 0]]),
                np.array([1, 1]),
            ),
            fixed_features={fixed_param_index: 1},
        )
        expected_points = np.array(
            [
                [0.0871293, 0.0202184, 0.83261985, 1.0],
                [0.11827443, 0.63992102, 0.14335329, 1.0],
                [0.56843395, 0.0187898, 0.6176355, 1.0],
            ]
        )
        self.assertTrue(np.shape(expected_points) == np.shape(generated_points))
        self.assertTrue(np.allclose(expected_points, generated_points))

    def testUniformGeneratorBadBounds(self):
        generator = UniformGenerator()
        with self.assertRaises(ValueError):
            generated_points, weights = generator.gen(n=1, bounds=[(-1, 1)])

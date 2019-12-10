#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from ax.models.random.sobol import SobolGenerator
from ax.utils.common.testutils import TestCase


class SobolGeneratorTest(TestCase):
    def setUp(self):
        self.tunable_param_bounds = (0, 1)
        self.fixed_param_bounds = (1, 100)

    def _create_bounds(self, n_tunable, n_fixed):
        tunable_bounds = [self.tunable_param_bounds] * n_tunable
        fixed_bounds = [self.fixed_param_bounds] * n_fixed
        return tunable_bounds + fixed_bounds

    def testSobolGeneratorAllTunable(self):
        generator = SobolGenerator(seed=0)
        bounds = self._create_bounds(n_tunable=3, n_fixed=0)
        generated_points, weights = generator.gen(n=3, bounds=bounds)

        expected_points = np.array(
            [
                [0.63552922, 0.17165081, 0.85513169],
                [0.92333341, 0.75570321, 0.72268772],
                [0.21601909, 0.48894, 0.11520141],
            ]
        )
        self.assertTrue(np.shape(expected_points) == np.shape(generated_points))
        self.assertTrue(np.allclose(expected_points, generated_points))
        self.assertTrue(np.all(weights == 1.0))
        self.assertEqual(generator._get_state(), {"init_position": 3})

    def testSobolGeneratorFixedSpace(self):
        generator = SobolGenerator(seed=0)
        bounds = self._create_bounds(n_tunable=0, n_fixed=2)
        n = 3
        generated_points, _ = generator.gen(
            n=3, bounds=bounds, fixed_features={0: 1, 1: 2}
        )
        expected_points = np.tile(np.array([[1, 2]]), (n, 1))
        self.assertTrue(np.shape(expected_points) == np.shape(generated_points))
        self.assertTrue(np.allclose(expected_points, generated_points))

    def testSobolGeneratorNoScramble(self):
        generator = SobolGenerator(scramble=False)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        generated_points, weights = generator.gen(
            n=3, bounds=bounds, fixed_features={fixed_param_index: 1}
        )
        expected_points = np.array(
            [[0.5, 0.5, 0.5, 1.0], [0.75, 0.25, 0.75, 1.0], [0.25, 0.75, 0.25, 1.0]]
        )
        self.assertTrue(np.shape(expected_points) == np.shape(generated_points))
        self.assertTrue(np.allclose(generated_points, expected_points))

    def testSobolGeneratorOnline(self):
        # Verify that the generator will return the expected arms if called
        # one at a time.
        generator = SobolGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)

        n = 3
        expected_points = np.array(
            [
                [0.63552922, 0.17165081, 0.85513169, 1],
                [0.92333341, 0.75570321, 0.72268772, 1],
                [0.21601909, 0.48894, 0.11520141, 1],
            ]
        )
        for i in range(n):
            generated_points, weights = generator.gen(
                n=1, bounds=bounds, fixed_features={fixed_param_index: 1}
            )
            self.assertEqual(weights, [1])
            self.assertTrue(np.allclose(generated_points, expected_points[i, :]))

    def testSobolGeneratorWithOrderConstraints(self):
        # Enforce dim_0 <= dim_1 <= dim_2 <= dim_3.
        # Enforce both fixed and tunable constraints.
        generator = SobolGenerator(seed=0)
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
                [0.0625397, 0.18969421, 0.38985136, 0.5],
                [0.14849217, 0.26198292, 0.47683588, 0.5],
                [0.04088604, 0.08176377, 0.49635732, 0.5],
            ]
        )
        self.assertTrue(np.shape(expected_points) == np.shape(generated_points))
        self.assertTrue(np.allclose(expected_points, generated_points))

    def testSobolGeneratorWithLinearConstraints(self):
        # Enforce dim_0 <= dim_1 <= dim_2 <= dim_3.
        # Enforce both fixed and tunable constraints.
        generator = SobolGenerator(seed=0)
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
                [0.21601909, 0.48894, 0.11520141, 1.0],
                [0.28908476, 0.00287463, 0.54073817, 1.0],
                [0.0625397, 0.18969421, 0.38985136, 1.0],
            ]
        )
        self.assertTrue(np.shape(expected_points) == np.shape(generated_points))
        self.assertTrue(np.allclose(expected_points, generated_points))

    def testSobolGeneratorOnlineRestart(self):
        # Ensure a single batch generation can also equivalently done by
        # a partial generation, re-initialization of a new SobolGenerator,
        # and a final generation.
        generator = SobolGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        generated_points_single_batch, _ = generator.gen(
            n=3, bounds=bounds, fixed_features={fixed_param_index: 1}
        )

        generator_first_batch = SobolGenerator(seed=0)
        generated_points_first_batch, _ = generator_first_batch.gen(
            n=1, bounds=bounds, fixed_features={fixed_param_index: 1}
        )
        generator_second_batch = SobolGenerator(
            init_position=generator_first_batch.init_position, seed=0
        )
        generated_points_second_batch, _ = generator_second_batch.gen(
            n=2, bounds=bounds, fixed_features={fixed_param_index: 1}
        )

        generated_points_two_trials = np.vstack(
            (generated_points_first_batch, generated_points_second_batch)
        )
        self.assertTrue(
            np.shape(generated_points_single_batch)
            == np.shape(generated_points_two_trials)
        )
        self.assertTrue(
            np.allclose(generated_points_single_batch, generated_points_two_trials)
        )

    def testSobolGeneratorBadBounds(self):
        generator = SobolGenerator()
        with self.assertRaises(ValueError):
            generated_points, weights = generator.gen(n=1, bounds=[(-1, 1)])

    def testSobolGeneratorMaxDraws(self):
        generator = SobolGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        with self.assertRaises(ValueError):
            generated_points, weights = generator.gen(
                n=3,
                bounds=bounds,
                linear_constraints=(
                    np.array([[1, 1, 0, 0], [0, 1, 1, 0]]),
                    np.array([1, 1]),
                ),
                fixed_features={fixed_param_index: 1},
                model_gen_options={"max_rs_draws": 0},
            )

    def testSobolGeneratorDedupe(self):
        generator = SobolGenerator(seed=0, deduplicate=True)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        generated_points, weights = generator.gen(
            n=2,
            bounds=bounds,
            linear_constraints=(
                np.array([[1, 1, 0, 0], [0, 1, 1, 0]]),
                np.array([1, 1]),
            ),
            fixed_features={fixed_param_index: 1},
            rounding_func=lambda x: x,
        )
        self.assertEqual(len(generated_points), 2)
        generated_points, weights = generator.gen(
            n=1,
            bounds=bounds,
            linear_constraints=(
                np.array([[1, 1, 0, 0], [0, 1, 1, 0]]),
                np.array([1, 1]),
            ),
            fixed_features={fixed_param_index: 1},
            rounding_func=lambda x: x,
        )
        self.assertEqual(len(generated_points), 1)
        self.assertIsNotNone(generator._get_state().get("generated_points"))

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from ax.exceptions.core import SearchSpaceExhausted
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
        self.assertEqual(np.shape(generated_points), (3, 3))
        np_bounds = np.array(bounds)
        self.assertTrue(np.alltrue(generated_points >= np_bounds[:, 0]))
        self.assertTrue(np.alltrue(generated_points <= np_bounds[:, 1]))
        self.assertTrue(np.all(weights == 1.0))
        self.assertEqual(generator._get_state(), {"init_position": 3})

    def testSobolGeneratorFixedSpace(self):
        generator = SobolGenerator(seed=0)
        bounds = self._create_bounds(n_tunable=0, n_fixed=2)
        generated_points, _ = generator.gen(
            n=3, bounds=bounds, fixed_features={0: 1, 1: 2}
        )
        self.assertEqual(np.shape(generated_points), (3, 2))
        np_bounds = np.array(bounds)
        self.assertTrue(np.alltrue(generated_points >= np_bounds[:, 0]))
        self.assertTrue(np.alltrue(generated_points <= np_bounds[:, 1]))

    def testSobolGeneratorNoScramble(self):
        generator = SobolGenerator(scramble=False)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        generated_points, weights = generator.gen(
            n=3, bounds=bounds, fixed_features={fixed_param_index: 1}
        )
        self.assertEqual(np.shape(generated_points), (3, 4))
        np_bounds = np.array(bounds)
        self.assertTrue(np.alltrue(generated_points >= np_bounds[:, 0]))
        self.assertTrue(np.alltrue(generated_points <= np_bounds[:, 1]))

    def testSobolGeneratorOnline(self):
        # Verify that the generator will return the expected arms if called
        # one at a time.
        bulk_generator = SobolGenerator(seed=0)
        generator = SobolGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        bulk_generated_points, bulk_weights = bulk_generator.gen(
            n=3, bounds=bounds, fixed_features={fixed_param_index: 1}
        )
        np_bounds = np.array(bounds)
        for expected_points in bulk_generated_points:
            generated_points, weights = generator.gen(
                n=1, bounds=bounds, fixed_features={fixed_param_index: 1}
            )
            self.assertEqual(weights, [1])
            self.assertTrue(np.alltrue(generated_points >= np_bounds[:, 0]))
            self.assertTrue(np.alltrue(generated_points <= np_bounds[:, 1]))
            self.assertTrue(generated_points[..., -1] == 1)
            self.assertTrue(np.array_equal(expected_points, generated_points.flatten()))

    def testSobolGeneratorWithOrderConstraints(self):
        # Enforce dim_0 <= dim_1 <= dim_2 <= dim_3.
        # Enforce both fixed and tunable constraints.
        generator = SobolGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        A = np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]])
        b = np.array([0, 0, 0])
        generated_points, weights = generator.gen(
            n=3,
            bounds=bounds,
            linear_constraints=(A, b),
            fixed_features={fixed_param_index: 0.5},
        )
        self.assertEqual(np.shape(generated_points), (3, 4))
        self.assertTrue(np.alltrue(generated_points[..., -1] == 0.5))
        self.assertTrue(
            np.array_equal(
                np.sort(generated_points[..., :-1], axis=-1),
                generated_points[..., :-1],
            )
        )

    def testSobolGeneratorWithLinearConstraints(self):
        # Enforce dim_0 <= dim_1 <= dim_2 <= dim_3.
        # Enforce both fixed and tunable constraints.
        generator = SobolGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        A = np.array([[1, 1, 0, 0], [0, 1, 1, 0]])
        b = np.array([1, 1])

        generated_points, weights = generator.gen(
            n=3,
            bounds=bounds,
            linear_constraints=(
                A,
                b,
            ),
            fixed_features={fixed_param_index: 1},
        )
        self.assertTrue(np.shape(generated_points) == (3, 4))
        self.assertTrue(np.alltrue(generated_points[..., -1] == 1))
        self.assertTrue(np.alltrue(generated_points @ A.transpose() <= b))

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
        with self.assertRaises(SearchSpaceExhausted):
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

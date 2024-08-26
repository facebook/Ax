#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
from ax.exceptions.core import SearchSpaceExhausted
from ax.models.random.uniform import UniformGenerator
from ax.utils.common.testutils import TestCase


class UniformGeneratorTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tunable_param_bounds = (0.0, 1.0)
        self.fixed_param_bounds = (1.0, 100.0)
        self.seed = 0
        self.expected_points = np.array(
            [
                [0.5488135, 0.71518937, 0.60276338],
                [0.54488318, 0.4236548, 0.64589411],
                [0.43758721, 0.891773, 0.96366276],
            ]
        )

    def _create_bounds(self, n_tunable: int, n_fixed: int) -> list[tuple[float, float]]:
        tunable_bounds = [self.tunable_param_bounds] * n_tunable
        fixed_bounds = [self.fixed_param_bounds] * n_fixed
        return tunable_bounds + fixed_bounds

    def test_with_all_tunable(self) -> None:
        generator = UniformGenerator(seed=self.seed)
        bounds = self._create_bounds(n_tunable=3, n_fixed=0)
        generated_points, weights = generator.gen(
            n=3, bounds=bounds, rounding_func=lambda x: x
        )
        self.assertTrue(np.shape(self.expected_points) == np.shape(generated_points))
        self.assertTrue(np.allclose(self.expected_points, generated_points))
        self.assertTrue(np.all(weights == 1.0))

    def test_with_fixed_space(self) -> None:
        generator = UniformGenerator(seed=self.seed)
        bounds = self._create_bounds(n_tunable=0, n_fixed=2)
        n = 3
        with self.assertRaises(SearchSpaceExhausted):
            generator.gen(
                n=3,
                bounds=bounds,
                fixed_features={0: 1, 1: 2},
                rounding_func=lambda x: x,
            )
        generator = UniformGenerator(seed=self.seed, deduplicate=False)
        generated_points, _ = generator.gen(
            n=3,
            bounds=bounds,
            fixed_features={0: 1, 1: 2},
            rounding_func=lambda x: x,
        )
        expected_points = np.tile(np.array([[1, 2]]), (n, 1))
        self.assertTrue(np.shape(expected_points) == np.shape(generated_points))
        self.assertTrue(np.allclose(expected_points, generated_points))

    def test_generating_one_by_one(self, init_position: int = 0) -> None:
        # Verify that the generator will return the expected arms if called
        # one at a time.
        generator = UniformGenerator(seed=self.seed, init_position=init_position)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)

        for i in range(init_position, 3):
            generated_points, weights = generator.gen(
                n=1,
                bounds=bounds,
                fixed_features={fixed_param_index: 1},
                rounding_func=lambda x: x,
            )
            self.assertEqual(weights, [1])
            self.assertTrue(
                np.allclose(generated_points[..., :-1], self.expected_points[i, :])
            )
            self.assertEqual(generated_points[..., -1], 1)
            self.assertEqual(generator.init_position, (i + 1) * n_tunable)

    def test_with_init_position(self) -> None:
        # These are multiples of 3 since there are 3 tunable parameters.
        self.test_generating_one_by_one(init_position=3)
        self.test_generating_one_by_one(init_position=6)

    def test_with_reloaded_state(self) -> None:
        # Check that a reloaded generator will produce the same samples.
        org_generator = UniformGenerator()
        bounds = self._create_bounds(n_tunable=3, n_fixed=0)
        # Generate some to advance the state.
        org_generator.gen(n=3, bounds=bounds, rounding_func=lambda x: x)
        # Construct a new generator with the state.
        new_generator = UniformGenerator(**org_generator._get_state())
        # Compare the generated samples.
        org_samples, _ = org_generator.gen(
            n=3, bounds=bounds, rounding_func=lambda x: x
        )
        new_samples, _ = new_generator.gen(
            n=3, bounds=bounds, rounding_func=lambda x: x
        )
        self.assertTrue(np.allclose(org_samples, new_samples))

    def test_with_order_constraints(self) -> None:
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
            rounding_func=lambda x: x,
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

    def test_with_linear_constraints(self) -> None:
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
            rounding_func=lambda x: x,
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

    def test_with_bad_bounds(self) -> None:
        generator = UniformGenerator()
        with self.assertRaises(ValueError):
            generated_points, weights = generator.gen(
                n=1, bounds=[(-1, 1)], rounding_func=lambda x: x
            )

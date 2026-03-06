#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
from ax.core.search_space import SearchSpaceDigest
from ax.generators.random.in_sample import InSampleUniformGenerator
from ax.utils.common.testutils import TestCase


class InSampleUniformGeneratorTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.generated_points = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
                [0.9, 1.0],
            ]
        )
        self.ssd = SearchSpaceDigest(
            feature_names=["x0", "x1"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )

    def test_basic_selection(self) -> None:
        generator = InSampleUniformGenerator(seed=0)
        points, weights = generator.gen(
            n=2,
            search_space_digest=self.ssd,
            generated_points=self.generated_points,
        )
        self.assertEqual(points.shape, (2, 2))
        self.assertTrue(np.all(weights == 1.0))
        # Each selected row must be present in the original set.
        for row in points:
            self.assertTrue(
                any(np.array_equal(row, gp) for gp in self.generated_points)
            )

    def test_selects_all(self) -> None:
        """Selecting all points should return all of them (in some order)."""
        generator = InSampleUniformGenerator(seed=0)
        points, weights = generator.gen(
            n=5,
            search_space_digest=self.ssd,
            generated_points=self.generated_points,
        )
        self.assertEqual(points.shape, (5, 2))
        self.assertTrue(np.all(weights == 1.0))
        # Should be a permutation of the input.
        self.assertEqual(
            {tuple(row) for row in points.tolist()},
            {tuple(row) for row in self.generated_points.tolist()},
        )

    def test_not_enough_points(self) -> None:
        generator = InSampleUniformGenerator(seed=0)
        with self.assertRaisesRegex(ValueError, "Cannot select 6 arms"):
            generator.gen(
                n=6,
                search_space_digest=self.ssd,
                generated_points=self.generated_points,
            )

    def test_no_generated_points(self) -> None:
        generator = InSampleUniformGenerator(seed=0)
        with self.assertRaisesRegex(ValueError, "Cannot select 1 arms: only 0"):
            generator.gen(
                n=1,
                search_space_digest=self.ssd,
                generated_points=None,
            )

    def test_reproducibility(self) -> None:
        """Same seed and init_position produce the same selection."""
        gen1 = InSampleUniformGenerator(seed=42)
        gen2 = InSampleUniformGenerator(seed=42)
        points1, _ = gen1.gen(
            n=2,
            search_space_digest=self.ssd,
            generated_points=self.generated_points,
        )
        points2, _ = gen2.gen(
            n=2,
            search_space_digest=self.ssd,
            generated_points=self.generated_points,
        )
        self.assertTrue(np.array_equal(points1, points2))

    def test_different_selections_across_calls(self) -> None:
        """Successive calls produce different selections (init_position advances)."""
        generator = InSampleUniformGenerator(seed=0)
        points1, _ = generator.gen(
            n=2,
            search_space_digest=self.ssd,
            generated_points=self.generated_points,
        )
        self.assertEqual(generator.init_position, 2)
        points2, _ = generator.gen(
            n=2,
            search_space_digest=self.ssd,
            generated_points=self.generated_points,
        )
        self.assertEqual(generator.init_position, 4)
        # With 5 points and n=2, different seeds should (almost surely)
        # produce different selections.
        self.assertFalse(np.array_equal(points1, points2))

    def test_gen_samples_raises(self) -> None:
        generator = InSampleUniformGenerator()
        with self.assertRaises(NotImplementedError):
            generator._gen_samples(
                n=1,
                tunable_d=2,
                bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
            )

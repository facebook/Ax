#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from itertools import product

import numpy as np
from ax.utils.common.testutils import TestCase
from ax.utils.stats.math_utils import relativize, unrelativize


class RelativizeTest(TestCase):
    def setUp(self) -> None:
        """Set up common test data."""
        self.means_t = np.array([3.0, 4.0, 5.0])
        self.sems_t = np.array([0.3, 0.4, 0.5])
        self.mean_c = 2.0
        self.sem_c = 0.2

    def test_relativize_bias_correction(self) -> None:
        """Test relativize with bias correction."""
        rel_means_bc, _ = relativize(
            means_t=self.means_t,
            sems_t=self.sems_t,
            mean_c=self.mean_c,
            sem_c=self.sem_c,
            bias_correction=True,
        )

        rel_means_no_bc, _ = relativize(
            means_t=self.means_t,
            sems_t=self.sems_t,
            mean_c=self.mean_c,
            sem_c=self.sem_c,
            bias_correction=False,
        )

        # Bias correction should produce different results
        self.assertFalse(np.allclose(rel_means_bc, rel_means_no_bc))

    def test_relativize_control_as_constant(self) -> None:
        """Test relativize with control as constant."""
        _, rel_sems = relativize(
            means_t=self.means_t,
            sems_t=self.sems_t,
            mean_c=self.mean_c,
            sem_c=self.sem_c,
            control_as_constant=True,
        )

        # With control as constant, variance calculation is simplified
        expected_rel_sems = self.sems_t / np.abs(self.mean_c)
        self.assertTrue(np.allclose(rel_sems, expected_rel_sems))

    def test_relativize_with_covariance(self) -> None:
        """Test relativize with non-zero covariance."""
        _, rel_sems_cov = relativize(
            means_t=self.means_t[:1],
            sems_t=self.sems_t[:1],
            mean_c=self.mean_c,
            sem_c=self.sem_c,
            cov_means=0.1,
            control_as_constant=False,
        )

        _, rel_sems_no_cov = relativize(
            means_t=self.means_t[:1],
            sems_t=self.sems_t[:1],
            mean_c=self.mean_c,
            sem_c=self.sem_c,
            cov_means=0.0,
            control_as_constant=False,
        )

        # Different covariance should produce different results
        self.assertFalse(np.allclose(rel_sems_cov, rel_sems_no_cov))

    def test_relativize_scalar_inputs(self) -> None:
        """Test relativize with scalar inputs."""
        rel_mean, rel_sem = relativize(
            means_t=4.0,
            sems_t=0.5,
            mean_c=self.mean_c,
            sem_c=self.sem_c,
            bias_correction=False,
        )

        # Should handle scalar inputs and return scalars or 0-dimensional arrays
        self.assertTrue(
            np.isscalar(rel_mean)
            or (isinstance(rel_mean, np.ndarray) and rel_mean.shape == ())
        )
        self.assertTrue(
            np.isscalar(rel_sem)
            or (isinstance(rel_sem, np.ndarray) and rel_sem.shape == ())
        )
        self.assertAlmostEqual(float(rel_mean), 1.0)  # (4-2)/2 = 1

    def test_relativize_zero_control_error(self) -> None:
        """Test that relativize raises error when control mean is too small."""
        with self.assertRaisesRegex(
            ValueError, "mean_control .* is smaller than 1 in 10 billion"
        ):
            relativize(
                means_t=np.array([1.0]),
                sems_t=np.array([0.1]),
                mean_c=1e-15,  # Very small control mean
                sem_c=0.1,
            )

    def test_relativize_same_values(self) -> None:
        """Test relativize when test and control values are the same."""
        means_t = np.array([2.0, 2.0])
        sems_t = np.array([0.1, 0.1])

        rel_means, _ = relativize(
            means_t=means_t,
            sems_t=sems_t,
            mean_c=2.0,
            sem_c=0.1,
            bias_correction=True,
        )

        # When values are the same, relative change should be zero
        expected_rel_means = np.array([0.0, 0.0])
        self.assertTrue(np.allclose(rel_means, expected_rel_means))

    def test_relativize_parameter_combinations(self) -> None:
        """Test relativize with various parameter combinations."""
        for bias_correction, as_percent, control_as_constant in product(
            [True, False], [True, False], [True, False]
        ):
            with self.subTest(
                bias_correction=bias_correction,
                as_percent=as_percent,
                control_as_constant=control_as_constant,
            ):
                rel_means, rel_sems = relativize(
                    means_t=self.means_t,
                    sems_t=self.sems_t,
                    mean_c=self.mean_c,
                    sem_c=self.sem_c,
                    bias_correction=bias_correction,
                    as_percent=as_percent,
                    control_as_constant=control_as_constant,
                )

                # Should return numpy arrays
                self.assertIsInstance(rel_means, np.ndarray)
                self.assertIsInstance(rel_sems, np.ndarray)

                # Should have same length as input
                self.assertEqual(len(rel_means), len(self.means_t))
                self.assertEqual(len(rel_sems), len(self.sems_t))

                # SEMs should be non-negative
                self.assertTrue(np.all(rel_sems >= 0))

    def test_relativize_negative_control_mean(self) -> None:
        """Test relativize with negative control mean."""
        means_t = np.array([1.0, 3.0])
        sems_t = np.array([0.1, 0.3])

        rel_means, _ = relativize(
            means_t=means_t,
            sems_t=sems_t,
            mean_c=-2.0,
            sem_c=0.2,
            bias_correction=False,
        )

        # Should handle negative control mean using absolute value
        # Expected: (1 - (-2)) / |-2| = 3/2 = 1.5, (3 - (-2)) / |-2| = 5/2 = 2.5
        expected_rel_means = np.array([1.5, 2.5])
        self.assertTrue(np.allclose(rel_means, expected_rel_means))


class UnrelativizeTest(TestCase):
    def test_unrelativize(self) -> None:
        means_t = np.array([-100.0, 101.0, 200.0, 300.0, 400.0])
        sems_t = np.array([2.0, 3.0, 2.0, 4.0, 0.0])
        mean_c = 200.0
        sem_c = 2.0

        for bias_correction, cov_means, as_percent, control_as_constant in product(
            (True, False, None),
            (0.5, 0.0),
            (True, False, None),
            (True, False, None),
        ):
            rel_mean_t, rel_sems_t = relativize(
                means_t,
                sems_t,
                mean_c,
                sem_c,
                cov_means=cov_means,
                # pyre-fixme[6]: For 6th argument expected `bool` but got
                #  `Optional[bool]`.
                bias_correction=bias_correction,
                # pyre-fixme[6]: For 7th argument expected `bool` but got
                #  `Optional[bool]`.
                as_percent=as_percent,
                # pyre-fixme[6]: For 8th argument expected `bool` but got
                #  `Optional[bool]`.
                control_as_constant=control_as_constant,
            )
            unrel_mean_t, unrel_sems_t = unrelativize(
                rel_mean_t,
                rel_sems_t,
                mean_c,
                sem_c,
                cov_means=cov_means,
                # pyre-fixme[6]: For 6th argument expected `bool` but got
                #  `Optional[bool]`.
                bias_correction=bias_correction,
                # pyre-fixme[6]: For 7th argument expected `bool` but got
                #  `Optional[bool]`.
                as_percent=as_percent,
                # pyre-fixme[6]: For 8th argument expected `bool` but got
                #  `Optional[bool]`.
                control_as_constant=control_as_constant,
            )
            self.assertTrue(np.allclose(means_t, unrel_mean_t))
            self.assertTrue(np.allclose(sems_t, unrel_sems_t))

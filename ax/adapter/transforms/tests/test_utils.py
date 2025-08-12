#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
from ax.adapter.transforms.log_y import match_ci_width
from ax.utils.common.testutils import TestCase
from numpy import typing as npt


class TestUtils(TestCase):
    def test_match_ci_width(self) -> None:
        # Tests the behavior of match_ci_width using a simple transform.

        def transform(y: npt.NDArray) -> npt.NDArray:
            return y * 2.0

        mean = np.array([1.0, 2.0])
        variance = np.array([0.1, 0.2])
        # Transform using variance.
        new_mean, new_variance = match_ci_width(
            mean=mean,
            sem=None,
            variance=variance,
            transform=transform,
        )
        self.assertTrue(np.allclose(new_mean, np.array([2.0, 4.0])))
        self.assertTrue(np.allclose(new_variance, np.array([0.4, 0.8])))
        # Transform using sem.
        new_mean, new_sem = match_ci_width(
            mean=mean,
            sem=np.sqrt(variance),
            variance=None,
            transform=transform,
        )
        self.assertTrue(np.allclose(new_mean, np.array([2.0, 4.0])))
        self.assertTrue(np.allclose(new_sem, np.array([np.sqrt(0.4), np.sqrt(0.8)])))
        # Transform with NaN variance.
        new_mean, new_variance = match_ci_width(
            mean=mean,
            sem=None,
            variance=np.array([np.nan, np.nan]),
            transform=transform,
        )
        self.assertTrue(np.allclose(new_mean, np.array([2.0, 4.0])))
        self.assertTrue(np.all(np.isnan(new_variance)))
        # Test with upper and lower bounds.
        new_mean, new_sem = match_ci_width(
            mean=mean,
            sem=np.array([0.1, 0.2]),
            variance=None,
            transform=transform,
            lower_bound=1.0,
            upper_bound=1.9,
        )
        self.assertTrue(np.allclose(new_mean, np.array([2.0, 3.8])))
        # Bounds are set to clip the CI to the mean from one side only.
        # We end up with doubling the halved CI width, ending up with the original sem.
        self.assertTrue(np.allclose(new_sem, np.array([0.1, 0.2])))

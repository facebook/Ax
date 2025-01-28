#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
from ax.models.discrete.ashr_utils import (
    Ashr,
    fit_ashr_em,
    GaussianMixture,
    marginal_densities,
    prior_grid,
)
from ax.utils.common.testutils import TestCase


class AshrModelTest(TestCase):
    def test_gaussian_mixture(self) -> None:
        normal_vars_by_class = np.array(
            [[0.0, 1.0, 1.0], [0.0, 1.0, 0.25], [0.0, 2.25, 1.0], [0.0, 4.0, 4.0]]
        )

        gm = GaussianMixture(
            normal_means_by_class=np.zeros((4, 3)),
            normal_vars_by_class=normal_vars_by_class,
            weights=np.ones((4, 3)) / 3.0,
        )

        variances = normal_vars_by_class.sum(axis=1) / 3.0
        self.assertTrue(np.allclose(gm.means, np.zeros(4)))
        self.assertTrue(np.allclose(gm.vars, variances))
        self.assertTrue(
            np.allclose(gm.tail_probabilities(left_tail=True), np.ones(4) / 3)
        )
        self.assertTrue(
            np.allclose(gm.tail_probabilities(left_tail=False), np.ones(4) / 3)
        )

    def test_prior_grid(self) -> None:
        prior_stds = prior_grid(
            Y=np.array([1, 2]), Yvar=np.array([1, 1]), grid_param=16.0
        )
        self.assertTrue((prior_stds == np.array([0.0, 0.1, 0.4, 1.6, 6.4])).all())

    def test_marginal_densities(self) -> None:
        ll = marginal_densities(
            Y=np.zeros(2), Yvar=np.array([1, 2]), prior_vars=np.array([0, 1])
        )
        self.assertTrue(
            (
                ll
                == np.array(
                    [
                        [1 / np.sqrt(2 * np.pi), 1 / np.sqrt(2 * 2 * np.pi)],
                        [1 / np.sqrt(2 * 2 * np.pi), 1 / np.sqrt(3 * 2 * np.pi)],
                    ]
                )
            ).all()
        )

    def test_fit_ashr_em(self) -> None:
        results = fit_ashr_em(
            ll=np.array([[1, 2], [0, 3]]), lambdas=np.ones(2), threshold=1.0, nsteps=1
        )
        self.assertTrue(np.allclose(results["pi"], np.array([1 / 6, 5 / 6])))
        self.assertTrue(
            np.allclose(
                results["weights"], np.array([[1.0 / 11, 10.0 / 11], [0.0, 1.0]])
            )
        )
        self.assertTrue(np.allclose(results["lfdr"], np.array([1.0 / 11, 0.0])))

    def test_ashr_posterior(self) -> None:
        a = Ashr(Y=np.ones(2), Yvar=np.ones(2), prior_vars=np.array([0, 1]))
        gm = a.posterior(w=np.ones((2, 2)))
        self.assertTrue(
            (gm.normal_means_by_class == np.array([[0, 0.5], [0, 0.5]])).all()
        )
        self.assertTrue(
            (gm.normal_vars_by_class == np.array([[0, 0.5], [0, 0.5]])).all()
        )

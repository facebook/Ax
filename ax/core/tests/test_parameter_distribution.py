#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.parameter_distribution import ParameterDistribution
from ax.core.search_space import RobustSearchSpace
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space
from scipy.stats._continuous_distns import norm_gen
from scipy.stats._distn_infrastructure import rv_frozen


class ParameterDistributionTest(TestCase):
    def test_parameter_distribution(self) -> None:
        dist = ParameterDistribution(
            parameters=["x1"],
            distribution_class="norm",
            distribution_parameters={"loc": 0.0, "scale": 1.0},
            multiplicative=True,
        )
        self.assertTrue(dist.multiplicative)
        self.assertIsNone(dist._distribution)
        dist_obj = dist.distribution
        self.assertEqual(dist.parameters, ["x1"])
        self.assertIsInstance(dist_obj, rv_frozen)
        self.assertIsInstance(dist_obj.dist, norm_gen)
        dist_kwds = dist_obj.kwds
        self.assertEqual(dist_kwds["loc"], 0.0)
        self.assertEqual(dist_kwds["scale"], 1.0)

        # Update the dist class.
        self.assertIsNotNone(dist._distribution)
        dist.distribution_class = "cauchy"
        self.assertIsNone(dist._distribution)
        # Update the parameters.
        dist.distribution  # Re-construct the distribution.
        self.assertIsNotNone(dist._distribution)
        dist.distribution_parameters = {"loc": 0.1, "scale": 0.5}
        self.assertIsNone(dist._distribution)

        # Test repr.
        expected_repr = (
            "ParameterDistribution("
            "parameters=['x1'], "
            "distribution_class=cauchy, "
            "distribution_parameters={'loc': 0.1, 'scale': 0.5}, "
            "multiplicative=True)"
        )
        self.assertEqual(str(dist), expected_repr)

        # Test weird distribution name.
        dist = ParameterDistribution(
            parameters=["x1"],
            distribution_class="dummy_dist",
            distribution_parameters={},
        )
        self.assertFalse(dist.multiplicative)
        with self.assertRaises(UserInputError):
            dist.distribution

        # Test `is_environmental`.
        rss = get_robust_search_space()
        self.assertFalse(rss.parameter_distributions[0].is_environmental(rss))
        params = list(rss.parameters.values())
        rss = RobustSearchSpace(
            parameters=params[2:],
            parameter_distributions=rss.parameter_distributions,
            num_samples=4,
            environmental_variables=params[:2],
        )
        self.assertTrue(rss.parameter_distributions[0].is_environmental(rss))

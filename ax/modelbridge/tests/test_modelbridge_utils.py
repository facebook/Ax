#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.core.search_space import RobustSearchSpace
from ax.modelbridge.modelbridge_utils import extract_parameter_distribution_samplers
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space
from ax.utils.testing.core_stubs import get_search_space


class TestModelBridgeUtils(TestCase):
    def test_extract_parameter_distribution_samplers(self):
        # Test with non-robust search space.
        ss = get_search_space()
        self.assertEqual(
            extract_parameter_distribution_samplers(ss, list(ss.parameters)),
            (None, False),
        )
        # Test with non-environmental search space.
        for multiplicative in (True, False):
            rss = get_robust_search_space()
            if multiplicative:
                for p in rss.parameter_distributions:
                    p.multiplicative = True
            sampler, mul_ = extract_parameter_distribution_samplers(
                rss, list(rss.parameters)
            )
            self.assertEqual(mul_, multiplicative)
            samples = sampler(5)
            self.assertEqual(samples.shape, (5, 4))
            constructor = np.ones if multiplicative else np.zeros
            self.assertTrue(np.equal(samples[:, 2:], constructor((5, 2))).all())
            # Exponential distribution is non-negative, so we can check for that.
            self.assertTrue(np.all(samples[:, 1] > 0))
            # Check that it works as expected if param_names is missing some
            # non-distributional parameters.
            sampler, mul_ = extract_parameter_distribution_samplers(
                rss, list(rss.parameters)[:-1]
            )
            samples = sampler(5)
            self.assertEqual(samples.shape, (5, 3))
            self.assertTrue(np.equal(samples[:, 2:], constructor((5, 1))).all())
            self.assertTrue(np.all(samples[:, 1] > 0))
            # Check that it errors if we're missing distributional parameters.
            with self.assertRaisesRegex(RuntimeError, "All distributional"):
                extract_parameter_distribution_samplers(rss, list(rss.parameters)[1:])
        # Test with environmental search space.
        all_params = list(rss.parameters.values())
        rss = RobustSearchSpace(
            parameters=all_params[2:],
            parameter_distributions=rss.parameter_distributions,
            environmental_variables=all_params[:2],
        )
        sampler, mul_ = extract_parameter_distribution_samplers(
            rss, list(rss.parameters)
        )
        self.assertFalse(mul_)
        samples = sampler(5)
        self.assertEqual(samples.shape, (5, 2))
        # Both are continuous distributions, should be non-zero.
        self.assertTrue(np.all(samples != 0))
        # Check for error if environmental variables are not at the end.
        with self.assertRaisesRegex(RuntimeError, "last entries"):
            extract_parameter_distribution_samplers(rss, list(rss.parameters)[::-1])

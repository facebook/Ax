#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from ax.benchmark.noise import GaussianMixtureNoise, GaussianNoise
from ax.utils.common.testutils import TestCase


class TestGaussianNoise(TestCase):
    def test_gaussian_noise(self) -> None:
        with self.subTest("get_noise_stds with float"):
            noise = GaussianNoise(noise_std=1.0)
            result = noise.get_noise_stds(outcome_names=["obj_0", "obj_1"])
            self.assertEqual(result, {"obj_0": 1.0, "obj_1": 1.0})

        with self.subTest("get_noise_stds with dict"):
            noise_dict = {"obj_0": 0.5, "obj_1": 0.3}
            noise = GaussianNoise(noise_std=noise_dict)
            result = noise.get_noise_stds(outcome_names=["obj_0", "obj_1"])
            self.assertEqual(result, noise_dict)

        with self.subTest("get_noise_stds with mismatched keys raises"):
            noise = GaussianNoise(noise_std={"wrong_key": 0.5})
            with self.assertRaisesRegex(
                ValueError, "Noise std must have keys equal to outcome names"
            ):
                noise.get_noise_stds(outcome_names=["obj_0"])

        with self.subTest("is_noiseless property"):
            self.assertTrue(GaussianNoise(noise_std=0.0).is_noiseless)
            self.assertTrue(GaussianNoise(noise_std={"a": 0.0, "b": 0.0}).is_noiseless)
            self.assertFalse(GaussianNoise(noise_std=0.1).is_noiseless)

        with self.subTest("noiseless case - mean equals Y_true"):
            noise = GaussianNoise(noise_std=0.0)
            df = pd.DataFrame(
                {
                    "Y_true": [1.0, 2.0, 3.0],
                    "metric_name": ["obj_0", "obj_0", "obj_0"],
                    "arm_name": ["0_0", "0_0", "0_0"],
                }
            )
            result = noise.add_noise(
                df=df.copy(),
                trial=None,
                outcome_names=["obj_0"],
                arm_weights=None,
            )
            self.assertTrue(np.array_equal(result["mean"], df["Y_true"]))
            self.assertTrue((result["sem"] == 0.0).all())

        with self.subTest("noisy case adds noise and sets sem"):
            noise_std = 2.0
            noise = GaussianNoise(noise_std=noise_std)
            df = pd.DataFrame(
                {
                    "Y_true": [1.0, 2.0, 3.0],
                    "metric_name": ["obj_0", "obj_0", "obj_0"],
                    "arm_name": ["0_0", "0_0", "0_0"],
                }
            )

            mock_noise = np.array([0.5, -0.3, 0.8])
            with patch("numpy.random.normal", return_value=mock_noise):
                result = noise.add_noise(
                    df=df.copy(),
                    trial=None,
                    outcome_names=["obj_0"],
                    arm_weights=None,
                )

            expected_mean = np.array([1.5, 1.7, 3.8])
            self.assertTrue(np.allclose(result["mean"].to_numpy(), expected_mean))
            self.assertTrue((result["sem"] == noise_std).all())

        with self.subTest("arm weights scale sem correctly"):
            noise = GaussianNoise(noise_std=1.0)
            df = pd.DataFrame(
                {
                    "Y_true": [1.0, 2.0],
                    "metric_name": ["obj_0", "obj_0"],
                    "arm_name": ["0_0", "0_1"],
                }
            )
            arm_weights = {"0_0": 1.0, "0_1": 4.0}

            mock_noise = np.array([0.1, 0.2])
            with patch("numpy.random.normal", return_value=mock_noise):
                result = noise.add_noise(
                    df=df.copy(),
                    trial=None,
                    outcome_names=["obj_0"],
                    arm_weights=arm_weights,
                )

            # sem = noise_std / sqrt(weight / sum_weights)
            expected_sem_0 = 1.0 / np.sqrt(1.0 / 5.0)  # sqrt(5) ≈ 2.236
            expected_sem_1 = 1.0 / np.sqrt(4.0 / 5.0)  # sqrt(1.25) ≈ 1.118

            self.assertAlmostEqual(result["sem"].iloc[0], expected_sem_0, places=5)
            self.assertAlmostEqual(result["sem"].iloc[1], expected_sem_1, places=5)


class TestGaussianMixtureNoise(TestCase):
    def test_gaussian_mixture_noise(self) -> None:
        with self.subTest("mixture_weights must sum to 1"):
            weights = torch.tensor([0.5, 0.3])  # sums to 0.8, not 1.0
            means = torch.tensor([0.0, 1.0])
            stds = torch.tensor([1.0, 1.0])
            with self.assertRaisesRegex(ValueError, "mixture_weights must sum to 1"):
                GaussianMixtureNoise(weights=weights, means=means, stds=stds)

        with self.subTest("add_noise returns correct structure with NaN sem"):
            weights = torch.tensor([1.0])
            means = torch.tensor([0.0])
            stds = torch.tensor([1.0])
            noise = GaussianMixtureNoise(weights=weights, means=means, stds=stds)

            df = pd.DataFrame(
                {
                    "Y_true": [1.0, 2.0, 3.0],
                    "metric_name": ["obj_0", "obj_0", "obj_0"],
                    "arm_name": ["0_0", "0_0", "0_0"],
                }
            )

            mock_samples = torch.tensor([0.5, -0.3, 0.8])
            with patch.object(noise._distribution, "sample", return_value=mock_samples):
                result = noise.add_noise(
                    df=df.copy(),
                    trial=None,
                    outcome_names=["obj_0"],
                    arm_weights=None,
                )

            expected_mean = np.array([1.5, 1.7, 3.8])
            self.assertTrue(np.allclose(result["mean"].to_numpy(), expected_mean))
            self.assertTrue(result["sem"].isna().all())

        with self.subTest("scale parameter multiplies noise"):
            weights = torch.tensor([1.0])
            means = torch.tensor([0.0])
            stds = torch.tensor([1.0])
            noise = GaussianMixtureNoise(
                weights=weights, means=means, stds=stds, scale=2.0
            )

            df = pd.DataFrame(
                {
                    "Y_true": [0.0, 0.0],
                    "metric_name": ["obj_0", "obj_0"],
                    "arm_name": ["0_0", "0_0"],
                }
            )

            mock_samples = torch.tensor([1.0, -1.0])
            with patch.object(noise._distribution, "sample", return_value=mock_samples):
                result = noise.add_noise(
                    df=df.copy(),
                    trial=None,
                    outcome_names=["obj_0"],
                    arm_weights=None,
                )

            expected_mean = np.array([2.0, -2.0])
            self.assertTrue(np.allclose(result["mean"].to_numpy(), expected_mean))

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Noise classes for benchmark problems.

Each `BenchmarkProblem` specifies a `Noise` instance that determines how
noise is added to the ground-truth evaluations. This allows for
mixing and matching of test functions (specifying the mean) with noise
models.

The abstract base class is `Noise`; subclasses include `GaussianNoise`
and `GaussianMixtureNoise`.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from ax.core.base_trial import BaseTrial
from torch import Tensor
from torch.distributions import Categorical, MixtureSameFamily, Normal


@dataclass(kw_only=True)
class Noise(ABC):
    """
    Abstract base class for noise in benchmark problems.

    A `Noise` object is responsible for adding noise to the ground-truth
    evaluations produced by a `BenchmarkTestFunction`.

    Subclasses must implement `_get_noise_and_sem` to specify how noise
    samples and standard errors are generated.
    """

    @abstractmethod
    def _get_noise_and_sem(
        self,
        df: pd.DataFrame,
        outcome_names: Sequence[str],
        arm_weights: Mapping[str, float] | None,
    ) -> tuple[npt.NDArray, npt.NDArray | float]:
        """
        Generate noise samples and standard errors for each row in the DataFrame.

        Args:
            df: A DataFrame with columns including
                ["metric_name", "arm_name", "Y_true"].
            outcome_names: The names of the outcomes.
            arm_weights: Mapping from arm name to weight, or None for
                single-arm trials.

        Returns:
            A tuple of (noise_samples, sem) where:
                - noise_samples: Array of noise values to add to Y_true
                - sem: Array of standard errors (or a scalar like NaN)
        """
        ...

    def add_noise(
        self,
        df: pd.DataFrame,
        trial: BaseTrial | None,
        outcome_names: Sequence[str],
        arm_weights: Mapping[str, float] | None,
    ) -> pd.DataFrame:
        """
        Add noise to the ground-truth evaluations.

        This method is the same for all Noise subclasses. It calls
        `_get_noise_and_sem` to get the noise samples and standard errors,
        then adds them to the DataFrame.

        Args:
            df: A DataFrame with columns including
                ["metric_name", "arm_name", "Y_true"].
            trial: The trial being evaluated.
            outcome_names: The names of the outcomes.
            arm_weights: Mapping from arm name to weight, or None for
                single-arm trials. Using arm weights will increase noise
                levels, since each arm is assumed to receive a fraction
                of the total sample budget.

        Returns:
            The original `df`, now with additional columns ["mean", "sem"].
        """
        noise, sem = self._get_noise_and_sem(
            df=df, outcome_names=outcome_names, arm_weights=arm_weights
        )
        df["mean"] = df["Y_true"] + noise
        df["sem"] = sem
        return df


@dataclass(kw_only=True)
class GaussianNoise(Noise):
    """
    Gaussian (normal) noise with specified standard deviation.

    This is the most common noise model for benchmark problems, where
    IID random normal noise is added to each observation.

    Args:
        noise_std: The standard deviation of the noise. Can be:
            - A float: The same noise level is used for all outcomes.
            - A mapping from outcome name to noise level: Different noise
              levels for specific outcomes.
    """

    noise_std: float | Mapping[str, float] = 0.0

    def get_noise_stds(self, outcome_names: Sequence[str]) -> dict[str, float]:
        """
        Get a dictionary mapping outcome names to noise standard deviations.

        Args:
            outcome_names: The names of the outcomes.

        Returns:
            A dictionary mapping each outcome name to its noise standard deviation.
        """
        noise_std = self.noise_std
        if isinstance(noise_std, float | int):
            return {name: float(noise_std) for name in outcome_names}
        if not set(noise_std.keys()) == set(outcome_names):
            raise ValueError(
                "Noise std must have keys equal to outcome names if given as a dict."
            )
        return dict(noise_std)

    @property
    def is_noiseless(self) -> bool:
        """Whether this noise model adds no noise."""
        noise_std = self.noise_std
        if isinstance(noise_std, float | int):
            return noise_std == 0.0
        return all(v == 0 for v in noise_std.values())

    def _get_noise_and_sem(
        self,
        df: pd.DataFrame,
        outcome_names: Sequence[str],
        arm_weights: Mapping[str, float] | None,
    ) -> tuple[npt.NDArray, npt.NDArray | float]:
        """
        Generate Gaussian noise samples and standard errors.

        For each row in ``df``, compute the standard error based on
        ``noise_stds[metric_name]`` adjusted by arm weights if applicable,
        then sample noise from a normal distribution with that standard error.

        Args:
            df: A DataFrame with columns ["metric_name", "arm_name", "Y_true"].
            outcome_names: The names of the outcomes.
            arm_weights: Mapping from arm name to weight, or None.

        Returns:
            A tuple of (noise_samples, sem_array).
        """
        noise_stds = self.get_noise_stds(outcome_names)
        noiseless = all(v == 0 for v in noise_stds.values())

        if noiseless:
            return np.zeros(len(df)), 0.0

        noise_std_ser = df["metric_name"].map(noise_stds)
        if arm_weights is not None:
            nlzd_arm_weights_sqrt = {
                arm_name: sqrt(weight / sum(arm_weights.values()))
                for arm_name, weight in arm_weights.items()
            }
            arm_weights_ser = df["arm_name"].map(nlzd_arm_weights_sqrt)
            sem = noise_std_ser / arm_weights_ser
        else:
            sem = noise_std_ser

        noise = np.random.normal(loc=0, scale=sem)
        return noise, sem.to_numpy()


def _create_gaussian_mixture(
    mixture_weights: Tensor,
    mixture_means: Tensor,
    mixture_stds: Tensor,
) -> MixtureSameFamily:
    """Create a Gaussian mixture distribution using PyTorch distributions.

    Args:
        mixture_weights: Weights for each Gaussian component (must sum to 1).
        mixture_means: Means for each Gaussian component.
        mixture_stds: Standard deviations for each Gaussian component.

    Returns:
        A MixtureSameFamily distribution representing the Gaussian mixture.
    """
    weight_sum = mixture_weights.sum().item()
    if not torch.isclose(torch.tensor(weight_sum), torch.tensor(1.0)):
        raise ValueError(f"mixture_weights must sum to 1, got {weight_sum}")
    mix = Categorical(probs=mixture_weights)
    comp = Normal(loc=mixture_means, scale=mixture_stds)
    return MixtureSameFamily(mix, comp)


@dataclass(kw_only=True)
class GaussianMixtureNoise(Noise):
    """
    Gaussian mixture noise for benchmark problems with non-Gaussian noise.

    This noise model samples from a mixture of Gaussians, which can
    represent more complex noise distributions than a single Gaussian.

    The noise is scaled by `scale` to match the scale of the outcomes.

    Args:
        weights: Weights for each Gaussian component (must sum to 1).
        means: Means for each Gaussian component.
        stds: Standard deviations for each Gaussian component.
        scale: Scaling factor for the noise (typically the standard
            deviation of the true outcomes).
    """

    weights: Tensor
    means: Tensor
    stds: Tensor
    scale: float = 1.0
    _distribution: MixtureSameFamily = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._distribution = _create_gaussian_mixture(
            self.weights, self.means, self.stds
        )

    def _get_noise_and_sem(
        self,
        df: pd.DataFrame,
        outcome_names: Sequence[str],
        arm_weights: Mapping[str, float] | None,
    ) -> tuple[npt.NDArray, npt.NDArray | float]:
        """
        Generate Gaussian mixture noise samples.

        Args:
            df: A DataFrame with columns ["metric_name", "arm_name", "Y_true"].
            outcome_names: The names of the outcomes (not used).
            arm_weights: Mapping from arm name to weight (not used).

        Returns:
            A tuple of (noise_samples, NaN) since GMM noise doesn't have
            a simple standard error representation.
        """
        n_samples = len(df)
        noise = self._distribution.sample((n_samples,)).numpy() * self.scale
        return noise, float("nan")

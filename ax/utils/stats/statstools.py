#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from logging import Logger
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from ax.utils.common.logger import get_logger
from ax.utils.stats.math_utils import relativize

logger: Logger = get_logger(__name__)
# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
num_mixed = Union[np.ndarray, list[float]]


def inverse_variance_weight(
    means: npt.NDArray,
    variances: npt.NDArray,
    conflicting_noiseless: str = "warn",
) -> tuple[float, float]:
    """Perform inverse variance weighting.

    Args:
        means: The means of the observations.
        variances: The variances of the observations.
        conflicting_noiseless: How to handle the case of
            multiple observations with zero variance but different means.
            Options are "warn" (default), "ignore" or "raise".

    """
    if conflicting_noiseless not in {"warn", "ignore", "raise"}:
        raise ValueError(
            f"Unsupported option `{conflicting_noiseless}` for conflicting_noiseless."
        )
    if len(means) != len(variances):
        raise ValueError("Means and variances must be of the same length.")
    # new_mean = \sum_i 1/var_i mean_i / \sum_i (1/var_i), unless any var = 0,
    # in which case we report the mean of all values with var = 0.
    idx_zero = variances == 0
    if idx_zero.any():
        means_z = means[idx_zero]
        if np.var(means_z) > 0:
            message = "Multiple observations zero variance but different means."
            if conflicting_noiseless == "warn":
                logger.warning(message)
            elif conflicting_noiseless == "raise":
                raise ValueError(message)
        return np.mean(means_z), 0
    inv_vars = np.divide(1.0, variances)
    sum_inv_vars = inv_vars.sum()
    new_mean = np.inner(inv_vars, means) / sum_inv_vars
    new_var = np.divide(1.0, sum_inv_vars)
    return new_mean, new_var


def total_variance(
    means: npt.NDArray,
    variances: npt.NDArray,
    sample_sizes: npt.NDArray,
) -> float:
    """Compute total variance."""
    variances = variances * sample_sizes
    weighted_variance_of_means = np.average(
        (means - means.mean()) ** 2, weights=sample_sizes
    )
    weighted_mean_of_variance = np.average(variances, weights=sample_sizes)
    return (weighted_variance_of_means + weighted_mean_of_variance) / sample_sizes.sum()


def positive_part_james_stein(
    means: num_mixed,
    sems: num_mixed,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Estimation method for Positive-part James-Stein estimator.

    This method takes a vector of K means (`y_i`) and standard errors
    (`sigma_i`) and calculates the positive-part James Stein estimator.

    Resulting estimates are the shrunk means and standard errors. The positive
    part James-Stein estimator shrinks each constituent average to the grand
    average:

        y_i - phi_i * y_i + phi_i * ybar

    The variable phi_i determines the amount of shrinkage. For `phi_i = 1`,
    `mu_hat` is equal to `ybar` (the mean of all `y_i`), while for `phi_i = 0`,
    `mu_hat` is equal to `y_i`. It can be shown that restricting `phi_i <= 1`
    dominates the unrestricted estimator, so this method restricts `phi_i` in
    this manner. The amount of shrinkage, `phi_i`, is determined by:

        (K - 3) * sigma2_i / s2

    That is, less shrinkage is applied when individual means are estimated with
    greater precision, and more shrinkage is applied when individual means are
    very tightly clustered together. We also restrict `phi_i` to never be larger
    than 1.

    The variance of the mean estimator is:

        (1 - phi_i) * sigma2_i
        + phi * sigma2_i / K
        + 2 * phi_i ** 2 * (y_i - ybar)^2 / (K - 3)

    The first term is the variance component from `y_i`, the second term is the
    contribution from the mean of all `y_i`, and the third term is the
    contribution from the uncertainty in the sum of squared deviations of `y_i`
    from the mean of all `y_i`.

    For more information, see
    https://ax.dev/docs/models.html#empirical-bayes-and-thompson-sampling.

    Args:
        means: Means of each arm
        sems: Standard errors of each arm
    Returns:
        mu_hat_i: Empirical Bayes estimate of each arm's mean
        sem_i: Empirical Bayes estimate of each arm's sem
    """
    if np.min(sems) < 0:
        raise ValueError("sems cannot be negative.")
    y_i = np.array(means)
    K = y_i.shape[0]
    if K < 4:
        raise ValueError(
            "Less than 4 measurements passed to positive_part_james_stein. "
            + "Returning raw estimates."
        )
    sigma2_i = np.power(sems, 2)
    ybar = np.mean(y_i)
    s2 = np.var(y_i - ybar, ddof=3)  # sample variance normalized by K-3
    phi_i = np.ones_like(sigma2_i) if s2 == 0 else np.minimum(1, sigma2_i / s2)
    mu_hat_i = y_i + phi_i * np.subtract(ybar, y_i)

    sigma_hat_i = np.sqrt(
        np.subtract(1.0, phi_i) * sigma2_i
        + phi_i * sigma2_i / K
        + np.multiply(2, phi_i**2) * (y_i - ybar) ** 2 / (K - 3)
    )
    return mu_hat_i, sigma_hat_i


def agresti_coull_sem(
    n_numer: pd.Series | npt.NDArray | int,
    n_denom: pd.Series | npt.NDArray | int,
    prior_successes: int = 2,
    prior_failures: int = 2,
) -> npt.NDArray | float:
    """Compute the Agresti-Coull style standard error for a binomial proportion.

    Reference:
    *Agresti, Alan, and Brent A. Coull. Approximate Is Better than 'Exact' for
    Interval Estimation of Binomial Proportions." The American Statistician,
    vol. 52, no. 2, 1998, pp. 119-126. JSTOR, www.jstor.org/stable/2685469.*

    """
    n_numer = np.array(n_numer)
    n_denom = np.array(n_denom)
    p_for_sem = (n_numer + prior_successes) / (
        n_denom + prior_successes + prior_failures
    )
    sem = np.sqrt(p_for_sem * (1 - p_for_sem) / n_denom)
    return sem


def marginal_effects(
    df: pd.DataFrame, covariates: list[str] | None = None
) -> pd.DataFrame:
    """
    This method calculates the relative (in %) change in the outcome achieved
    by using any individual factor level versus randomizing across all factor
    levels. It does this by estimating a baseline under the experiment by
    marginalizing over all factors/levels. For each factor level, then,
    it conditions on that level for the individual factor and then marginalizes
    over all levels for all other factors.

    Args:
        df: Dataframe containing columns named mean and sem. All other columns
            are assumed to be factors for which to calculate marginal effects.
        covariates: List of columns to be used as covariates. If None, then use
            all columns in df that are not named "mean" or "sem".

    Returns:
        A dataframe containing columns "Name", "Level", "Beta" and "SE"
            corresponding to the factor, level, effect and standard error.
            Results are relativized as percentage changes.
    """
    covariates = covariates or [col for col in df.columns if col not in ["mean", "sem"]]
    formatted_vals = []
    overall_mean, overall_sem = inverse_variance_weight(
        df["mean"],
        np.power(df["sem"], 2),
    )
    for cov in covariates:
        if len(df[cov].unique()) <= 1:
            next
        df_gb = df.groupby(cov)
        for name, group_df in df_gb:
            group_mean, group_var = inverse_variance_weight(
                group_df["mean"], np.power(group_df["sem"], 2)
            )
            effect, effect_sem = relativize(
                group_mean,
                np.sqrt(group_var),
                overall_mean,
                overall_sem,
                cov_means=0.0,
                as_percent=True,
            )
            formatted_vals.append(
                {"Name": cov, "Level": name, "Beta": effect, "SE": effect_sem}
            )
    return pd.DataFrame(formatted_vals)[["Name", "Level", "Beta", "SE"]]

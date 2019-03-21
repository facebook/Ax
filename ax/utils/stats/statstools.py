#!/usr/bin/env python3

from typing import Dict, List, NamedTuple, Tuple, Union

import numpy as np
import pandas as pd
from ax.core.data import Data
from ax.utils.common.logger import get_logger
from scipy.linalg import cho_factor, cho_solve


logger = get_logger("Statstools")
num_mixed = Union[np.ndarray, List[float]]


class CondMeanVar(NamedTuple):
    cond_mean: float
    cond_var: float


def two_way_eb(
    arms: np.ndarray,
    contexts: np.ndarray,
    mus: np.ndarray,
    ns: np.ndarray,
    sems: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical Bayes method-of-moments estimator for the two-way model. See
    Casella et al., 'Variance Components', 1992, pg 429.
    df has columns: mean, sem, n, arm, context.
    """
    df = pd.DataFrame(
        {"arm": arms, "context": contexts, "mean": mus, "n": ns, "sem": sems}
    )
    df.set_index("context", inplace=True)
    df["strata_mean"] = df.groupby("context").apply(
        lambda x: np.average(x["mean"], weights=x["n"])
    )
    df["strata_normalized_mean"] = df["mean"] / df["strata_mean"]
    df["strata_normalized_sem"] = df["sem"] / df["strata_mean"]
    df["y_ijdot"] = df["strata_normalized_mean"] * df["n"]

    df.reset_index(inplace=True)
    total_samples = df["n"].sum()
    n_arms = df["arm"].nunique()
    n_contexts = df["context"].nunique()
    b_dot = df.shape[0]

    if b_dot != n_arms * n_contexts:
        raise ValueError("Missing data for some arm x context pairs.")

    k_1 = (df.groupby("arm")["n"].sum() ** 2).sum() / total_samples

    k_3 = (df["n"] ** 2).sum() / total_samples

    k_12_df = (
        df.groupby("arm")[["n"]]
        .agg([sum, lambda x: sum(x ** 2)])
        .rename(columns={"<lambda>": "sum_sq"})
    )
    k_12_df.columns = k_12_df.columns.droplevel()
    k_12 = (k_12_df["sum_sq"] / k_12_df["sum"]).sum()

    T_A_df = df.groupby("arm")["y_ijdot", "n"].agg("sum")
    T_A = (T_A_df["y_ijdot"] ** 2 / T_A_df["n"]).sum()

    T_AB = (df["y_ijdot"] ** 2 / df["n"]).sum()

    T_mu = (df["y_ijdot"]).sum() ** 2 / total_samples

    var_eps = (df["strata_normalized_sem"] ** 2 * (df["n"] - 1) * df["n"]).sum() / (
        total_samples - b_dot
    )
    var_theta = max(
        (T_AB - T_A - (b_dot - n_arms) * var_eps) / (total_samples - k_12), 0
    )
    var_alpha = max(
        (T_A - T_mu - (k_12 - k_3) * var_theta - (n_arms - 1) * var_eps)
        / (total_samples - k_1),
        0,
    )

    mu = (df["strata_normalized_mean"] * df["n"]).sum() / df["n"].sum()

    cond_moments = _get_cond_moments(
        y=df["strata_normalized_mean"].as_matrix(),
        mu=mu,
        var_eps_i=df["strata_normalized_sem"].as_matrix() ** 2,
        var_alpha=var_alpha,
        var_theta=var_theta,
        n_arms=n_arms,
        n_contexts=n_contexts,
    )

    return (
        cond_moments.cond_mean * df["strata_mean"].values,
        np.diagonal(cond_moments.cond_var) * (df["strata_mean"] ** 2).values,
    )


def _get_var_eta(
    var_alpha: float, var_theta: float, n_arms: int, n_contexts: int
) -> np.ndarray:
    # eta is ordered as (eta_1., eta_2., eta_I.), i.e. grouping together
    # and concatenating observations according to the i index, not the g index.
    return np.kron(np.identity(n_arms), np.identity(n_contexts) * var_theta + var_alpha)


def _get_var_y(
    var_eps_i: float, var_alpha: float, var_theta: float, n_arms: int, n_contexts: int
) -> np.ndarray:
    var_y = _get_var_eta(var_alpha, var_theta, n_arms, n_contexts)
    np.fill_diagonal(var_y, np.diagonal(var_y) + var_eps_i)
    return var_y


def _get_cond_moments(
    y: float,
    mu: float,
    var_eps_i: float,
    var_alpha: float,
    var_theta: float,
    n_arms: int,
    n_contexts: int,
) -> CondMeanVar:
    """
    Calculate the conditional mean of the unknown parameters of interest
    (the eta's), given the data y and the parameters of the data generating
    process.
    """
    var_eta = _get_var_eta(var_alpha, var_theta, n_arms, n_contexts)
    var_y = _get_var_y(var_eps_i, var_alpha, var_theta, n_arms, n_contexts)
    # pyre-fixme[16]: Optional type has no attribute `decomp_cholesky`.
    L = cho_factor(var_y)
    # pyre-fixme[16]: Optional type has no attribute `decomp_cholesky`.
    cond_mean = mu + var_eta @ cho_solve(L, y - mu)
    # pyre-fixme[16]: Optional type has no attribute `decomp_cholesky`.
    cond_var = var_eta - var_eta @ cho_solve(L, var_eta.T)
    return CondMeanVar(cond_mean=cond_mean, cond_var=cond_var)


def inverse_variance_weight(
    means: np.ndarray, variances: np.ndarray, conflicting_noiseless: str = "warn"
) -> Tuple[float, float]:
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
    means: np.ndarray, variances: np.ndarray, sample_sizes: np.ndarray
) -> float:
    """Compute total variance."""
    variances = variances * sample_sizes
    weighted_variance_of_means = np.average(
        (means - means.mean()) ** 2, weights=sample_sizes
    )
    weighted_mean_of_variance = np.average(variances, weights=sample_sizes)
    return (weighted_variance_of_means + weighted_mean_of_variance) / sample_sizes.sum()


def positive_part_james_stein(
    means: num_mixed, sems: num_mixed
) -> Tuple[np.ndarray, np.ndarray]:
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

    For more information, see https://fburl.com/empirical_bayes.

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
    if s2 == 0:
        phi_i = 1
    else:
        phi_i = np.minimum(1, sigma2_i / s2)
    mu_hat_i = y_i + phi_i * (ybar - y_i)
    sigma_hat_i = np.sqrt(
        (1 - phi_i) * sigma2_i
        + phi_i * sigma2_i / K
        + 2 * phi_i ** 2 * (y_i - ybar) ** 2 / (K - 3)
    )
    return mu_hat_i, sigma_hat_i


def relativize(
    means_t: Union[np.ndarray, List[float], float],
    sems_t: Union[np.ndarray, List[float], float],
    mean_c: float,
    sem_c: float,
    bias_correction: bool = True,
    cov_means: Union[np.ndarray, List[float], float] = 0.0,
    as_percent: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ratio estimator based on the delta method. Adapted from Deltoid3

    This uses the delta method (i.e. a Taylor series approximation) to estimate
    the mean and standard deviation of the sampling distribution of the ratio
    between test and control -- that is, the sampling distribution of an
    estimator of the true population value under the assumption that the means
    in test and control have a known covariance:

        (mu_t / mu_c) - 1.

    Under a second-order Taylor expansion, the sampling distribution of the
    relative change in empirical means, which is `m_t / m_c - 1`, is
    approximately normally distributed with mean

        [(mu_t - mu_c) / mu_c] - [(sigma_c)^2 * mu_t] / (mu_c)^3

    and variance

        (sigma_t / mu_c)^2
        - 2 * mu_t _ sigma_tc / mu_c^3
        + [(sigma_c * mu_t)^2 / (mu_c)^4]

    as the higher terms are assumed to be close to zero in the full Taylor
    series. To estimate these parameters, we plug in the empirical means and
    standard errors. This gives us the estimators:

        [(m_t - m_c) / m_c] - [(s_c)^2 * m_t] / (m_c)^3

    and

        (s_t / m_c)^2 - 2 * m_t * s_tc / m_c^3 + [(s_c * m_t)^2 / (m_c)^4]

    Note that the delta method does NOT take as input the empirical standard
    deviation of a metric, but rather the standard error of the mean of that
    metric -- that is, the standard deviation of the metric after division by
    the square root of the total number of observations.

    Args:
        means_t: Sample means (test)
        sems_t: Sample standard errors of the means (test)
        mean_c: Sample mean (control)
        sem_c: Sample standard error of the mean (control)
        cov_means: Sample covariance between test and control
        as_percent: If true, return results in percent (* 100)

    Returns:
        rel_hat: Inferred means of the sampling distribution of
            the relative change `(mean_t / mean_c) - 1`
        sem_hat: Inferred standard deviation of the sampling
            distribution of rel_hat -- i.e. the standard error.

    """
    # if mean_c is too small, bail (taken from Deltoid3)
    epsilon = 1e-10
    if np.any(np.abs(mean_c) < epsilon):
        raise ValueError(
            "mean_control ({0} +/- {1}) is smaller than 1 in 10 billion, "
            "which is too small to reliably analyze ratios using the delta "
            "method. This usually occurs because winsorization has truncated "
            "all values down to zero. Try using a delta type that applies "
            "no winsorization.".format(mean_c, sem_c)
        )
    m_t = np.array(means_t)
    s_t = np.array(sems_t)
    cov_t = np.array(cov_means)
    c = m_t / mean_c
    r_hat = (m_t - mean_c) / np.abs(mean_c)
    if bias_correction:
        r_hat = r_hat - m_t * sem_c ** 2 / np.abs(mean_c) ** 3
    # If everything's the same, then set r_hat to zero
    same = (m_t == mean_c) & (s_t == sem_c)
    r_hat = ~same * r_hat
    var = ((s_t ** 2) - 2 * c * cov_t + (c ** 2) * (sem_c ** 2)) / (mean_c ** 2)
    if as_percent:
        return (r_hat * 100, np.sqrt(var) * 100)
    else:
        return (r_hat, np.sqrt(var))


def mean_from_nonnull(
    mean_nonnull: Union[float, np.ndarray], frac_nonnull: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Compute mean from non-null mean.

    Args:
        mean_nonnull: Sample mean across non-null
            observations
        frac_nonnull: Fraction of non-null
            observations (must be numeric or of the same shape as mean_nonnull)

    Returns:
        Union[numeric, array_like]: Sample means across all observations
    """
    return np.multiply(frac_nonnull, mean_nonnull)


def var_from_nonnull(
    var_nonnull: Union[float, np.ndarray],
    mean_nonnull: Union[float, np.ndarray],
    frac_nonnull: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute variance from non-null variance and mean.

    Args:
        var_nonnull: Sample variance across
            non-null observations
        mean_nonnull: Sample mean across non-null
            observations (must be of the same shape as mean_nonnull)
        frac_nonnull: Fraction of non-null
            observations (must be numeric or of the same shape as mean_nonnull)

    Returns:
        Union[numeric, array_like]: Sample variance across all observations
    """
    return np.multiply(frac_nonnull, var_nonnull) + frac_nonnull * np.subtract(
        1, frac_nonnull
    ) * (mean_nonnull ** 2)


def estimate_correlation_from_splits(
    split_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Compute metric correlation matrices for all arms from random splits.

    Args:
        split_df: A data frame of the type returned by the .df
            attribute of a Data object with a 'random_split' column containing
            the ids of the splits.

    Returns:
        correlations: A dictionary keyed on arm names
            whose values are pandas DataFrames indexed (both row and column) by
            metric names, represending the correlation matrix of metrics for the
            respective arm.

    For now this works only for data without context / a single context_stratum.
    Requires that the data has a single batch and start_time / end_time pair.

    """
    # serialization is necessary o/w we cannot check for uniqueness
    if "context" in split_df.columns:
        if len(split_df["context"].unique()) > 1:
            raise ValueError(
                "Correlation estimation not supported for contextual data."
            )
    if "batch" in split_df.columns:
        if len(split_df["batch"].unique()) > 1:
            raise ValueError(
                "Correlation estimation only supports data from a single batch."
            )
    if "start_time" in split_df.columns:
        if len(np.unique(split_df[["start_time", "end_time"]], axis=0)) > 2:
            raise ValueError(
                "Correlation estimation only supports a single start_time / "
                + "end_time pair."
            )
    df_grouped = split_df.groupby("arm")
    correlations = {}
    for arm in df_grouped.groups.keys():
        ad = df_grouped.get_group(arm)
        mean_est = ad.pivot(index="random_split", columns="metric_key", values="mean")
        correlations[arm] = mean_est.corr()
    return correlations


def get_covariance_from_correlation(
    correlation: Union[np.ndarray, pd.DataFrame],
    sems: Union[np.ndarray, List[float], pd.Series],
) -> Union[np.ndarray, pd.DataFrame]:
    """Compute covariance matrix from correlation matrix and sems. """
    return correlation * np.outer(sems, sems)


def get_correlation_from_covariance(
    covariance: Union[np.ndarray, pd.DataFrame],
) -> Union[np.ndarray, pd.DataFrame]:
    """Compute correlation matrix from covariance matrix. """
    sems = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sems, sems)


def get_marginal_frac_nonnull(
    fracs_nonnull: Union[pd.Series, np.ndarray, List[float]],
    counts: Union[pd.Series, np.ndarray, List[float]],
) -> float:
    """Compute marginal fraction of non_nulls from those in groups."""
    return np.inner(np.array(counts) / np.sum(counts), fracs_nonnull)


def get_marginal_mean(
    means: Union[np.ndarray, pd.Series, pd.DataFrame],
    counts: Union[np.ndarray, pd.Series, pd.DataFrame],
) -> Union[float, pd.Series]:
    """Compute marginal mean from means in groups."""
    return np.sum(means * (counts / np.sum(counts, axis=0)), axis=0)


def get_marginal_mean_and_variance(
    variances: pd.Series, means: pd.Series, counts: pd.Series, bias: bool = False
) -> Tuple[float, float]:
    """Compute marginal variance from variances and means in groups.

    If bias is False, this assumes that input variances are estimates of the
    population variances (i.e. normalized by `N-1`), o/w it assumes they are
    estimates of the sample variance (i.e. normalized by `N`).
    """
    marginal_mean = get_marginal_mean(means, counts)
    weights = counts / np.sum(counts, axis=0)
    if bias:
        marginal_variance = np.sum(
            weights * (variances + (means - marginal_mean) ** 2), axis=0
        )
    else:
        marginal_variance = np.sum(
            (weights - 1 / (np.sum(counts, axis=0) - 1)) * variances, axis=0
        ) + np.sum(weights * (means - marginal_mean) ** 2, axis=0)
    return marginal_mean, marginal_variance


def agresti_coull_sem(
    n_numer: Union[pd.Series, np.ndarray, int],
    n_denom: Union[pd.Series, np.ndarray, int],
    prior_successes: int = 2,
    prior_failures: int = 2,
) -> Union[np.ndarray, float]:
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


# TODO Migrate add_weighted_status_quo


def benjamini_hochberg(p_values: List[float], alpha: float) -> List[int]:
    """Perform Benjamini-Hochberg correction of a list of p-values

    Args:
        p_values: The list of (uncorrected) p-values.
        alpha: The false discovery rate.

    Returns:
        List[int]: The indices of the "discovered" p-valsues in the input list.

    """
    for pval in p_values:
        if pval <= 0 or pval > 1:
            raise ValueError("p-values outside interval [0, 1]")
    ordered_indcs = sorted(range(len(p_values)), key=lambda i: p_values[i])
    ordered_pvals = np.array([p_values[i] for i in ordered_indcs])
    BHc = np.arange(1, len(p_values) + 1) * alpha / len(p_values)
    return [idx for idx, d in zip(ordered_indcs, ordered_pvals <= BHc) if d]


def relativize_rates_against_mean(
    ps: Union[pd.Series, np.ndarray, List[float]],
    ns: Union[pd.Series, np.ndarray, List[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Relativizes a set of rates to the overall mean rate. That is, the rate
    that serves as the basis is, sum(ps * ns) / sum(ns).

    Args:
        ps: Rates to relativize
        ns: Sample size for each rate

    Returns:
        means: Relative risk (relative to the overall mean)
        sems: Standard error of relative risk
    """
    ps = np.array(ps)
    ns = np.array(ns)
    K = len(ps)
    p = np.sum(ps * ns) / np.sum(ns)
    if p == 0:
        raise ValueError("All supplied rates are zero: cannot relativize.")
    sem = np.sqrt(p * (1 - p) / np.sum(ns))
    rel_means = np.zeros(K)
    rel_sems = np.zeros(K)
    for i in range(K):
        n1 = ns[i]
        other_idx = [idx != i for idx in range(K)]
        n0 = np.sum(ns[other_idx])
        frac1 = n1 / np.sum(ns)
        frac0 = 1 - frac1
        p1 = ps[i]
        p0 = np.average(ps[other_idx], weights=ns[other_idx])
        v1 = p1 * (1 - p1)
        v0 = p0 * (1 - p0)
        cov = frac0 * (v0 / n0 + p0 ** 2) + frac1 * (p0 * p1) - p * p0
        rel_mean, rel_sem = relativize(p1, np.sqrt(v1 / n1), p, sem, cov_means=cov)
        rel_means[i] = rel_mean
        rel_sems[i] = rel_sem
    return rel_means, rel_sems


def ancillary_james_stein(
    means_unbiased: Union[List[float], np.ndarray, pd.Series],
    sems_unbiased: Union[List[float], np.ndarray, pd.Series],
    means_biased: Union[List[float], np.ndarray, pd.Series],
    sems_biased: Union[List[float], np.ndarray, pd.Series],
    weights: Union[List[float], np.ndarray, pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """Takes a set of means and variances and performs empirical bayes shrinkage
    using a James-Stein style estimator which allows for one set of means to be
    biased.

    This estimates the (fixed) bias as the difference between the biased and
    unbiased means (weighted by the weights argument).
    Args:
        means_unbiased: Unbiased means
        sems_unbiased: SEMs for
            unbiased means
        means_biased: Biased means
        sems_biased: SEMs for biased means
        ns: weights for calculation of
            bias term

    Returns:
        means: Empirical bayes estimate of underlying mean
        sems: Standard error
    """
    m_unbiased = np.array(means_unbiased)
    s_unbiased = np.array(sems_unbiased)
    m_biased = np.array(means_biased)
    s_biased = np.array(sems_biased)
    bias = np.average(m_biased - m_unbiased, weights=weights)
    alpha = s_unbiased ** 2 / (bias ** 2 + s_unbiased ** 2 + s_biased ** 2)
    eb_est = alpha * m_biased + np.subtract(1, alpha) * m_unbiased
    eb_var = (alpha ** 2) * s_biased ** 2 + (
        np.subtract(1, alpha) ** 2
    ) * s_unbiased ** 2
    return eb_est, np.sqrt(eb_var)


def additive_smoothing(arms, data, prior_successes=2, prior_failures=2):
    df = data.df
    successes = df["mean"] * df["n"]
    failures = df["n"] - successes
    df["mean"] = (successes + prior_successes) / (
        successes + prior_successes + failures + prior_failures
    )
    df["sem"] = agresti_coull_sem(
        n_numer=successes,
        n_denom=successes + failures,
        prior_successes=prior_successes,
        prior_failures=prior_failures,
    )
    df["n"] = df["n"] + prior_successes + prior_failures
    prior_mean = prior_successes / (prior_successes + prior_failures)
    prior_sem = np.sqrt(
        prior_mean * (1 - prior_mean) / (prior_successes + prior_failures)
    )
    new_dfs = []
    dfg = df.groupby("metric_key")
    for metric in dfg.groups.keys():
        metric_df = dfg.get_group(metric)
        arms_not_in_data = [arm for arm in arms if arm not in metric_df["arm"]]
        if len(arms_not_in_data) > 0:
            new_dfs.append(
                pd.DataFrame(
                    {
                        "metric_key": metric,
                        "arm": arms_not_in_data,
                        "batch": -1,
                        "mean": prior_mean,
                        "sem": prior_sem,
                        "n": prior_successes + prior_failures,
                    }
                )
            )
        else:
            new_dfs.append(pd.DataFrame())
    new_df = pd.concat(new_dfs + [df])
    return Data(new_df)


def estimate_aggregated_moments_normal(
    means: Union[List[float], np.ndarray, pd.Series],
    sems: Union[List[float], np.ndarray, pd.Series],
    ns: Union[List[float], np.ndarray, pd.Series],
) -> Tuple[float, float]:
    """
    This estimates the mean and standard error of a set of data assuming
    individual observations are distributed normal with provided means, variances
    and sample size. That is, this function provides the mean and variance that
    would have been found when taking a simple sample mean if the distribution of
    the underlying data were normally distributed for each observation.

    Args:
        means: Means
        sems: SEMs for means
        ns: Sample size

    Returns:
        mean: Aggregated mean (assuming normality of observations)
        sems: Aggregated SEM (assuming normality of observations)
    """
    # Cast all to numpy to please typechecker
    means = np.array(means)
    sems = np.array(sems)
    ns = np.array(ns)

    K = len(means)
    probs = ns / np.sum(ns)
    vrs = ns * np.array(sems) ** 2
    overall_mn = np.average(means, weights=probs)
    E_vr = np.average(vrs, weights=probs ** 2)
    vr_E = np.average((means - overall_mn) ** 2, weights=probs) * K / (K - 1)
    overall_vr = E_vr + vr_E
    overall_sem = np.sqrt(overall_vr / np.sum(ns))
    return overall_mn, overall_sem

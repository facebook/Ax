#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import numpy.typing as npt


def relativize(
    means_t: npt.NDArray | list[float] | float,
    sems_t: npt.NDArray | list[float] | float,
    mean_c: npt.NDArray | float,
    sem_c: npt.NDArray | float,
    bias_correction: bool = True,
    cov_means: npt.NDArray | list[float] | float = 0.0,
    as_percent: bool = False,
    control_as_constant: bool = False,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Ratio estimator based on the delta method.

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
        mean_c: Sample mean (control). Can be a scalar representing a single control
            group, or an array of control means (one per test mean) for cases where
            different test observations have different control values (e.g., multi-trial
            experiments with per-trial status quos).
        sem_c: Sample standard error of the mean (control). Can be a scalar or an array
            matching the shape of mean_c.
        bias_correction: Whether to apply bias correction when computing relativized
            metric values. Uses a second-order Taylor expansion for approximating
            the means and standard errors of the ratios.
        cov_means: Sample covariance between test and control
        as_percent: If true, return results in percent (* 100)
        control_as_constant: If true, control is treated as a constant.
            bias_correction, sem_c, and cov_means are ignored when this is true.


    Returns:
        rel_hat: Inferred means of the sampling distribution of
            the relative change `(mean_t - mean_c) / abs(mean_c)`
        sem_hat: Inferred standard deviation of the sampling
            distribution of rel_hat -- i.e. the standard error.

    """
    # if mean_c is too small, bail
    epsilon = 1e-10
    if np.any(np.abs(mean_c) < epsilon):
        raise ValueError(
            "mean_control ({} +/- {}) is smaller than 1 in 10 billion, "
            "which is too small to reliably analyze ratios using the delta "
            "method. This usually occurs because winsorization has truncated "
            "all values down to zero. Try using a delta type that applies "
            "no winsorization.".format(mean_c, sem_c)
        )
    m_t = np.array(means_t)
    s_t = np.array(sems_t)
    cov_t = np.array(cov_means)
    abs_mean_c = np.abs(mean_c)
    r_hat = (m_t - mean_c) / abs_mean_c

    if control_as_constant:
        var = (s_t / abs_mean_c) ** 2
    else:
        c = m_t / mean_c
        if bias_correction and not np.all(np.isnan(sem_c)):
            r_hat = r_hat - m_t * sem_c**2 / abs_mean_c**3

        # If everything's the same, then set r_hat to zero
        same = (m_t == mean_c) & (s_t == sem_c)
        r_hat = ~same * r_hat
        var = ((s_t**2) - 2 * c * cov_t + (c**2) * (sem_c**2)) / (mean_c**2)
    if as_percent:
        return (r_hat * 100, np.sqrt(var) * 100)
    else:
        return (r_hat, np.sqrt(var))


def unrelativize(
    means_t: npt.NDArray | list[float] | float,
    sems_t: npt.NDArray | list[float] | float,
    mean_c: float,
    sem_c: float,
    bias_correction: bool = True,
    cov_means: npt.NDArray | list[float] | float = 0.0,
    as_percent: bool = False,
    control_as_constant: bool = False,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Reverse operation of ax.utils.stats.math_utils.relativize.

    Args:
        means_t: Relativized sample means (test) to be unrelativized
        sems_t: Relativized sample SEM of the means (test) to be unrelativized
        mean_c: Unrelativized control mean
        sem_c: Unrelativized control SEM of the mean
        bias_correction: if `means_t` and `sems_t` are obtained with
                         `bias_correction=True` in ax.utils.stats.math_utils.relativize
        cov_means: Sample covariance between the **unrelativized** test and control
        as_percent: If true, assuming `means_t` and `sems_t` are percentages
                    (i.e., 1 means 1%).
        control_as_constant: If true, control is treated as a constant.
            bias_correction, sem_c, and cov_means are ignored when this is true.

    Returns:
        m_t: Inferred sample (test) means in the unrelativized scale
        s_t: Inferred SEM of sample (test) means in the unrelativized scale
    """
    means_t = np.array(means_t, dtype=float)
    sems_t = np.array(sems_t, dtype=float)

    if as_percent:
        means_t = means_t / 100
        sems_t = sems_t / 100

    abs_mean_c = np.abs(mean_c)
    m_t = means_t * abs_mean_c + mean_c

    if control_as_constant:
        s_t = sems_t * abs_mean_c
    else:
        if bias_correction:
            m_t = m_t / (1 - (sem_c / abs_mean_c) ** 2)

        var = sems_t**2
        c = m_t / mean_c
        s_t2 = var * (mean_c**2) + 2 * c * cov_means - (c**2) * (sem_c**2)

        # This is only positive when sems_t > sem_c * mean_c * (means_t + 1)
        # If above condition cannot be guaranteed, use control_as_constant = True
        s_t = np.sqrt(s_t2.clip(min=0.0))

    # if means_t is 0.0 exactly, return control mean and sem directly
    if np.isscalar(means_t):
        if means_t == 0.0:
            m_t = mean_c
            s_t = sem_c
    else:
        m_t = np.array(m_t)
        s_t = np.array(s_t)
        m_t[means_t == 0.0] = mean_c
        s_t[means_t == 0.0] = sem_c

    # pyre-fixme[7]: Expected `Tuple[ndarray[typing.Any, typing.Any],
    #  ndarray[typing.Any, typing.Any]]` but got `Tuple[Union[ndarray[typing.Any,
    #  dtype[typing.Any]], float], Union[ndarray[typing.Any, dtype[typing.Any]],
    #  float]]`.
    return m_t, s_t

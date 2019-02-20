#!/usr/bin/env python3

import math
import sys
from typing import Tuple

import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats


EPSILON = np.sqrt(sys.float_info.epsilon)


class MathAssumptionException(Exception):
    pass


def check_significance_level(alpha: float) -> None:
    """
    Ensure that significance level is bounded between 0 and 1.
    """
    if alpha < 0 - EPSILON or alpha > 1 + EPSILON:
        raise MathAssumptionException("alpha must be in [0, 1]")
    return


def check_power(power: float, alpha: float) -> None:
    """
    Ensure that power is bounded between alpha and 1.
    """
    if power < alpha - EPSILON or power > 1 + EPSILON:
        raise MathAssumptionException("power must be in [alpha, 1]")
    return


def check_sample_size(sample_size: int) -> None:
    """
    Ensure that sample size is greater than or equal to 2.
    """
    if sample_size < 2 - EPSILON:
        raise MathAssumptionException("sample_size must be in [2, Inf)")
    return


def check_variance(variance: float) -> None:
    """
    Ensure that variance is positive.
    """
    if variance < 1e-16:
        raise MathAssumptionException("variance must be in [1e-16, Inf)")
    return


def power(n: int, delta: float, variance: float, alpha: float = 0.05) -> float:
    """
    Calculate the power of a proposed experimental design, which is
    described by:

      * n: The sample size, which is assumed to be the same in the
          treatment and control groups. We enforce the assumption that
          n >= 2.
      * delta: The difference in means between the treatment and control
          groups
      * variance: The variance of the metric being tested, which is assumed
          to be the same in the treatment and control groups. We enforce
          the assumption that variance >= 1e-16.
      * alpha: The significance level at which the null will be rejected,
          which is usually p = 0.05. We enforce the assumption that
          0 <= alpha <= 1.
    """
    # Validate inputs.
    check_sample_size(n)
    check_variance(variance)
    check_significance_level(alpha)

    # Determine the standard error of the sampling distribution of empirical
    # differences in means.
    se_delta = np.sqrt(variance / n + variance / n)

    # Determine the critical values of the difference in means test under
    # the null hypothesis. The null hypothesis assumes that the difference
    # in means has a normal distribution with mean = 0 and standard
    # deviation = se_delta. The critical values are the lower and upper
    # quantiles of this sampling distribution and they enclose a total
    # probability equal to the specified significance level.
    # pyre-ignore [16]: Module `scipy.stats` has no attribute `norm`.
    critical_value_lower = stats.norm.ppf(alpha / 2.0, 0.0, se_delta)
    # pyre-ignore [16]: Module `scipy.stats` has no attribute `norm`.
    critical_value_upper = stats.norm.isf(alpha / 2.0, 0.0, se_delta)

    # Determine the probability of rejecting the null under the alternative
    # hypothesis, which assumes that the difference in means has a normal
    # distribution with mean = delta and standard deviation = se_delta. The
    # probabilities of observing a value below critical_value_lower and of
    # observing a value above critical_value_upper are generally not
    # symmetric, unlike the critical values.
    # pyre-ignore [16]: Module `scipy.stats` has no attribute `norm`.
    p_lower = stats.norm.cdf(critical_value_lower, delta, se_delta)
    # pyre-ignore [16]: Module `scipy.stats` has no attribute `norm`.
    p_upper = stats.norm.sf(critical_value_upper, delta, se_delta)

    # Determine the power, which is the total probability under the alternative
    # hypothesis of observing a difference in means inside the rejection
    # region for the hypothesis test.
    calculated_power = p_lower + p_upper

    # Validate the power before returning it.
    check_power(calculated_power, alpha)

    # Return the final value when it is valid.
    return calculated_power


def sample_size(
    target_power: float, delta: float, variance: float, alpha: float = 0.05
) -> Tuple[int, float]:
    """
    Calculate the minimum sample size for an experimental design that
    provides a target level of power under the assumption that the
    treatment and control groups will both be of the same size. A proposed
    experimental design is described by:

      * target_power: The desired power of the experiment design. We
          enforce the assumption that 0 <= power <= 1.
      * delta: The difference in means between the treatment and control
          groups.
      * variance: The variance of the metric being tested, which is assumed
          to be the same in the treatment and control groups. We enforce
          the assumption that variance >= 1e-16.
      * alpha: The significance level at which the null will be rejected,
          which is usually p = 0.05. We enforce the assumption that
          0 <= alpha <= 1.
    """
    # Validate inputs.
    check_power(target_power, alpha)
    check_variance(variance)
    check_significance_level(alpha)

    # Construct a closure that calculates the signed gap between the target
    # power and the achieved power given a proposed sample size. The closure
    # ensures that all of the other parameters of the experiment design are
    # held fixed while the sample size varies.
    def power_gap(n: int) -> float:
        achieved_power = power(n, delta, variance, alpha)
        return target_power - achieved_power

    # Use a univariate bisection technique to find the required sample size to
    # achieve the target level of power, which we assume lies between 2 and
    # 2,000,000,000,000. If the world population goes above 2 trillion, this
    # number might need to be revisited.
    lower_bound = 2e0
    upper_bound = 2e12
    x_star = optimize.bisect(power_gap, lower_bound, upper_bound)

    # Because the result of this optimization will typically lie between two
    # integers, we round up to the nearest integer and convert to an integer
    # type. This ensures that the power is never below the target value.
    n = math.ceil(x_star)

    # Determine the exact power achieved by using the integer-valued sample
    # size.
    achieved_power = power(n, delta, variance, alpha)

    # Return a tuple containing the required sample size and the exact power
    # achieved at that sample size.
    return n, achieved_power


def effect_size(
    target_power: float, n: int, variance: float, alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Calculate the smallest effect size that a proposed experimental design
    would detect at a requested level of power under the assumption that the
    treatment and control groups will both be of the same size. A proposed
    experimental design is described by:

      * target_power: The desired power of the experiment design. We enforce
          the assumption that 0 <= power <= 1.
      * n: The number of units in the test group or the control group, under
          the assumption that both groups have equal sample sizes.
      * variance: The variance of the metric being tested, which is assumed
          to be the same in the treatment and control groups. We enforce
          the assumption that variance >= 1e-16.
      * alpha: The significance level at which the null will be rejected,
          which is usually p = 0.05. We enforce the assumption that
          0 <= alpha <= 1.
    """
    # Validate inputs.
    check_power(target_power, alpha)
    check_sample_size(n)
    check_variance(variance)
    check_significance_level(alpha)

    # Construct a closure that calculates the signed gap between the target
    # power and the achieved power given a proposed sample size. The closure
    # ensures that all of the other parameters of the experiment design are
    # held fixed while the sample size varies.
    def power_gap(delta: float) -> float:
        achieved_power = power(n, delta, variance, alpha)
        return target_power - achieved_power

    # Use a univariate bisection technique to find the minimal effect size
    # that can achieve the target level of power. We assume this effect size
    # lives inside the interval [1e-6 * variance, 1e6 * variance], which might
    # not hold for a metric that is extremely small or extremely large relative
    # to its variance.
    lower_bound = min(1e-6 * variance, 1)
    upper_bound = 1e6 * variance
    delta, results = optimize.bisect(
        f=power_gap, a=lower_bound, b=upper_bound, full_output=True, maxiter=1000
    )

    # Determine the power achieved when measuring this minimal effect size.
    achieved_power = power(n, delta, variance, alpha)

    # Return a tuple containing the minimal effect size and the achieved power
    # when trying to detect that effect size.
    return delta, achieved_power

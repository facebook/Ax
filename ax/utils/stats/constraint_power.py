#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from functools import partial

import numpy as np
import scipy.optimize as optimize
from ax.exceptions.core import AxError
from ax.utils.common.logger import get_logger
from scipy.stats import norm


logger = get_logger("constraint_power")


def constraint_power(
    n, mu_t, mu_c, v_t, v_c, constraint, constraint_direction, n_sim, alpha, prob=0.5
):
    """Simulate to determine power for a non-inferiority test against a constraint
        given a set of sufficient statistics.

    Arguments:
        n: Total sample size to use in power calculation
        mu_t: The assumed mean in treatment group.
        mu_c: The assumed mean in control group.
        v_t: The assumed variance in the treatment group (NOT the
            variance of the estimate of the mean).
        v_c: The assumed variance in the control group (NOT the
            variance of the estimate of the mean).
        constraint: The relative constraint to test against. (e.g.
            0.05 means testing that the treatment group is at least 5% greater
            than the status_quo).
        constraint_direction: One of 'GT' or 'LT'. The direction the
            constraint applies. GT implies that the treatment group must be
            greater than the status quo, LT vice versa.
        n_sim: The number of simulations to run to estimate power.
        alpha: The desired significance level.
        prob: The probability of assignment to treatment (1 - prob is
            the assignment probability to control).

    Returns:
        power: The realized power for the given parameters.
    """
    n_t = prob * n
    n_c = n - n_t
    m_t = np.random.normal(loc=mu_t, scale=np.sqrt(v_t / n_t), size=n_sim)
    sem_t = np.random.normal(
        loc=np.sqrt(v_t), scale=np.sqrt(v_t / (2 * n_t)), size=n_sim
    ) / np.sqrt(n_t)
    m_c = np.random.normal(
        loc=(1 + constraint) * mu_c, scale=np.sqrt(v_c / n_c), size=n_sim
    )
    sem_c = np.random.normal(
        loc=np.sqrt(v_c), scale=np.sqrt(v_c / (2 * n_c)), size=n_sim
    ) / np.sqrt(n_c)
    effect = (m_t - m_c) / np.sqrt(sem_c ** 2 + sem_t ** 2)
    if constraint_direction == "GT":
        return np.mean(effect > norm.ppf(1 - alpha))
    else:
        return np.mean(effect < norm.ppf(alpha))


def constraint_sample_size(
    mu_t,
    mu_c,
    v_t,
    v_c,
    constraint=0.0,
    constraint_direction="GT",
    n_sim=5000,
    alpha=0.05,
    power=0.9,
    verbose=True,
):
    """Determine the necessary sample size to detect a change relative to a
    constraint using non-inferiority testing.

    Arguments:
        mu_t: The assumed mean in the treatment group.
        mu_c: The assumed mean in the control group.
        v_t: The assumed variance in the treatment group (NOT the
            variance of the estimate of the mean).
        v_c: The assumed variance in the control group (NOT the
            variance of the estimate of the mean).
        constraint: The relative constraint to test against. (e.g.
            0.05 means testing that the treatment group is at least 5% greater
            than the status_quo).
        constraint_direction: One of 'GT' or 'LT'. The direction the
            constraint applies. GT implies that the treatment group must be
            greater than the status quo, LT vice versa.
        n_sim: The number of simulations to run to estimate power.
        alpha: The desired significance level.
        power: The desired power.
        verbose: If True log results of all individual sample
            size calculcations.

    Returns:
        n: The necessary sample size for the given arguments. Returns
            inf if required size is greater than 1e10.
    """
    power_fn = partial(
        constraint_power,
        mu_t=mu_t,
        mu_c=mu_c,
        v_t=v_t,
        v_c=v_c,
        constraint=constraint,
        constraint_direction=constraint_direction,
        n_sim=n_sim,
        alpha=alpha,
    )

    def power_gap(n):
        return power_fn(n=n) - power

    lower_bound = 2e0
    upper_bound = 1e12
    try:
        n_star = optimize.bisect(power_gap, lower_bound, upper_bound)
    except ValueError:
        if power_fn(n=lower_bound) > power:
            n_star = lower_bound
        elif power_fn(n=upper_bound) < power:
            n_star = np.inf
        else:
            raise AxError("Bisection failed to determine sample size.")
    n = int(n_star) + 1 if n_star < np.inf else np.inf
    if verbose:
        if n_star < np.inf:
            msg = "With sample size of {}, ".format(n) + "realized power is: {}".format(
                power_fn(n=n)
            )
        else:
            msg = "Desired power not achievable with n < {}.".format(int(upper_bound))
        logger.info(msg)
    return n


def feasible_constraint(
    mu_t,
    mu_c,
    v_t,
    v_c,
    n,
    constraint_direction="GT",
    n_sim=5000,
    alpha=0.05,
    power=0.9,
    verbose=True,
):
    """Determine the constraint for which a given sample size would have the
    specified level of power to detect a violation using non-inferiority testing.

    Arguments:
        n: Sample size to use in power calculation.
        mu_t: The assumed mean in the treatment group.
        mu_c: The assumed mean in the control group.
        v_t: The assumed variance in the treatment group (NOT the
            variance of the estimate of the mean).
        v_c: The assumed variance in the control group (NOT the
            variance of the estimate of the mean).
        constraint: The relative constraint to test against. (e.g.
            0.05 means testing that the treatment group is at least 5% greater
            than the status_quo)
        constraint_direction: One of 'GT' or 'LT'. The direction the
            constraint applies. GT implies that the treatment group must be
            greater than the status quo, LT vice versa.
        n_sim: The number of simulations to run to estimate power.
        alpha: The desired significance level.
        verbose: If True log results of all individual sample
            size calculcations.

    Returns:
        numeric: The necessary sample size for the given arguments.
    """
    power_fn = partial(
        constraint_power,
        mu_t=mu_t,
        mu_c=mu_c,
        v_t=v_t,
        v_c=v_c,
        n=n,
        constraint_direction=constraint_direction,
        n_sim=n_sim,
        alpha=alpha,
    )

    def power_gap(constraint):
        return power_fn(constraint=constraint) - power

    lower_bound = -1.0
    upper_bound = 1.0
    try:
        constraint_star = optimize.bisect(power_gap, lower_bound, upper_bound)
    except ValueError:
        if (
            power_fn(constraint=lower_bound) > power
            and power_fn(constraint=upper_bound) > power
        ):
            constraint_star = lower_bound
        elif (
            power_fn(constraint=lower_bound) < power
            and power_fn(constraint=upper_bound) < power
        ):
            constraint_star = np.nan
        else:
            raise AxError("Bisection failed to determine sample size.")
    if verbose:
        if not np.isnan(constraint_star):
            realized_power = power_fn(constraint=constraint_star)
            msg = "With constraint of {}.3f%, realized power is: {}.3f".format(
                constraint_star * 100, realized_power
            )
        else:
            msg = "No constraint is feasible in [{}.3f%, {}.3f%]".format(
                lower_bound * 100, upper_bound * 100
            )
        logger.info(msg)
    return constraint_star

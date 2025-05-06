#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
import pandas as pd
import scipy

from ax.core.data import Data
from ax.utils.stats.statstools import relativize
from scipy.stats import norm


def check_experiment_effects_per_metric(
    data: Data,
    objective_names: set[str] | None = None,
    no_effect_alpha: float = 0.05,
) -> pd.DataFrame:
    """Checks whether an experiment has effects per metric separately.
    The results will include the p-value and whether there was an effect
    for each of the available metrics in the data. `check_experiment_effects`
    below checks whether there is any effect overall across all metrics.

    This checks whether a randomization test can show that there are any
    effects whatsoever. This test is performed independently on each metric
    and it is based on the Welch's test for testing whether the means across
    groups are identical assuming unequal variances across groups.

    Args:
        data: The metric data on which to perform the test of no effect.
        objective_names: The names of the objective metrics used in this analysis
            used for labeling the is_objective column in the output frame.
            If None, omit the is_objective column.
        no_effect_alpha: The confidence level at which to reject the
            sharp null of no experimental effects.

    Returns:
        A dataframe containing the results of the test of no effect per metric.

    """
    df = data.df
    df_grouped = df.groupby(["metric_name", "trial_index"])
    cols = ["metric_name", "trial_index", "p_value", "has_effect"]

    if objective_names is not None:
        cols.append("is_objective")

    df_tone = pd.DataFrame(columns=cols)

    for metric_name, trial_index in df_grouped.groups.keys():
        dfm = df_grouped.get_group((metric_name, trial_index))

        p_value, f_stat = no_effect_test_welch(
            means=list(dfm["mean"].values),
            sems=list(dfm["sem"].values),
            ns=list(dfm["n"]),
        )
        has_effect = bool(p_value < no_effect_alpha)

        d = {
            "trial_index": trial_index,
            "metric_name": metric_name,
            "p_value": p_value,
            "has_effect": has_effect,
        }
        if objective_names is not None:
            d["is_objective"] = metric_name in objective_names

        df_tone = pd.concat([df_tone, pd.DataFrame([d])], ignore_index=True)

    df_tone["trial_index"] = df_tone["trial_index"].astype(int)
    df_tone["metric_name"] = df_tone["metric_name"].astype("string")
    df_tone["has_effect"] = df_tone["has_effect"].astype(bool)

    if "is_objective" in df_tone:
        df_tone["is_objective"] = df_tone["is_objective"].astype(bool)

    return df_tone


def check_experiment_effects(
    data: Data,
    objective_names: set[str],
    status_quo_name: str = "status_quo",
    no_effect_alpha: float = 0.05,
    ci_alpha: float = 0.05,
) -> tuple[bool, list[str], pd.DataFrame]:
    """Checks whether an experiment did anything.

    Basic version without support for contexts. This checks whether a
        randomization test can show that there are any effects whatsoever.
        This test is performed independently on each metric, and the minimum
        Bonferroni adjusted p-value is returned. Additionally, returns a data
        frame showing the minimum and maximum effects consistent with the
        observed data.

    Args:
        data: The data on which to perform the check.
        objective_name: The name of the objective metric used in this analysis.
        no_effect_alpha: The confidence level at which to reject the
            sharp null of no experimental effects.
        ci_alpha: The confidence level to use to form a joint confidence
            interval for all effects.
        status_quo_name: The name of the status quo arm.

    Returns:
        effective: True if the null of no treatment effects can be rejected.
        ineffective_on_objectives: List of objectives on which the
            null of no treatment effects can be rejected.
        bounds_df: The minimum and maximum bounds on possible effects.

    """
    df = data.df
    df_grouped = df.groupby("metric_name")
    K = len(df_grouped)
    ps = []
    fx_bounds = []

    # list of objectives on which there were no effects
    # if list is non-empty, we'll show a warning
    ineffective_on_objectives = []

    for metric in df_grouped.groups.keys():
        dfm = df_grouped.get_group(metric)
        p, z, null_z = ri_test_of_no_effect(
            dfm["mean"],
            dfm["sem"],
            dfm["n"],
            dfm["arm_name"],
            status_quo_name=status_quo_name,
        )
        min_fx, max_fx = estimate_effect_bounds(
            dfm["mean"],
            dfm["sem"],
            dfm["arm_name"],
            status_quo_name=status_quo_name,
            alpha=ci_alpha,
        )
        fx_bounds.append((metric, min_fx, max_fx, p))
        ps.append(p * K)

        if metric in objective_names and p * K > no_effect_alpha:
            ineffective_on_objectives.append(metric)

    effective = np.min(ps) < no_effect_alpha
    bounds_df = pd.DataFrame(fx_bounds, columns=["metric_name", "min", "max", "p"])
    bounds_df.sort_values(by="p", inplace=True)
    return effective, ineffective_on_objectives, bounds_df


# pyre-fixme[3]: Return type must be annotated.
def ri_test_of_no_effect(
    means: list[float],
    sems: list[float],
    ns: list[int],
    names: list[str],
    status_quo_name: str,
    M: int = 10000,
):
    # pyre-fixme[9]: means has type `List[float]`; used as `ndarray`.
    means = np.array(means)
    # pyre-fixme[9]: sems has type `List[float]`; used as `ndarray`.
    sems = np.array(sems)
    # pyre-fixme[9]: ns has type `List[int]`; used as `ndarray`.
    ns = np.array(ns)
    # pyre-fixme[9]: names has type `List[str]`; used as `ndarray`.
    names = np.array(names)
    if len(names) != len(means):
        raise ValueError("Length of means and names must be equal.")
    K = len(means)
    # pyre-fixme[58]: `/` is not supported for operand types `List[int]` and `Any`.
    ps = ns / np.sum(ns)
    vars = np.power(sems, 2) * np.array(ns)
    mn = np.average(means, weights=ps)
    E_vr = np.average(vars, weights=ps**2)
    # pyre-fixme[58]: `-` is not supported for operand types `List[float]` and `Any`.
    vr_E = np.average((means - mn) ** 2, weights=ps) * K / (K - 1)
    vr = E_vr + vr_E
    z_stats = []
    p_value = 0.0
    actual_z = estimate_largest_z(means, sems, names, status_quo_name=status_quo_name)
    for _ in range(M):
        # pyre-fixme[6]: For 1st argument expected `bool` but got `List[int]`.
        b_mns = np.random.normal(loc=mn, scale=np.sqrt(vr / ns), size=K)
        # pyre-fixme[6]: For 1st argument expected `bool` but got `List[int]`.
        b_sems = np.random.normal(loc=np.sqrt(vr), scale=np.sqrt(vr / (2 * ns)), size=K)
        b_sems /= np.sqrt(ns)
        ri_arms = np.random.choice(names, K, replace=False)
        new_z = estimate_largest_z(
            b_mns, b_sems, ri_arms, status_quo_name=status_quo_name
        )
        z_stats.append(new_z)
        if new_z >= actual_z:
            p_value += 1.0 / M
    return p_value, actual_z, z_stats


def no_effect_test_welch(
    means: list[float],
    sems: list[float],
    ns: list[int],
) -> tuple[float, float]:
    r"""
    Welch's F test of no effect for the means of the arms. It tests whether the means
    across arms are identical assuming unequal variances across arms.

    The Welch's F test statistic equals:
        W* = \sum w_i (mean_i-mu_hat)^2 /(K-1) / [(1 + 2*(K-2)/(K^2-1)) \sum h_i],
    where
        mean_i, sem_i, n_i are observed means, standard errors and sample sizes
        across K groups,
        var_i = sem_i^2 * n_i is the variance of each group,
        w_i = n_i / var_i, W = sum_i w_i, mu_hat = \sum w_i * mean_i / W,
        h_i = (1-w_i/W)^2 / (n_i-1) and f = (K^2-1) / (3*\sum_i h_i).
    The value of W* is compared to an F distribution with K-1 and f degrees of freedom.

    Args:
        means: The means of the arms.
        sems: The standard errors of the arms.
        ns: The number of users per arm.

    Returns: A tuple containing
        - the p-value and
        - the test-statistic value.
    """
    means_arr = np.array(means)
    sems_arr = np.array(sems)
    ns_arr = np.array(ns)

    K = len(means_arr)

    variances = np.multiply(sems_arr**2, ns_arr)
    ws = np.divide(ns_arr, variances)

    W = np.sum(ws)

    overall_mean = np.dot(means_arr, ws) / W
    # mean square error between groups: the difference between the
    # group means and the overall weighted mean
    bg = np.dot((means_arr - overall_mean) ** 2, ws) / (K - 1)

    hs = np.divide((1.0 - ws / W) ** 2, ns_arr - 1)
    H = np.sum(hs)
    wg = 1 + 2 * (K - 2) * H / (K**2 - 1)
    dfd = (K**2 - 1) / (3 * H)

    f_stat = bg / wg
    # pyre-ignore
    p_value = 1 - scipy.stats.f.cdf(f_stat, dfn=K - 1, dfd=dfd)

    return p_value, f_stat


# pyre-fixme[3]: Return type must be annotated.
def estimate_effect_bounds(
    means: list[float],
    sems: list[float],
    names: list[str],
    status_quo_name: str | None,
    alpha: float,
):
    # pyre-fixme[9]: means has type `List[float]`; used as `ndarray[typing.Any,
    #  dtype[typing.Any]]`.
    means = np.asarray(means)
    # pyre-fixme[9]: sems has type `List[float]`; used as `ndarray[typing.Any,
    #  dtype[typing.Any]]`.
    sems = np.asarray(sems)
    # pyre-fixme[9]: names has type `List[str]`; used as `ndarray[typing.Any,
    #  dtype[typing.Any]]`.
    names = np.asarray(names)
    if status_quo_name is not None:
        # pyre-fixme[16]: `float` has no attribute `item`.
        m_c = means[names == status_quo_name].item()
        sem_c = sems[names == status_quo_name].item()
        means_t = means[names != status_quo_name]
        sems_t = sems[names != status_quo_name]
        fx, fx_sems = relativize(
            means_t=means_t, sems_t=sems_t, mean_c=m_c, sem_c=sem_c
        )
    else:
        fx, fx_sems = means, sems
    z = norm.ppf(1 - alpha / 2)
    # pyre-fixme[58]: `-` is not supported for operand types
    #  `Union[np.ndarray[typing.Any, typing.Any], typing.List[float]]` and `float`.
    # pyre-fixme[58]: `*` is not supported for operand types `float` and
    #  `Union[np.ndarray[typing.Any, typing.Any], typing.List[float]]`.
    min_fx = fx - z * fx_sems
    # pyre-fixme[58]: `+` is not supported for operand types
    #  `Union[np.ndarray[typing.Any, typing.Any], typing.List[float]]` and `float`.
    # pyre-fixme[58]: `*` is not supported for operand types `float` and
    #  `Union[np.ndarray[typing.Any, typing.Any], typing.List[float]]`.
    max_fx = fx + z * fx_sems
    return np.min(min_fx), np.max(max_fx)


# pyre-fixme[3]: Return type must be annotated.
def estimate_largest_z(
    means: list[float], sems: list[float], names: list[str], status_quo_name: str
):
    mn_sq = [mn for i, mn in enumerate(means) if names[i] == status_quo_name][0]
    sem_sq = [sm for i, sm in enumerate(sems) if names[i] == status_quo_name][0]
    arm_mns = [mn for i, mn in enumerate(means) if names[i] != status_quo_name]
    arm_sems = [sm for i, sm in enumerate(sems) if names[i] != status_quo_name]
    effects = [mn - mn_sq for mn in arm_mns]
    effect_sems = [np.sqrt(sem**2 + sem_sq**2) for sem in arm_sems]
    return np.max(np.abs(effects) / effect_sems)

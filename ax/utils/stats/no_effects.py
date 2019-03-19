#!/usr/bin/env python3

from typing import List, Optional

import numpy as np
from ax.utils.stats.statstools import relativize
from scipy.stats import norm


def estimate_largest_z(
    means: List[float], sems: List[float], names: List[str], status_quo_name: str
):
    mn_sq = [mn for i, mn in enumerate(means) if names[i] == status_quo_name][0]
    sem_sq = [sm for i, sm in enumerate(sems) if names[i] == status_quo_name][0]
    arm_mns = [mn for i, mn in enumerate(means) if names[i] != status_quo_name]
    arm_sems = [sm for i, sm in enumerate(sems) if names[i] != status_quo_name]
    effects = [mn - mn_sq for mn in arm_mns]
    effect_sems = [np.sqrt(sem ** 2 + sem_sq ** 2) for sem in arm_sems]
    return np.max(np.abs(effects) / effect_sems)


def ri_test_of_no_effect(
    means: List[float],
    sems: List[float],
    ns: List[int],
    names: List[str],
    status_quo_name: str,
    M: int = 10000,
):
    means = np.array(means)
    sems = np.array(sems)
    ns = np.array(ns)
    names = np.array(names)
    if len(names) != len(means):
        raise ValueError("Length of means and names must be equal.")
    K = len(means)
    ps = ns / np.sum(ns)
    vars = np.power(sems, 2) * np.array(ns)
    mn = np.average(means, weights=ps)
    E_vr = np.average(vars, weights=ps ** 2)
    vr_E = np.average((means - mn) ** 2, weights=ps) * K / (K - 1)
    vr = E_vr + vr_E
    z_stats = []
    p_value = 0.0
    actual_z = estimate_largest_z(means, sems, names, status_quo_name=status_quo_name)
    for _ in range(M):
        b_mns = np.random.normal(loc=mn, scale=np.sqrt(vr / ns), size=K)
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


def estimate_effect_bounds(
    means: List[float],
    sems: List[float],
    names: List[str],
    status_quo_name: Optional[str],
    alpha: float,
):
    means = np.array(means)
    sems = np.array(sems)
    names = np.array(names)
    if status_quo_name is not None:
        m_c = np.asscalar(means[names == status_quo_name])
        sem_c = np.asscalar(sems[names == status_quo_name])
        means_t = means[names != status_quo_name]
        sems_t = sems[names != status_quo_name]
        fx, fx_sems = relativize(
            means_t=means_t, sems_t=sems_t, mean_c=m_c, sem_c=sem_c
        )
    else:
        fx, fx_sems = means, sems
    z = norm.ppf(1 - alpha / 2)
    min_fx = fx - z * fx_sems
    max_fx = fx + z * fx_sems
    return np.min(min_fx), np.max(max_fx)

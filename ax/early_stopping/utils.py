#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger

import pandas as pd
from ax.core.experiment import Experiment
from ax.core.map_data import MAP_KEY, MapData
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UnsupportedError
from ax.utils.common.logger import get_logger
from pyre_extensions import assert_is_instance

logger: Logger = get_logger(__name__)


def _is_worse(a: float, b: float, minimize: bool) -> bool:
    """Determine if value `a` is worse than value `b` based on optimization direction.

    Args:
        a: The first value to compare.
        b: The second value (threshold) to compare against.
        minimize: If True, use minimization logic (a is worse if a > b).
            If False, use maximization logic (a is worse if a < b).

    Returns:
        True if `a` is worse than `b` according to the optimization direction,
        False otherwise.
    """
    return a > b if minimize else a < b


def _interval_boundary(
    progression: float, min_progression: float, interval: float
) -> float:
    """Calculate the interval boundary for a given progression by rounding down.

    Interval boundaries are at: min_prog, min_prog + interval,
    min_prog + 2*interval, etc. This method rounds down the given
    progression to the nearest (lower or equal) interval boundary.

    For example, with min_prog=0 and interval=10:
    - progression=0 -> boundary=0
    - progression=5 -> boundary=0 (rounds down)
    - progression=10 -> boundary=10
    - progression=15 -> boundary=10 (rounds down)
    - progression=23 -> boundary=20 (rounds down)

    Args:
        progression: The progression value to calculate boundary for.
        min_progression: The minimum progression value (start of first interval).
        interval: The interval size.

    Returns:
        The interval boundary that this progression is at or past (rounded down).
    """
    interval_num = (progression - min_progression) // interval
    return min_progression + interval_num * interval


def align_partial_results(
    df: pd.DataFrame,
    metrics: list[str],
    interpolation: str = "slinear",
    do_forward_fill: bool = False,
    # TODO: Allow normalizing step (e.g. subtract min time stamp)
) -> pd.DataFrame:
    """Helper function to align partial results with heterogeneous index

    Args:
        df: The DataFrame containing the raw data (in long format).
        metrics: The signatures of the metrics to consider.
        interpolation: The interpolation method used to fill missing values
            (if applicable). See `pandas.DataFrame.interpolate` for
            available options. Limit area is `inside`.
        forward_fill: If True, performs a forward fill after interpolation.
            This is useful for scalarizing learning curves when some data
            is missing. For instance, suppose we obtain a curve for task_1
            for progression in [a, b] and task_2 for progression in [c, d]
            where b < c. Performing the forward fill on task_1 is a possible
            solution.

    Returns:
        A two-tuple containing a dict mapping the provided metric names to the
        index-normalized and interpolated dataframes containing the mean (sem).
        The dataframes are indexed by timestamp ("map_key") and have columns
        corresponding to the trial index, e.g.:
        mean = {
            "metric_signature": pd.DataFrame(
                timestamp         0           1          2           3          4
                0.0        146.138620  113.057480  44.627226  143.375669  65.033535
                1.0        117.388086   90.815154  35.847504  115.168704  52.239184
                2.0         99.950007   77.324501  30.522333   98.060315  44.479018
                3.0               NaN         NaN        NaN         NaN  39.772239
            )
        }
    """
    missing_metrics = set(metrics) - set(df["metric_signature"])
    if missing_metrics:
        raise ValueError(f"Metrics {missing_metrics} not found in input dataframe")
    # select relevant metrics
    df = df[df["metric_signature"].isin(metrics)]
    # log some information about raw data
    for m in metrics:
        df_m = df[df["metric_signature"] == m]
        if len(df_m) > 0:
            logger.debug(
                f"Metric {m} raw data has observations from "
                f"{df_m[MAP_KEY].min()} to {df_m[MAP_KEY].max()}."
            )
        else:
            logger.info(f"No data from metric {m} yet.")
    # drop arm names (assumes 1:1 map between trial indices and arm names)
    # NOTE: this is not the case for BatchTrials and repeated arms
    # if we didn't catch that there were multiple arms per trial, the interpolation
    # code below would interpolate between data points from potentially different arms,
    # as only the trial index is used to differentiate distinct data for interpolation.
    for trial_index, trial_group in df.groupby("trial_index"):
        if len(trial_group["arm_name"].unique()) != 1:
            raise UnsupportedError(
                f"Trial {trial_index} has multiple arm names: "
                f"{trial_group['arm_name'].unique()}."
            )

    for arm_name, arm_group in df.groupby("arm_name"):
        if len(arm_group["trial_index"].unique()) != 1:
            raise UnsupportedError(
                f"Arm {arm_name} has multiple trial indices: "
                f"{arm_group['trial_index'].unique()}."
            )

    df = df.drop("arm_name", axis=1)
    # remove duplicates (same trial, metric, step), which can happen
    # if the same progression is erroneously reported more than once
    df = df.drop_duplicates(
        subset=["trial_index", "metric_signature", MAP_KEY], keep="first"
    )
    has_sem = not df["sem"].isnull().all()
    # wide dataframe with hierarchical columns aligned to common index
    # (outer join of map keys across "trial_index", "metric_signature")
    wide_df: pd.DataFrame = df.pivot(
        index=MAP_KEY,
        columns=["metric_signature", "trial_index"],
        values=["mean", *(["sem"] if has_sem else [])],
    )
    # interpolation is only possible for columns with at least 2 entries,
    # will raise `ValueError` otherwise
    active = wide_df.notna().sum(axis=0) > 1
    active_cols = wide_df.columns[active]
    # interpolate / fill missing results
    wide_df[active_cols] = wide_df[active_cols].interpolate(
        method=interpolation, limit_area="inside", axis=0
    )
    if do_forward_fill:
        # do forward fill (with valid observations) to handle instances
        # where one task only has data for early progressions
        wide_df[active_cols] = wide_df[active_cols].fillna(method="pad", axis=0)

    return wide_df


def estimate_early_stopping_savings(experiment: Experiment) -> float:
    """Estimate resource savings from early stopping trials.

    Uses the average progression across completed trials as a baseline to estimate
    how much resource each early stopped trial would have consumed.
    Savings are computed as:

        savings = total_resources_saved / total_resources_used

    Args:
        experiment: The experiment to analyze.

    Returns:
        Resource savings as a fraction of total usage. For example, 0.11 indicates
        the experiment would have used 11% more resources without early stopping.
    """

    map_data = assert_is_instance(experiment.lookup_data(), MapData)
    if map_data.df.empty:
        return 0
    # Get max progression (resources used) for each trial
    resources_used = map_data.map_df.groupby("trial_index")[MAP_KEY].max()

    trials_by_status = experiment.trial_indices_by_status
    stopped_trials = trials_by_status[TrialStatus.EARLY_STOPPED]
    completed_trials = trials_by_status[TrialStatus.COMPLETED]

    # Baseline: average resources used across completed trials
    avg_completed_resources_used = resources_used.loc[[*completed_trials]].mean()

    # Calculate resources saved per stopped trial (clip negatives to zero)
    stopped_resources_used = resources_used.loc[[*stopped_trials]]
    resources_saved = (avg_completed_resources_used - stopped_resources_used).clip(
        lower=0
    )

    # Return fraction of total resources saved
    return resources_saved.sum() / resources_used.sum()

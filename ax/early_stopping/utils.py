#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict
from logging import Logger

import pandas as pd
from ax.core.experiment import Experiment
from ax.core.map_data import MAP_KEY, MapData
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UnsupportedError
from ax.utils.common.logger import get_logger
from pyre_extensions import assert_is_instance

logger: Logger = get_logger(__name__)


def align_partial_results(
    df: pd.DataFrame,
    progr_key: str,  # progression key
    metrics: list[str],
    interpolation: str = "slinear",
    do_forward_fill: bool = False,
    # TODO: Allow normalizing progr_key (e.g. subtract min time stamp)
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Helper function to align partial results with heterogeneous index

    Args:
        df: The DataFrame containing the raw data (in long format).
        progr_key: The key of the column indexing progression (such as
            the number of training examples, timestamps, etc.).
        metrics: The names of the metrics to consider.
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
            "metric_name": pd.DataFrame(
                timestamp         0           1          2           3          4
                0.0        146.138620  113.057480  44.627226  143.375669  65.033535
                1.0        117.388086   90.815154  35.847504  115.168704  52.239184
                2.0         99.950007   77.324501  30.522333   98.060315  44.479018
                3.0               NaN         NaN        NaN         NaN  39.772239
            )
        }
    """
    missing_metrics = set(metrics) - set(df["metric_name"])
    if missing_metrics:
        raise ValueError(f"Metrics {missing_metrics} not found in input dataframe")
    # select relevant metrics
    df = df[df["metric_name"].isin(metrics)]
    # log some information about raw data
    for m in metrics:
        df_m = df[df["metric_name"] == m]
        if len(df_m) > 0:
            logger.debug(
                f"Metric {m} raw data has observations from "
                f"{df_m[progr_key].min()} to {df_m[progr_key].max()}."
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
                f"Arm {arm_name} has multiple tiral indices: "
                f"{arm_group['trial_index'].unique()}."
            )

    df = df.drop("arm_name", axis=1)
    # remove duplicates (same trial, metric, progr_key), which can happen
    # if the same progression is erroneously reported more than once
    df = df.drop_duplicates(
        subset=["trial_index", "metric_name", progr_key], keep="first"
    )
    # set multi-index over trial, metric, and progression key
    df = df.set_index(["trial_index", "metric_name", progr_key])
    # sort index
    df = df.sort_index()
    # drop sem if all NaN (assumes presence of sem column)
    has_sem = not df["sem"].isnull().all()
    if not has_sem:
        df = df.drop("sem", axis=1)
    # create the common index that every map result will be re-indexed w.r.t.
    index_union = df.index.levels[2].unique()
    # loop through (trial, metric) combos and align data
    dfs_mean = defaultdict(list)
    dfs_sem = defaultdict(list)
    for tidx in df.index.levels[0]:  # this could be slow if there are many trials
        for metric in df.index.levels[1]:
            # grab trial+metric sub-df and reindex to common index
            df_ridx = df.loc[(tidx, metric)].reindex(index_union)
            # interpolate / fill missing results
            # TODO: Allow passing of additional kwargs to `interpolate`
            # TODO: Allow using an arbitrary prediction model for this instead
            try:
                df_interp = df_ridx.interpolate(
                    method=interpolation, limit_area="inside"
                )
                if do_forward_fill:
                    # do forward fill (with valid observations) to handle instances
                    # where one task only has data for early progressions
                    df_interp = df_interp.fillna(method="pad")
            except ValueError:
                df_interp = df_ridx

            # renaming column to trial index, append results
            dfs_mean[metric].append(df_interp["mean"].rename(tidx))
            if has_sem:
                dfs_sem[metric].append(df_interp["sem"].rename(tidx))

    # combine results into output dataframes
    dfs_mean = {metric: pd.concat(dfs, axis=1) for metric, dfs in dfs_mean.items()}
    dfs_sem = {metric: pd.concat(dfs, axis=1) for metric, dfs in dfs_sem.items()}

    return dfs_mean, dfs_sem


def estimate_early_stopping_savings(experiment: Experiment) -> float:
    """Estimate resource savings due to early stopping by considering
    COMPLETED and EARLY_STOPPED trials. First, use the mean of final
    progressions of the set completed trials as a benchmark for the
    length of a single trial. The savings is then estimated as:

    resource_savings =
      1 - actual_resource_usage / (num_trials * length of single trial)

    Args:
        experiment: The experiment.

    Returns:
        The estimated resource savings as a fraction of total resource usage (i.e.
        0.11 estimated savings indicates we would expect the experiment to have used 11%
        more resources without early stopping present).
    """

    map_data = assert_is_instance(experiment.lookup_data(), MapData)
    if len(map_data.df) == 0:
        return 0
    # Get final number of steps of each trial
    trial_resources = (
        map_data.map_df[["trial_index", MAP_KEY]]
        .groupby("trial_index")
        .max()
        .reset_index()
    )

    early_stopped_trial_idcs = experiment.trial_indices_by_status[
        TrialStatus.EARLY_STOPPED
    ]
    completed_trial_idcs = experiment.trial_indices_by_status[TrialStatus.COMPLETED]

    # Assume that any early stopped trial would have had the mean number of steps of
    # the completed trials
    mean_completed_trial_resources = trial_resources[
        trial_resources["trial_index"].isin(completed_trial_idcs)
    ][MAP_KEY].mean()

    # Calculate the steps saved per early stopped trial. If savings are estimated to be
    # negative assume no savings
    stopped_trial_resources = trial_resources[
        trial_resources["trial_index"].isin(early_stopped_trial_idcs)
    ][MAP_KEY]
    saved_trial_resources = (
        mean_completed_trial_resources - stopped_trial_resources
    ).clip(0)

    # Return the ratio of the total saved resources over the total resources used plus
    # the total saved resources
    return saved_trial_resources.sum() / trial_resources[MAP_KEY].sum()

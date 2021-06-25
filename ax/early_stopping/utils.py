#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd
from ax.utils.common.logger import get_logger

logger = get_logger(__name__)


def align_partial_results(
    df: pd.DataFrame,
    progr_key: str,  # progression key
    metrics: List[str],
    interpolation: str = "slinear",
    # TODO: Allow normalizing progr_key (e.g. subtract min time stamp)
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Helper function to align partial results with heterogeneous index

    Args:
        df: The DataFrame containining the raw data (in long format).
        progr_key: The key of the column indexing progression (such as
            the number of training examples, timestamps, etc.).
        metrics: The names of the metrics to consider.
        interpolation: The interpolation method used to fill missing values
            (if applicable). See `pandas.DataFrame.interpolate` for
            available options.

    Returns:
        A two-tuple containig a dict mapping the provided metric names to the
        index-normalized and interpolated mean (sem).
    """
    missing_metrics = set(metrics) - set(df["metric_name"])
    if missing_metrics:
        raise ValueError(f"Metrics {missing_metrics} not found in input dataframe")
    # select relevant metrics
    df = df[df["metric_name"].isin(metrics)]
    # drop arm names (assumes 1:1 map between trial indices and arm names)
    df = df.drop("arm_name", axis=1)
    # set multi-index over trial, metric, and progression key
    df = df.set_index(["trial_index", "metric_name", progr_key])
    # sort index
    df = df.sort_index(level=["trial_index", progr_key])
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
            # interpolate / fill missing results (only fills in between points,
            # does not extrapolate)
            # TODO: Allow passing of additional kwargs to `interpolate`
            # TODO: Allow using an arbitrary prediction model for this instead
            try:
                df_interp = df_ridx.interpolate(
                    method=interpolation, limit_area="inside"
                )
            except ValueError as e:
                df_interp = df_ridx
                logger.info(
                    f"Got exception `{e}` during interpolation. "
                    "Using uninterpolated values instead."
                )
            # renaming column to trial index, append results
            dfs_mean[metric].append(df_interp["mean"].rename(tidx))
            if has_sem:
                dfs_sem[metric].append(df_interp["sem"].rename(tidx))

    # combine results into output dataframes
    dfs_mean = {metric: pd.concat(dfs, axis=1) for metric, dfs in dfs_mean.items()}
    dfs_sem = {metric: pd.concat(dfs, axis=1) for metric, dfs in dfs_sem.items()}

    return dfs_mean, dfs_sem

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional

import pandas as pd
from ax.core import Experiment
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment


def exp_to_df(
    exp: Experiment,
    metrics: Optional[List[Metric]] = None,
    key_components: Optional[List[str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Transforms an experiment to a DataFrame. Only supports Experiment and
    SimpleExperiment.

    Transforms an Experiment into a dataframe with rows keyed by trial_index
    and arm_name, metrics pivoted into one row.

    Args:
        exp: An Experiment that may have pending trials.
        metrics: Override list of metrics to return. Return all metrics if None.
        key_components: fields that combine to make a unique key corresponding
            to rows, similar to the list of fields passed to a GROUP BY.
            Defaults to ['arm_name', 'trial_index'].
        **kwargs: Custom named arguments, useful for passing complex
            objects from call-site to the `fetch_data` callback.

    Returns:
        DataFrame: A dataframe of inputs and metrics by trial and arm.
    """
    key_components = key_components or ["trial_index", "arm_name"]

    # Accept Experiment and SimpleExperiment
    if isinstance(exp, MultiTypeExperiment):
        raise ValueError("Cannot transform MultiTypeExperiments to DataFrames.")

    results = exp.fetch_data(metrics, **kwargs).df
    if len(results.index) == 0:  # Handle empty case
        return results
    key_col = "-".join(key_components)
    key_vals = results[key_components[0]].astype("str")
    for key in key_components[1:]:
        key_vals = key_vals + results[key].astype("str")
    results[key_col] = key_vals

    metric_vals = results.pivot(
        index=key_col, columns="metric_name", values="mean"
    ).reset_index()
    metadata = results[key_components + [key_col]].drop_duplicates()
    metric_and_metadata = pd.merge(metric_vals, metadata, on=key_col)
    arm_names_and_params = pd.DataFrame(
        [{"arm_name": name, **arm.parameters} for name, arm in exp.arms_by_name.items()]
    )
    exp_df = pd.merge(metric_and_metadata, arm_names_and_params, on="arm_name")
    return exp_df.drop(key_col, axis=1).sort_values(key_components)

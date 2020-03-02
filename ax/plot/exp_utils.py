#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from ax.core import Experiment
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from pandas import DataFrame


def _rename_tuples(input):
    if isinstance(input, tuple):
        if not input[1]:
            return input[0]
        else:
            return input[1]  # "_".join(input)
    else:
        return input


def _compact_column(df, column):
    metrics_name_list = list(df[column].columns)
    temp = df[column][metrics_name_list[0]]
    del df[column]
    df[column] = temp


def exp_to_df(
    exp: Experiment,
    metrics: Optional[List[Metric]] = None,
    key_components: Optional[List[str]] = None,
) -> DataFrame:
    """Transforms an experiment to a DataFrame. Only supports SimpleExperiments.

    Transforms an Experiment into a dataframe with rows keyed by trial_index
    and arm_name, metrics pivoted into one row.

    Args:
        exp: An Experiment that may have pending trials.
        metrics: Override list of metrics to return. Return all metrics if None.
        key_components: fields that combine to make a unique key corresponding
            to rows, similar to the list of fields passed to a GROUP BY.
            Defaults to ['arm_name', 'trial_index'].

    Returns:
        DataFrame: A dataframe of inputs and metrics by trial and arm.
    """
    key_components = key_components or ["trial_index", "arm_name"]

    # Accept Experiment and SimpleExperiment
    if isinstance(exp, MultiTypeExperiment):
        raise ValueError("Cannot transform MultiTypeExperiments to DataFrames.")

    results = exp.fetch_data(metrics).df
    if len(results.index) == 0:  # Handle empty case
        return results
    key_col = "-".join(key_components)
    key_vals = results[key_components[0]].astype("str")
    for key in key_components[1:]:
        key_vals = key_vals + results[key].astype("str")
    results[key_col] = key_vals
    metrics_pivot = results.pivot(
        index=key_col, columns="metric_name", values=["mean"] + key_components
    )

    for key in key_components:
        _compact_column(metrics_pivot, key)
    inputs = DataFrame(
        [
            dict(arm.parameters, arm_name=name)
            for i, (name, arm) in enumerate(exp.arms_by_name.items())
        ]
    )
    metrics_pivot = metrics_pivot.reset_index(drop=True)
    results = metrics_pivot.merge(inputs, on="arm_name", copy=False)  # pyre-ignore

    results.rename(columns=_rename_tuples, inplace=True)
    results = results.loc[:, ~results.columns.duplicated()]
    return results

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import pandas as pd
from ax.core.experiment import Experiment
from ax.plot.base import AxPlotConfig, AxPlotTypes
from ax.service.utils.report_utils import _get_shortest_unique_suffix_dict, exp_to_df
from plotly import express as px, graph_objs as go


def prepare_experiment_for_plotting(
    experiment: Experiment,
    ignored_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Strip variables not desired in the final plot and truncate names for readability

    Args:
        experiment: Experiment containing trials to plot
        ignored_names: Metrics present in the experiment data we wish to exclude from
            the final plot. By default we ignore ["generation_method", "trial_status",
            "arm_name"]

    Returns:
        df.DataFrame: data frame ready for ingestion by plotly
    """
    ignored_names = (
        ["generation_method", "trial_status", "arm_name"]
        if ignored_names is None
        else ignored_names
    )

    df = exp_to_df(experiment)

    dropped = df.drop(ignored_names, axis=1)

    renamed = dropped.rename(
        # pyre-fixme[6] Expected `typing.Union[
        # typing.Callable[[Optional[typing.Hashable]], Optional[typing.Hashable]],
        # None, typing.Mapping[Optional[typing.Hashable], typing.Any]]` for 1st
        # parameter `columns` to call `pd.core.frame.DataFrame.rename` but got
        # `typing.Dict[str, str]`.
        columns=_get_shortest_unique_suffix_dict([str(c) for c in dropped.columns])
    )

    return renamed


def plot_parallel_coordinates_plotly(
    experiment: Experiment, ignored_names: Optional[List[str]] = None
) -> go.Figure:
    """Plot trials as a parallel coordinates graph

    Args:
        experiment: Experiment containing trials to plot
        ignored_names: Metrics present in the experiment data we wish to exclude from
            the final plot. By default we ignore ["generation_method", "trial_status",
            "arm_name"]

    Returns:
        go.Figure: parellel coordinates plot of all experiment trials
    """
    df = prepare_experiment_for_plotting(
        experiment=experiment, ignored_names=ignored_names
    )

    return px.parallel_coordinates(df, color=df.columns[0])


def plot_parallel_coordinates(
    experiment: Experiment, ignored_names: Optional[List[str]] = None
) -> AxPlotConfig:
    """Plot trials as a parallel coordinates graph

    Args:
        experiment: Experiment containing trials to plot
        ignored_names: Metrics present in the experiment data we wish to exclude from
            the final plot. By default we ignore ["generation_method", "trial_status",
            "arm_name"]

    Returns:
        AxPlotConfig: parellel coordinates plot of all experiment trials
    """
    fig = plot_parallel_coordinates_plotly(
        experiment=experiment, ignored_names=ignored_names
    )

    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)

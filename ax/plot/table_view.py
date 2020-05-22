#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import pandas as pd
import plotly.graph_objs as go
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.modelbridge.factory import get_empirical_bayes_thompson, get_thompson
from ax.plot.base import AxPlotConfig, AxPlotTypes, PlotMetric, Z
from ax.plot.helper import get_plot_data
from ax.plot.scatter import _error_scatter_data


COLOR_SCALE = ["#ff3333", "#ff6666", "#ffffff", "#99ff99", "#33ff33"]


def get_color(x: float, ci: float, rel: bool, reverse: bool):
    """Determine the color of the table cell."""
    if not rel:
        # Color coding is meant to be relative to the status quo,
        # and thus doesn't make sense if rel = False
        return "#ffffff"

    r = min(math.floor(abs(x) / ci), 2) if ci > 0 else 2
    index = int(2 + r * math.copysign(1, x))
    color_scale = list(COLOR_SCALE)
    if reverse:
        color_scale = list(reversed(color_scale))
    return color_scale[index]


def table_view_plot(
    experiment: Experiment,
    data: Data,
    use_empirical_bayes: bool = True,
    only_data_frame: bool = False,
    arm_noun: str = "arm",
):
    """Table of means and confidence intervals.

    Table is of the form:

    +-------+------------+-----------+
    |  arm  |  metric_1  |  metric_2 |
    +=======+============+===========+
    |  0_0  | mean +- CI |    ...    |
    +-------+------------+-----------+
    |  0_1  |    ...     |    ...    |
    +-------+------------+-----------+

    """
    model_func = get_empirical_bayes_thompson if use_empirical_bayes else get_thompson
    model = model_func(experiment=experiment, data=data)

    # We don't want to include metrics from a collection,
    # or the chart will be too big to read easily.
    # Example:
    # experiment.metrics = {
    #   'regular_metric': Metric(),
    #   'collection_metric: CollectionMetric()', # collection_metric =[metric1, metric2]
    # }
    # model.metric_names = [regular_metric, metric1, metric2] # "exploded" out
    # We want to filter model.metric_names and get rid of metric1, metric2
    metric_names = [
        metric_name
        for metric_name in model.metric_names
        if metric_name in experiment.metrics
    ]

    metric_name_to_lower_is_better = {
        metric_name: experiment.metrics[metric_name].lower_is_better
        for metric_name in metric_names
    }

    plot_data, _, _ = get_plot_data(
        model=model,
        generator_runs_dict={},
        # pyre-fixme[6]: Expected `Optional[typing.Set[str]]` for 3rd param but got
        #  `List[str]`.
        metric_names=metric_names,
    )

    if plot_data.status_quo_name:
        status_quo_arm = plot_data.in_sample.get(plot_data.status_quo_name)
        rel = True
    else:
        status_quo_arm = None
        rel = False

    records = []
    colors = []
    records_with_mean = []
    records_with_ci = []
    for metric_name in metric_names:
        arm_names, _, ys, ys_se = _error_scatter_data(
            # pyre-fixme[6]: Expected
            #  `List[typing.Union[ax.plot.base.PlotInSampleArm,
            #  ax.plot.base.PlotOutOfSampleArm]]` for 1st param but got
            #  `List[ax.plot.base.PlotInSampleArm]`.
            arms=list(plot_data.in_sample.values()),
            y_axis_var=PlotMetric(metric_name, pred=True, rel=rel),
            x_axis_var=None,
            status_quo_arm=status_quo_arm,
        )

        results_by_arm = list(zip(arm_names, ys, ys_se))
        colors.append(
            [
                get_color(
                    x=y,
                    ci=Z * y_se,
                    rel=rel,
                    # pyre-fixme[6]: Expected `bool` for 4th param but got
                    #  `Optional[bool]`.
                    reverse=metric_name_to_lower_is_better[metric_name],
                )
                for (_, y, y_se) in results_by_arm
            ]
        )
        records.append(
            [
                "{:.3f} &plusmn; {:.3f}".format(y, Z * y_se)
                for (_, y, y_se) in results_by_arm
            ]
        )
        records_with_mean.append({arm_name: y for (arm_name, y, _) in results_by_arm})
        records_with_ci.append(
            {arm_name: Z * y_se for (arm_name, _, y_se) in results_by_arm}
        )

    if only_data_frame:
        return tuple(
            pd.DataFrame.from_records(records, index=metric_names)
            for records in [records_with_mean, records_with_ci]
        )

    def transpose(m):
        return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

    records = [[name.replace(":", " : ") for name in metric_names]] + transpose(records)
    colors = [["#ffffff"] * len(metric_names)] + transpose(colors)
    # pyre-fixme[6]: Expected `List[str]` for 1st param but got `List[float]`.
    header = [f"<b>{x}</b>" for x in [f"{arm_noun}s"] + arm_names]
    column_widths = [300] + [150] * len(arm_names)

    trace = go.Table(
        header={"values": header, "align": ["left"]},
        cells={"values": records, "align": ["left"], "fill": {"color": colors}},
        columnwidth=column_widths,
    )
    layout = go.Layout(
        width=sum(column_widths),
        margin=go.layout.Margin(l=0, r=20, b=20, t=20, pad=4),  # noqa E741
    )
    fig = go.Figure(data=[trace], layout=layout)
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)

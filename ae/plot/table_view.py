#!/usr/bin/env python3

import math

import plotly.graph_objs as go
from ae.lazarus.ae.core.data import Data
from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.generator.factory import get_empirical_bayes_thompson
from ae.lazarus.ae.plot.base import AEPlotConfig, AEPlotTypes, PlotMetric, Z
from ae.lazarus.ae.plot.helper import get_plot_data
from ae.lazarus.ae.plot.scatter import _error_scatter_data


COLOR_SCALE = ["#ffaaa5", "#ffd3b6", "#ffffff", "#dcedc1", "#a8e6cf"]


def get_color(x, ci):
    r = min(math.floor(abs(x) / ci), 2)
    return COLOR_SCALE[int(2 + r * math.copysign(1, x))]


def table_view_plot(experiment: Experiment, data: Data):
    """ Render a table of the form:
                  metric1      metric2
    arms    mean +- CI   ...
    0_0     ...
    0_1
    """
    generator = get_empirical_bayes_thompson(experiment=experiment, data=data)

    results = {}
    plot_data, _, _ = get_plot_data(
        generator=generator, generator_runs_dict={}, metric_names=generator.metric_names
    )
    for metric_name in generator.metric_names:
        arms, _, ys, ys_se = _error_scatter_data(
            arms=list(plot_data.in_sample.values()),
            y_axis_var=PlotMetric(metric_name, True),
            x_axis_var=None,
            rel=True,
            status_quo_arm=plot_data.in_sample.get(plot_data.status_quo_name),
        )
        # add spaces to metric name to it wraps
        metric_name = metric_name.replace(":", " : ")
        # results[metric] will hold a list of tuples, one tuple per arm
        results[metric_name] = list(zip(arms, ys, ys_se))

    # cells and colors are both lists of lists
    # each top-level list corresponds to a column,
    # so the first is a list of arms
    cells = [[f"<b>{x}</b>" for x in arms]]
    colors = [["#ffffff"] * len(arms)]
    metric_names = []
    for metric_name, list_of_tuples in sorted(results.items()):
        cells.append(
            [
                "{:.3f} &plusmn; {:.3f}".format(y, Z * y_se)
                for (_, y, y_se) in list_of_tuples
            ]
        )
        metric_names.append(metric_name)
        colors.append([get_color(y, Z * y_se) for (_, y, y_se) in list_of_tuples])

    header = ["arms"] + metric_names
    header = [f"<b>{x}</b>" for x in header]
    trace = go.Table(
        header={"values": header, "align": ["left"]},
        cells={"values": cells, "align": ["left"], "fill": {"color": colors}},
    )
    layout = go.Layout(
        height=min([400, len(arms) * 20 + 200]),
        width=175 * len(header),
        margin=go.Margin(l=0, r=20, b=20, t=20, pad=4),  # noqa E741
    )
    fig = go.Figure(data=[trace], layout=layout)
    return AEPlotConfig(data=fig, plot_type=AEPlotTypes.GENERIC)

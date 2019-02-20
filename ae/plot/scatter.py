#!/usr/bin/env python3

import numbers
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objs as go
from ae.lazarus.ae.generator.base import Generator
from ae.lazarus.ae.plot.base import (
    CI_OPACITY,
    DECIMALS,
    AEPlotConfig,
    AEPlotTypes,
    PlotInSampleCondition,
    PlotMetric,
    PlotOutOfSampleCondition,
    Z,
)
from ae.lazarus.ae.plot.color import COLORS, DISCRETE_COLOR_SCALE, rgba
from ae.lazarus.ae.plot.helper import (
    TNullableGeneratorRunsDict,
    _format_CI,
    _format_dict,
    _wrap_metric,
    condition_name_to_tuple,
    get_plot_data,
    resize_subtitles,
)
from ae.lazarus.ae.utils.stats.statstools import relativize
from plotly import tools


# type aliases
Traces = List[Dict[str, Any]]


def _error_scatter_data(
    conditions: List[Union[PlotInSampleCondition, PlotOutOfSampleCondition]],
    y_axis_var: PlotMetric,
    x_axis_var: Optional[PlotMetric] = None,
    rel: bool = False,
    status_quo_condition: Optional[PlotInSampleCondition] = None,
) -> Tuple[List[float], Optional[List[float]], List[float], List[float]]:
    y_metric_key = "y_hat" if y_axis_var.pred else "y"
    y_sd_key = "se_hat" if y_axis_var.pred else "se"

    condition_names = [a.name for a in conditions]
    y = [getattr(a, y_metric_key).get(y_axis_var.metric, np.nan) for a in conditions]
    y_se = [getattr(a, y_sd_key).get(y_axis_var.metric, np.nan) for a in conditions]

    # Delta method if relative to status quo condition
    if rel:
        if status_quo_condition is None:
            raise ValueError(
                "`status_quo_condition` cannot be None for relative effects."
            )
        y_rel, y_se_rel = relativize(
            means_t=y,
            sems_t=y_se,
            mean_c=getattr(status_quo_condition, y_metric_key).get(y_axis_var.metric),
            sem_c=getattr(status_quo_condition, y_sd_key).get(y_axis_var.metric),
            as_percent=True,
        )
        y = y_rel.tolist()
        y_se = y_se_rel.tolist()

    # x can be metric for a metric or condition names
    if x_axis_var is None:
        x = condition_names
        x_se = None
    else:
        x_metric_key = "y_hat" if x_axis_var.pred else "y"
        x_sd_key = "se_hat" if x_axis_var.pred else "se"
        x = [
            getattr(a, x_metric_key).get(x_axis_var.metric, np.nan) for a in conditions
        ]
        x_se = [getattr(a, x_sd_key).get(x_axis_var.metric, np.nan) for a in conditions]

        if rel:
            # Delta method if relative to status quo condition
            x_rel, x_se_rel = relativize(
                means_t=x,
                sems_t=x_se,
                mean_c=getattr(status_quo_condition, x_metric_key).get(
                    x_axis_var.metric
                ),
                sem_c=getattr(status_quo_condition, x_sd_key).get(x_axis_var.metric),
                as_percent=True,
            )
            x = x_rel.tolist()
            x_se = x_se_rel.tolist()
    return x, x_se, y, y_se


def _error_scatter_trace(
    conditions: List[Union[PlotInSampleCondition, PlotOutOfSampleCondition]],
    y_axis_var: PlotMetric,
    x_axis_var: Optional[PlotMetric] = None,
    y_axis_label: Optional[str] = None,
    x_axis_label: Optional[str] = None,
    rel: bool = False,
    status_quo_condition: Optional[PlotInSampleCondition] = None,
    show_CI: bool = True,
    name: str = "In-sample",
    color: Tuple[int] = COLORS.STEELBLUE.value,
    visible: bool = True,
    legendgroup: Optional[str] = None,
    showlegend: bool = True,
    hoverinfo: str = "text",
    show_condition_details_on_hover: bool = True,
    show_context: bool = False,
) -> Dict[str, Any]:
    """Plot scatterplot with error bars.

    Args:
        conditions (List[Union[PlotInSampleCondition, PlotOutOfSampleCondition]]):
            a list of in-sample or out-of-sample conditions.
            In-sample conditions have observed data, while out-of-sample conditions
            just have predicted data. As a result,
            when passing out-of-sample conditions, pred must be True.
        y_axis_var (PlotMetric): name of metric for y-axis, along with whether
            it is observed or predicted.
        x_axis_var (Optional[PlotMetric], optional): name of metric for x-axis,
            along with whether it is observed or predicted. If None, condition names
            are automatically used.
        y_axis_label (Optional[str], optional): custom label to use for y axis.
            If None, use metric name from `y_axis_var`.
        x_axis_label (Optional[str], optional): custom label to use for x axis.
            If None, use metric name from `x_axis_var` if that is not None.
        rel (bool, optional): if True, points are treated as relative to status
            quo and '%' is appended to labels. If rel=True, `status_quo_condition`
            must be set.
        status_quo_condition (Optional[PlotInSampleCondition], optional): the status quo
            condition. Necessary for relative metrics.
        show_CI (bool, optional): if True, plot confidence intervals.
        name (string): name of trace. Default is "In-sample".
        color (Tuple[int], optional): color as rgb tuple. Default is
            (128, 177, 211), which corresponds to COLORS.STEELBLUE.
        visible (bool, optional): if True, trace is visible (default).
        legendgroup (string, optional): group for legends.
        showlegend (bool, optional): if True, legend if rendered.
        hoverinfo (string, optional): information to show on hover. Default is
            custom text.
        show_condition_details_on_hover (bool, optional): if True, display
            parameterizations of conditions on hover. Default is True.
        show_context (bool, optional): if True and show_condition_details_on_hover,
            context will be included in the hover.
    """
    x, x_se, y, y_se = _error_scatter_data(
        conditions=conditions,
        y_axis_var=y_axis_var,
        x_axis_var=x_axis_var,
        rel=rel,
        status_quo_condition=status_quo_condition,
    )
    labels = []

    condition_names = [a.name for a in conditions]
    for i in range(len(condition_names)):
        heading = "<b>Condition {}</b><br>".format(condition_names[i])
        x_lab = (
            "{name}: {estimate}{perc} {ci}<br>".format(
                name=x_axis_var.metric if x_axis_label is None else x_axis_label,
                estimate=(
                    round(x[i], DECIMALS) if isinstance(x[i], numbers.Number) else x[i]
                ),
                ci="" if x_se is None else _format_CI(x[i], x_se[i], rel),
                perc="%" if rel else "",
            )
            if x_axis_var is not None
            else ""
        )
        y_lab = "{name}: {estimate}{perc} {ci}<br>".format(
            name=y_axis_var.metric if y_axis_label is None else y_axis_label,
            estimate=(
                round(y[i], DECIMALS) if isinstance(y[i], numbers.Number) else y[i]
            ),
            ci="" if y_se is None else _format_CI(y[i], y_se[i], rel),
            perc="%" if rel else "",
        )

        parameterization = (
            _format_dict(conditions[i].params, "Parameterization")
            if show_condition_details_on_hover
            else ""
        )

        context = (
            # Expected `Dict[str, Optional[Union[bool, float, str]]]` for 1st anonymous
            # parameter to call `ae.lazarus.ae.plot.helper._format_dict` but got
            # `Optional[Dict[str, Union[float, str]]]`.
            # pyre-fixme[6]:
            _format_dict(conditions[i].context_stratum, "Context")
            if show_condition_details_on_hover
            and show_context  # noqa W503
            and conditions[i].context_stratum  # noqa W503
            else ""
        )

        labels.append(
            "{condition_name}<br>{xlab}{ylab}{param_blob}{context}".format(
                condition_name=heading,
                xlab=x_lab,
                ylab=y_lab,
                param_blob=parameterization,
                context=context,
            )
        )
        i += 1
    trace = go.Scatter(  # pyre-ignore[16]: `plotly.graph_objs` has no attr. `Scatter`
        x=x,
        y=y,
        marker={"color": rgba(color)},
        mode="markers",
        name=name,
        text=labels,
        hoverinfo=hoverinfo,
    )

    if show_CI:
        if x_se is not None:
            trace.update(
                error_x={
                    "type": "data",
                    "array": np.multiply(x_se, Z),
                    "color": rgba(color, CI_OPACITY),
                }
            )
        if y_se is not None:
            trace.update(
                error_y={
                    "type": "data",
                    "array": np.multiply(y_se, Z),
                    "color": rgba(color, CI_OPACITY),
                }
            )
    if visible is not None:
        trace.update(visible=visible)
    if legendgroup is not None:
        trace.update(legendgroup=legendgroup)
    if showlegend is not None:
        trace.update(showlegend=showlegend)
    return trace


def _multiple_metric_traces(
    generator: Generator,
    metric_x: str,
    metric_y: str,
    generator_runs_dict: TNullableGeneratorRunsDict,
    rel: bool,
) -> Traces:
    """Plot traces for multiple metrics given a generator and metrics.

    Args:
        generator (Generator): generator to draw predictions from.
        metric_x (str): metric to plot on the x-axis.
        metric_y (str): metric to plot on the y-axis.
        generator_runs_dict (Dict[str, GeneratorRun], optional): a mapping from
            generator run name to generator run.
        rel (bool): if True, use relative effects.

    """
    plot_data, _, _ = get_plot_data(
        generator,
        generator_runs_dict if generator_runs_dict is not None else {},
        {metric_x, metric_y},
    )

    status_quo_condition = (
        None
        if plot_data.status_quo_name is None
        else plot_data.in_sample.get(plot_data.status_quo_name)
    )

    traces = [
        _error_scatter_trace(
            # Expected `List[Union[PlotInSampleCondition, PlotOutOfSampleCondition]]`
            # for 1st anonymous parameter to call
            # `ae.lazarus.ae.plot.scatter._error_scatter_trace` but got
            # `List[PlotInSampleCondition]`.
            # pyre-fixme[6]:
            list(plot_data.in_sample.values()),
            x_axis_var=PlotMetric(metric_x, pred=False),
            y_axis_var=PlotMetric(metric_y, pred=False),
            rel=rel,
            status_quo_condition=status_quo_condition,
            visible=False,
        ),
        _error_scatter_trace(
            # Expected `List[Union[PlotInSampleCondition, PlotOutOfSampleCondition]]`
            # for 1st anonymous parameter to call
            # `ae.lazarus.ae.plot.scatter._error_scatter_trace` but got
            # `List[PlotInSampleCondition]`.
            # pyre-fixme[6]:
            list(plot_data.in_sample.values()),
            x_axis_var=PlotMetric(metric_x, pred=True),
            y_axis_var=PlotMetric(metric_y, pred=True),
            rel=rel,
            status_quo_condition=status_quo_condition,
            visible=True,
        ),
    ]

    for i, (generator_run_name, cand_conditions) in enumerate(
        (plot_data.out_of_sample or {}).items(), start=1
    ):
        traces.append(
            _error_scatter_trace(
                # pyre: Expected `List[Union[PlotInSampleCondition,
                # pyre: PlotOutOfSampleCondition]]` for 1st anonymous
                # pyre: parameter to call `ae.lazarus.ae.plot.scatter.
                # pyre: _error_scatter_trace` but got
                # pyre-fixme[6]: `List[PlotOutOfSampleCondition]`.
                list(cand_conditions.values()),
                x_axis_var=PlotMetric(metric_x, pred=True),
                y_axis_var=PlotMetric(metric_y, pred=True),
                rel=rel,
                status_quo_condition=status_quo_condition,
                name=generator_run_name,
                color=DISCRETE_COLOR_SCALE[i],
            )
        )
    return traces


def plot_multiple_metrics(
    generator: Generator,
    metric_x: str,
    metric_y: str,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel: bool = True,
) -> AEPlotConfig:
    """Plot raw values or predictions of two metrics for conditions.

    All conditions used in the generator are included in the plot. Additional
    conditions can be passed through the `generator_runs_dict` argument.

    Args:
        generator (Generator): generator to draw predictions from.
        metric_x (str): metric to plot on the x-axis.
        metric_y (str): metric to plot on the y-axis.
        generator_runs_dict (Dict[str, GeneratorRun], optional): a mapping from
            generator run name to generator run.
        rel (bool, optional): if True, use relative effects. Default is True.

    """
    traces = _multiple_metric_traces(
        generator, metric_x, metric_y, generator_runs_dict, rel
    )
    num_cand_traces = len(generator_runs_dict) if generator_runs_dict is not None else 0

    layout = go.Layout(  # pyre-ignore[16]
        title="Objective Tradeoffs",
        hovermode="closest",
        updatemenus=[
            {
                "x": 1.25,
                "y": 0.67,
                "buttons": [
                    {
                        "args": [
                            {
                                "error_x.width": 4,
                                "error_x.thickness": 2,
                                "error_y.width": 4,
                                "error_y.thickness": 2,
                            }
                        ],
                        "label": "Yes",
                        "method": "restyle",
                    },
                    {
                        "args": [
                            {
                                "error_x.width": 0,
                                "error_x.thickness": 0,
                                "error_y.width": 0,
                                "error_y.thickness": 0,
                            }
                        ],
                        "label": "No",
                        "method": "restyle",
                    },
                ],
                "yanchor": "middle",
                "xanchor": "left",
            },
            {
                "x": 1.25,
                "y": 0.57,
                "buttons": [
                    {
                        "args": [
                            {"visible": ([False, True] + [True] * num_cand_traces)}
                        ],
                        "label": "Modeled",
                        "method": "restyle",
                    },
                    {
                        "args": [
                            {"visible": ([True, False] + [False] * num_cand_traces)}
                        ],
                        "label": "Observed",
                        "method": "restyle",
                    },
                ],
                "yanchor": "middle",
                "xanchor": "left",
            },
        ],
        annotations=[
            {
                "x": 1.18,
                "y": 0.7,
                "xref": "paper",
                "yref": "paper",
                "text": "Show CI",
                "showarrow": False,
                "yanchor": "middle",
            },
            {
                "x": 1.18,
                "y": 0.6,
                "xref": "paper",
                "yref": "paper",
                "text": "Type",
                "showarrow": False,
                "yanchor": "middle",
            },
        ],
        xaxis={
            "title": metric_x + (" (%)" if rel else ""),
            "zeroline": True,
            "zerolinecolor": "red",
        },
        yaxis={
            "title": metric_y + (" (%)" if rel else ""),
            "zeroline": True,
            "zerolinecolor": "red",
        },
        width=800,
        height=600,
        font={"size": 10},
    )

    fig = go.Figure(data=traces, layout=layout)  # pyre-ignore[16]
    return AEPlotConfig(data=fig, plot_type=AEPlotTypes.GENERIC)


# @TODO: we can eventually pass an Objective object directly to this function
# to extract object and outcome constraints.


def plot_objective_vs_constraints(
    generator: Generator,
    objective: str,
    subset_metrics: Optional[List[str]] = None,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel: bool = True,
) -> AEPlotConfig:
    """Plot the tradeoff between an objetive and all other metrics in a generator.

    All conditions used in the generator are included in the plot. Additional
    conditions can be passed through via the `generator_runs_dict` argument.

    Args:
        generator (Generator): generator to draw predictions from.
        objective (str): metric to optimize. Plotted on the x-axis.
        subset_metrics (List[str]): list of metrics to plot on the y-axes
            if need a subset of all metrics in the generator.
        generator_runs_dict (Dict[str, GeneratorRun], optional): a mapping from
            generator run name to generator run.
        rel (bool, optional): if True, use relative effects. Default is True.

    """
    if subset_metrics is not None:
        metrics = subset_metrics
    else:
        metrics = [m for m in generator.metric_names if m != objective]

    metric_dropdown = []

    # set plotted data to the first outcome
    plot_data = _multiple_metric_traces(
        generator, objective, metrics[0], generator_runs_dict, rel
    )

    for metric in metrics:
        otraces = _multiple_metric_traces(
            generator, objective, metric, generator_runs_dict, rel
        )

        # Current version of Plotly does not allow updating the yaxis label
        # on dropdown (via relayout) simultaneously with restyle
        metric_dropdown.append(
            {
                "args": [
                    {
                        "y": [t["y"] for t in otraces],
                        "error_y.array": [t["error_y"]["array"] for t in otraces],
                        "text": [t["text"] for t in otraces],
                    },
                    {"yaxis.title": metric + (" (%)" if rel else "")},
                ],
                "label": metric,
                "method": "update",
            }
        )

    num_cand_traces = len(generator_runs_dict) if generator_runs_dict is not None else 0

    layout = go.Layout(  # pyre-ignore[16]
        title="Objective Tradeoffs",
        hovermode="closest",
        updatemenus=[
            {
                "x": 1.25,
                "y": 0.62,
                "buttons": [
                    {
                        "args": [
                            {
                                "error_x.width": 4,
                                "error_x.thickness": 2,
                                "error_y.width": 4,
                                "error_y.thickness": 2,
                            }
                        ],
                        "label": "Yes",
                        "method": "restyle",
                    },
                    {
                        "args": [
                            {
                                "error_x.width": 0,
                                "error_x.thickness": 0,
                                "error_y.width": 0,
                                "error_y.thickness": 0,
                            }
                        ],
                        "label": "No",
                        "method": "restyle",
                    },
                ],
                "yanchor": "middle",
                "xanchor": "left",
            },
            {
                "x": 1.25,
                "y": 0.52,
                "buttons": [
                    {
                        "args": [
                            {"visible": ([False, True] + [True] * num_cand_traces)}
                        ],
                        "label": "Modeled",
                        "method": "restyle",
                    },
                    {
                        "args": [
                            {"visible": ([True, False] + [False] * num_cand_traces)}
                        ],
                        "label": "Observed",
                        "method": "restyle",
                    },
                ],
                "yanchor": "middle",
                "xanchor": "left",
            },
            {
                "x": 1.25,
                "y": 0.72,
                "yanchor": "middle",
                "xanchor": "left",
                "buttons": metric_dropdown,
            },
        ],
        annotations=[
            {
                "x": 1.18,
                "y": 0.72,
                "xref": "paper",
                "yref": "paper",
                "text": "Y-Axis",
                "showarrow": False,
                "yanchor": "middle",
            },
            {
                "x": 1.18,
                "y": 0.62,
                "xref": "paper",
                "yref": "paper",
                "text": "Show CI",
                "showarrow": False,
                "yanchor": "middle",
            },
            {
                "x": 1.18,
                "y": 0.52,
                "xref": "paper",
                "yref": "paper",
                "text": "Type",
                "showarrow": False,
                "yanchor": "middle",
            },
        ],
        xaxis={
            "title": objective + (" (%)" if rel else ""),
            "zeroline": True,
            "zerolinecolor": "red",
        },
        yaxis={
            "title": metrics[0] + (" (%)" if rel else ""),
            "zeroline": True,
            "zerolinecolor": "red",
        },
        width=900,
        height=600,
        font={"size": 10},
    )

    fig = go.Figure(data=plot_data, layout=layout)  # pyre-ignore[16]
    return AEPlotConfig(data=fig, plot_type=AEPlotTypes.GENERIC)


def lattice_multiple_metrics(
    generator: Generator,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel: bool = True,
    show_condition_details_on_hover: bool = False,
) -> AEPlotConfig:
    """Plot raw values or predictions of combinations of two metrics for conditions.

    Args:
        generator (Generator): generator to draw predictions from.
        generator_runs_dict (Dict[str, GeneratorRun], optional): a mapping from
            generator run name to generator run.
        rel (bool, optional): if True, use relative effects. Default is True.
        show_condition_details_on_hover (bool, optional): if True, display
            parameterizations of conditions on hover. Default is False.

    """
    metrics = generator.metric_names
    fig = tools.make_subplots(
        rows=len(metrics),
        cols=len(metrics),
        print_grid=False,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    plot_data, _, _ = get_plot_data(
        generator,
        generator_runs_dict if generator_runs_dict is not None else {},
        metrics,
    )
    status_quo_condition = (
        None
        if plot_data.status_quo_name is None
        else plot_data.in_sample.get(plot_data.status_quo_name)
    )

    # iterate over all combinations of metrics and generate scatter traces
    for i, o1 in enumerate(metrics, start=1):
        for j, o2 in enumerate(metrics, start=1):
            if o1 != o2:
                # in-sample observed and predicted
                obs_insample_trace = _error_scatter_trace(
                    # Expected `List[Union[PlotInSampleCondition,
                    # PlotOutOfSampleCondition]]` for 1st anonymous parameter to call
                    # `ae.lazarus.ae.plot.scatter._error_scatter_trace` but got
                    # `List[PlotInSampleCondition]`.
                    # pyre-fixme[6]:
                    list(plot_data.in_sample.values()),
                    x_axis_var=PlotMetric(o1, pred=False),
                    y_axis_var=PlotMetric(o2, pred=False),
                    rel=rel,
                    status_quo_condition=status_quo_condition,
                    showlegend=(i is 1 and j is 2),
                    legendgroup="In-sample",
                    visible=False,
                    show_condition_details_on_hover=show_condition_details_on_hover,
                )
                predicted_insample_trace = _error_scatter_trace(
                    # Expected `List[Union[PlotInSampleCondition,
                    # PlotOutOfSampleCondition]]` for 1st anonymous parameter to call
                    # `ae.lazarus.ae.plot.scatter._error_scatter_trace` but got
                    # `List[PlotInSampleCondition]`.
                    # pyre-fixme[6]:
                    list(plot_data.in_sample.values()),
                    x_axis_var=PlotMetric(o1, pred=True),
                    y_axis_var=PlotMetric(o2, pred=True),
                    rel=rel,
                    status_quo_condition=status_quo_condition,
                    legendgroup="In-sample",
                    showlegend=(i is 1 and j is 2),
                    visible=True,
                    show_condition_details_on_hover=show_condition_details_on_hover,
                )
                fig.append_trace(obs_insample_trace, j, i)
                fig.append_trace(predicted_insample_trace, j, i)

                # iterate over generators here
                for k, (generator_run_name, cand_conditions) in enumerate(
                    (plot_data.out_of_sample or {}).items(), start=1
                ):
                    fig.append_trace(
                        _error_scatter_trace(
                            # pyre: Expected
                            # pyre: `List[Union[PlotInSampleCondition,
                            # pyre: PlotOutOfSampleCondition]]` for 1st
                            # pyre: anonymous parameter to call `ae.lazarus.ae.
                            # pyre: plot.scatter._error_scatter_trace` but got
                            # pyre-fixme[6]: `List[PlotOutOfSampleCondition]`.
                            list(cand_conditions.values()),
                            x_axis_var=PlotMetric(o1, pred=True),
                            y_axis_var=PlotMetric(o2, pred=True),
                            rel=rel,
                            status_quo_condition=status_quo_condition,
                            name=generator_run_name,
                            color=DISCRETE_COLOR_SCALE[k],
                            showlegend=(i is 1 and j is 2),
                            legendgroup=generator_run_name,
                            show_condition_details_on_hover=show_condition_details_on_hover,
                        ),
                        j,
                        i,
                    )
            else:
                # if diagonal is set to True, add box plots
                fig.append_trace(
                    go.Box(  # pyre-ignore[16]
                        y=[
                            condition.y[o1]
                            for condition in plot_data.in_sample.values()
                        ],
                        name=None,
                        marker={"color": rgba(COLORS.STEELBLUE.value)},
                        showlegend=False,
                        legendgroup="In-sample",
                        visible=False,
                        hoverinfo="none",
                    ),
                    j,
                    i,
                )
                fig.append_trace(
                    go.Box(  # pyre-ignore[16]
                        y=[
                            condition.y_hat[o1]
                            for condition in plot_data.in_sample.values()
                        ],
                        name=None,
                        marker={"color": rgba(COLORS.STEELBLUE.value)},
                        showlegend=False,
                        legendgroup="In-sample",
                        hoverinfo="none",
                    ),
                    j,
                    i,
                )

                for k, (generator_run_name, cand_conditions) in enumerate(
                    (plot_data.out_of_sample or {}).items(), start=1
                ):
                    fig.append_trace(
                        go.Box(  # pyre-ignore[16]
                            y=[
                                condition.y_hat[o1]
                                for condition in cand_conditions.values()
                            ],
                            name=None,
                            marker={"color": rgba(DISCRETE_COLOR_SCALE[k])},
                            showlegend=False,
                            legendgroup=generator_run_name,
                            hoverinfo="none",
                        ),
                        j,
                        i,
                    )

    fig["layout"].update(
        height=800,
        width=960,
        font={"size": 10},
        hovermode="closest",
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.05,
            "xanchor": "left",
            "yanchor": "middle",
        },
        updatemenus=[
            {
                "x": 0.35,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "middle",
                "buttons": [
                    {
                        "args": [
                            {
                                "error_x.width": 0,
                                "error_x.thickness": 0,
                                "error_y.width": 0,
                                "error_y.thickness": 0,
                            }
                        ],
                        "label": "No",
                        "method": "restyle",
                    },
                    {
                        "args": [
                            {
                                "error_x.width": 4,
                                "error_x.thickness": 2,
                                "error_y.width": 4,
                                "error_y.thickness": 2,
                            }
                        ],
                        "label": "Yes",
                        "method": "restyle",
                    },
                ],
            },
            {
                "x": 0.1,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "middle",
                "buttons": [
                    {
                        "args": [
                            {
                                "visible": (
                                    (
                                        [False, True]
                                        + [True] * len(plot_data.out_of_sample or {})
                                    )
                                    * (len(metrics) ** 2)
                                )
                            }
                        ],
                        "label": "Modeled",
                        "method": "restyle",
                    },
                    {
                        "args": [
                            {
                                "visible": (
                                    (
                                        [True, False]
                                        + [False] * len(plot_data.out_of_sample or {})
                                    )
                                    * (len(metrics) ** 2)
                                )
                            }
                        ],
                        "label": "In-sample",
                        "method": "restyle",
                    },
                ],
            },
        ],
        annotations=[
            {
                "x": 0.02,
                "y": 1.1,
                "xref": "paper",
                "yref": "paper",
                "text": "Type",
                "showarrow": False,
                "yanchor": "middle",
                "xanchor": "left",
            },
            {
                "x": 0.30,
                "y": 1.1,
                "xref": "paper",
                "yref": "paper",
                "text": "Show CI",
                "showarrow": False,
                "yanchor": "middle",
                "xanchor": "left",
            },
        ],
    )

    # add metric names to axes - add to each subplot if boxplots on the
    # diagonal and axes are not shared; else, add to the leftmost y-axes
    # and bottom x-axes.
    for i, o in enumerate(metrics):
        pos_x = len(metrics) * len(metrics) - len(metrics) + i + 1
        pos_y = 1 + (len(metrics) * i)
        fig["layout"]["xaxis{}".format(pos_x)].update(
            title=_wrap_metric(o), titlefont={"size": 10}
        )
        fig["layout"]["yaxis{}".format(pos_y)].update(
            title=_wrap_metric(o), titlefont={"size": 10}
        )

    # do not put x-axis ticks for boxplots
    boxplot_xaxes = []
    for trace in fig["data"]:
        if trace["type"] == "box":
            # stores the xaxes which correspond to boxplot subplots
            # since we use xaxis1, xaxis2, etc, in plotly.py
            boxplot_xaxes.append("xaxis{}".format(trace["xaxis"][1:]))
        else:
            # clear all error bars since default is no CI
            trace["error_x"].update(width=0, thickness=0)
            trace["error_y"].update(width=0, thickness=0)
    for xaxis in boxplot_xaxes:
        fig["layout"][xaxis]["showticklabels"] = False

    return AEPlotConfig(data=fig, plot_type=AEPlotTypes.GENERIC)


# Single metric fitted values
def _single_metric_traces(
    generator: Generator,
    metric: str,
    generator_runs_dict: TNullableGeneratorRunsDict,
    rel: bool,
    show_condition_details_on_hover: bool = True,
    showlegend: bool = True,
    show_CI: bool = True,
) -> Traces:
    """Plot scatterplots with errors for a single metric (y-axis).

    Conditions are plotted on the x-axis.

    Args:
        generator (Generator): generator to draw predictions from.
        metric (str): name of metric to plot.
        generator_runs_dict (Dict[str, GeneratorRun], optional): a mapping from
            generator run name to generator run.
        rel (bool): if True, plot relative predictions.
        show_condition_details_on_hover (bool, optional): if True, display
            parameterizations of conditions on hover. Default is True.
        show_legend (bool, optional): if True, show legend for trace.
        show_CI (bool, optional): if True, render confidence intervals.

    """
    plot_data, _, _ = get_plot_data(generator, generator_runs_dict or {}, {metric})

    status_quo_condition = (
        None
        if plot_data.status_quo_name is None
        else plot_data.in_sample.get(plot_data.status_quo_name)
    )

    traces = [
        _error_scatter_trace(
            # Expected `List[Union[PlotInSampleCondition, PlotOutOfSampleCondition]]`
            # for 1st anonymous parameter to call
            # `ae.lazarus.ae.plot.scatter._error_scatter_trace` but got
            # `List[PlotInSampleCondition]`.
            # pyre-fixme[6]:
            list(plot_data.in_sample.values()),
            x_axis_var=None,
            y_axis_var=PlotMetric(metric, pred=True),
            rel=rel,
            status_quo_condition=status_quo_condition,
            legendgroup="In-sample",
            showlegend=showlegend,
            show_condition_details_on_hover=show_condition_details_on_hover,
            show_CI=show_CI,
        )
    ]

    # Candidates
    for i, (generator_run_name, cand_conditions) in enumerate(
        (plot_data.out_of_sample or {}).items(), start=1
    ):
        traces.append(
            _error_scatter_trace(
                # pyre: Expected `List[Union[PlotInSampleCondition,
                # pyre: PlotOutOfSampleCondition]]` for 1st anonymous
                # pyre: parameter to call `ae.lazarus.ae.plot.scatter.
                # pyre: _error_scatter_trace` but got
                # pyre-fixme[6]: `List[PlotOutOfSampleCondition]`.
                list(cand_conditions.values()),
                x_axis_var=None,
                y_axis_var=PlotMetric(metric, pred=True),
                rel=rel,
                status_quo_condition=status_quo_condition,
                name=generator_run_name,
                color=DISCRETE_COLOR_SCALE[i],
                legendgroup=generator_run_name,
                showlegend=showlegend,
                show_condition_details_on_hover=show_condition_details_on_hover,
                show_CI=show_CI,
            )
        )
    return traces


def plot_fitted(
    generator: Generator,
    metric: str,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel: bool = True,
    custom_condition_order: Optional[List[str]] = None,
    custom_condition_order_name: str = "Custom",
    show_CI: bool = True,
) -> AEPlotConfig:
    """Plot fitted metrics.

    Args:
        generator (Generator): generator to use for predictions.
        metric (str): metric to plot predictions for.
        generator_runs_dict (Dict[str, GeneratorRun], optional): a mapping from
            generator run name to generator run.
        rel (bool, optional): if True, use relative effects. Default is True.
        custom_condition_order (List[str], optional): a list of condition names in the
            order corresponding to how they should be plotted on the x-axis.
            If not None, this is the default ordering.
        custom_condition_order_name (str, optional): name for custom ordering to
            show in the ordering dropdown. Default is 'Custom'.
        show_CI (bool, optional): if True, render confidence intervals.

    """
    traces = _single_metric_traces(
        generator, metric, generator_runs_dict, rel, show_CI=show_CI
    )

    # order condition name sorting condition numbers within batch
    names_by_condition = sorted(
        np.unique(np.concatenate([d["x"] for d in traces])),
        key=lambda x: condition_name_to_tuple(x),
    )

    # get condition names sorted by effect size
    names_by_effect = list(
        OrderedDict.fromkeys(
            np.concatenate([d["x"] for d in traces])
            .flatten()
            .take(np.argsort(np.concatenate([d["y"] for d in traces]).flatten()))
        )
    )

    # options for ordering conditions (x-axis)
    xaxis_categoryorder = "array"
    xaxis_categoryarray = names_by_condition

    order_options = [
        {
            "args": [
                {
                    "xaxis.categoryorder": "array",
                    "xaxis.categoryarray": names_by_condition,
                }
            ],
            "label": "Name",
            "method": "relayout",
        },
        {
            "args": [
                {"xaxis.categoryorder": "array", "xaxis.categoryarray": names_by_effect}
            ],
            "label": "Effect Size",
            "method": "relayout",
        },
    ]

    # if a custom order has been passed, default to that
    if custom_condition_order is not None:
        xaxis_categoryorder = "array"
        xaxis_categoryarray = custom_condition_order
        order_options = [
            {
                "args": [
                    {
                        "xaxis.categoryorder": "array",
                        "xaxis.categoryarray": custom_condition_order,
                    }
                ],
                "label": custom_condition_order_name,
                "method": "relayout",
            }
        ] + order_options

    layout = go.Layout(  # pyre-ignore[16]
        title="Predicted Outcomes",
        hovermode="closest",
        updatemenus=[
            {
                "x": 1.25,
                "y": 0.67,
                "buttons": list(order_options),
                "yanchor": "middle",
                "xanchor": "left",
            }
        ],
        yaxis={
            "zerolinecolor": "red",
            "title": "{}{}".format(metric, " (%)" if rel else ""),
        },
        xaxis={
            "tickangle": 45,
            "categoryorder": xaxis_categoryorder,
            "categoryarray": xaxis_categoryarray,
        },
        annotations=[
            {
                "x": 1.18,
                "y": 0.72,
                "xref": "paper",
                "yref": "paper",
                "text": "Sort By",
                "showarrow": False,
                "yanchor": "middle",
            }
        ],
        font={"size": 10},
    )

    fig = go.Figure(data=traces, layout=layout)  # pyre-ignore[16]
    return AEPlotConfig(data=fig, plot_type=AEPlotTypes.GENERIC)


def tile_fitted(
    generator: Generator,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel: bool = True,
    show_condition_details_on_hover: bool = False,
    show_CI: bool = True,
) -> AEPlotConfig:
    """Tile version of fitted outcome plots.

    Args:
        generator (Generator): generator to use for predictions.
        generator_runs_dict (Dict[str, GeneratorRun], optional): a mapping from
            generator run name to generator run.
        rel (bool, optional): if True, use relative effects. Default is True.
        show_condition_details_on_hover (bool, optional): if True, display
            parameterizations of conditions on hover. Default is False.
        show_CI (bool, optional): if True, render confidence intervals.

    """
    metrics = generator.metric_names
    nrows = int(np.ceil(len(metrics) / 2))
    ncols = min(len(metrics), 2)

    # make subplots (plot per row)
    fig = tools.make_subplots(
        rows=nrows,
        cols=ncols,
        print_grid=False,
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=tuple(metrics),
        horizontal_spacing=0.05,
        vertical_spacing=0.30 / nrows,
    )

    name_order_args: Dict[str, Any] = {}
    name_order_axes: Dict[str, Dict[str, Any]] = {}
    effect_order_args: Dict[str, Any] = {}

    for i, metric in enumerate(metrics):
        data = _single_metric_traces(
            generator,
            metric,
            generator_runs_dict,
            rel,
            showlegend=i == 0,
            show_condition_details_on_hover=show_condition_details_on_hover,
            show_CI=show_CI,
        )

        # order condition name sorting condition numbers within batch
        names_by_condition = sorted(
            np.unique(np.concatenate([d["x"] for d in data])),
            key=lambda x: condition_name_to_tuple(x),
        )

        # get condition names sorted by effect size
        names_by_effect = list(
            OrderedDict.fromkeys(
                np.concatenate([d["x"] for d in data])
                .flatten()
                .take(np.argsort(np.concatenate([d["y"] for d in data]).flatten()))
            )
        )

        # options for ordering conditions (x-axis)
        # Note that xaxes need to be references as xaxis, xaxis2, xaxis3, etc.
        # for the purposes of updatemenus argument (dropdown) in layout.
        # However, when setting the initial ordering layout, the keys should be
        # xaxis1, xaxis2, xaxis3, etc. Note the discrepancy for the initial
        # axis.
        label = "" if i == 0 else i + 1
        name_order_args["xaxis{}.categoryorder".format(label)] = "array"
        name_order_args["xaxis{}.categoryarray".format(label)] = names_by_condition
        effect_order_args["xaxis{}.categoryorder".format(label)] = "array"
        effect_order_args["xaxis{}.categoryarray".format(label)] = names_by_effect
        name_order_axes["xaxis{}".format(i + 1)] = {
            "categoryorder": "array",
            "categoryarray": names_by_condition,
        }
        name_order_axes["yaxis{}".format(i + 1)] = {
            "ticksuffix": "%" if rel else "",
            "zerolinecolor": "red",
        }
        for d in data:
            fig.append_trace(d, int(np.floor(i / ncols)) + 1, i % ncols + 1)

    order_options = [
        {"args": [name_order_args], "label": "Name", "method": "relayout"},
        {"args": [effect_order_args], "label": "Effect Size", "method": "relayout"},
    ]

    # if odd number of plots, need to manually remove the last blank subplot
    # generated by `tools.make_subplots`
    if len(metrics) % 2 == 1:
        del fig["layout"]["xaxis{}".format(nrows * ncols)]
        del fig["layout"]["yaxis{}".format(nrows * ncols)]

    # allocate 400 px per plot
    fig["layout"].update(
        margin={"t": 0},
        hovermode="closest",
        updatemenus=[
            {
                "x": 0.15,
                "y": 1 + 0.40 / nrows,
                "buttons": order_options,
                "xanchor": "left",
                "yanchor": "middle",
            }
        ],
        font={"size": 10},
        width=650 if ncols == 1 else 950,
        height=300 * nrows,
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1 + 0.20 / nrows,
            "xanchor": "left",
            "yanchor": "middle",
        },
        **name_order_axes
    )

    # append dropdown annotations
    fig["layout"]["annotations"] += [
        {
            "x": 0.5,
            "y": 1 + 0.40 / nrows,
            "xref": "paper",
            "yref": "paper",
            "font": {"size": 14},
            "text": "Predicted Outcomes",
            "showarrow": False,
            "xanchor": "center",
            "yanchor": "middle",
        },
        {
            "x": 0.05,
            "y": 1 + 0.40 / nrows,
            "xref": "paper",
            "yref": "paper",
            "text": "Sort By",
            "showarrow": False,
            "xanchor": "left",
            "yanchor": "middle",
        },
    ]

    fig = resize_subtitles(figure=fig, size=10)
    return AEPlotConfig(data=fig, plot_type=AEPlotTypes.GENERIC)


def interact_fitted(
    generator: Generator,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel: bool = True,
    show_condition_details_on_hover: bool = True,
    show_CI: bool = True,
) -> AEPlotConfig:
    """Interactive fitted outcome plots for each condition used in fitting the generator.

    Choose the outcome to plot using a dropdown.

    Args:
        generator (Generator): generator to use for predictions.
        generator_runs_dict (Dict[str, GeneratorRun], optional): a mapping from
            generator run name to generator run.
        rel (bool, optional): if True, use relative effects. Default is True.
        show_condition_details_on_hover (bool, optional): if True, display
            parameterizations of conditions on hover. Default is True.
        show_CI (bool, optional): if True, render confidence intervals.

    """
    traces_per_metric = (
        1 if generator_runs_dict is None else len(generator_runs_dict) + 1
    )
    metrics = sorted(generator.metric_names)

    traces = []
    dropdown = []

    for i, metric in enumerate(metrics):
        data = _single_metric_traces(
            generator,
            metric,
            generator_runs_dict,
            rel,
            showlegend=i == 0,
            show_condition_details_on_hover=show_condition_details_on_hover,
            show_CI=show_CI,
        )

        for d in data:
            d["visible"] = i == 0
            traces.append(d)

        # only the first two traces are visible (corresponding to first outcome
        # in dropdown)
        is_visible = [False] * (len(metrics) * traces_per_metric)
        for j in range((traces_per_metric * i), (traces_per_metric * (i + 1))):
            is_visible[j] = True

        # on dropdown change, restyle
        dropdown.append(
            {"args": ["visible", is_visible], "label": metric, "method": "restyle"}
        )

    layout = go.Layout(  # pyre-ignore[16]
        xaxis={"title": "Condition", "zeroline": False},
        yaxis={
            "ticksuffix": "%" if rel else "",
            "title": ("Relative " if rel else "") + "Effect",
            "zeroline": True,
            "zerolinecolor": "red",
        },
        hovermode="closest",
        updatemenus=[
            {
                "buttons": dropdown,
                "x": 0.075,
                "xanchor": "left",
                "y": 1.1,
                "yanchor": "middle",
            }
        ],
        annotations=[
            {
                "font": {"size": 12},
                "showarrow": False,
                "text": "Metric",
                "x": 0.05,
                "xanchor": "right",
                "xref": "paper",
                "y": 1.1,
                "yanchor": "middle",
                "yref": "paper",
            }
        ],
        legend={
            "orientation": "h",
            "x": 0.065,
            "xanchor": "left",
            "y": 1.2,
            "yanchor": "middle",
        },
        height=500,
    )

    if traces_per_metric > 1:
        layout["annotations"].append(
            {
                "font": {"size": 12},
                "showarrow": False,
                "text": "Condition Source",
                "x": 0.05,
                "xanchor": "right",
                "xref": "paper",
                "y": 1.2,
                "yanchor": "middle",
                "yref": "paper",
            }
        )

    return AEPlotConfig(
        data=go.Figure(data=traces, layout=layout),  # pyre-ignore[16]
        plot_type=AEPlotTypes.GENERIC,
    )

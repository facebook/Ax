#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numbers
import warnings
from collections import OrderedDict

from logging import Logger
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objs as go
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.observation import Observation, ObservationFeatures
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.registry import Models
from ax.plot.base import (
    AxPlotConfig,
    AxPlotTypes,
    CI_OPACITY,
    DECIMALS,
    PlotInSampleArm,
    PlotMetric,
    PlotOutOfSampleArm,
    Z,
)
from ax.plot.color import BLUE_SCALE, COLORS, DISCRETE_COLOR_SCALE, rgba
from ax.plot.helper import (
    _format_CI,
    _format_dict,
    _wrap_metric,
    arm_name_to_sort_key,
    arm_name_to_tuple,
    get_plot_data,
    infer_is_relative,
    resize_subtitles,
    TNullableGeneratorRunsDict,
)
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast_optional
from ax.utils.stats.statstools import relativize
from plotly import subplots

logger: Logger = get_logger(__name__)

# type aliases
Traces = List[Dict[str, Any]]


def _error_scatter_data(
    arms: Iterable[Union[PlotInSampleArm, PlotOutOfSampleArm]],
    y_axis_var: PlotMetric,
    x_axis_var: Optional[PlotMetric] = None,
    status_quo_arm: Optional[PlotInSampleArm] = None,
) -> Tuple[List[float], Optional[List[float]], List[float], List[float]]:
    y_metric_key = "y_hat" if y_axis_var.pred else "y"
    y_sd_key = "se_hat" if y_axis_var.pred else "se"

    arm_names = [a.name for a in arms]
    y = [getattr(a, y_metric_key).get(y_axis_var.metric, np.nan) for a in arms]
    y_se = [getattr(a, y_sd_key).get(y_axis_var.metric, np.nan) for a in arms]

    # Delta method if relative to status quo arm
    if y_axis_var.rel:
        if status_quo_arm is None:
            raise ValueError("`status_quo_arm` cannot be None for relative effects.")
        y_rel, y_se_rel = relativize(
            means_t=y,
            sems_t=y_se,
            mean_c=getattr(status_quo_arm, y_metric_key).get(y_axis_var.metric),
            sem_c=getattr(status_quo_arm, y_sd_key).get(y_axis_var.metric),
            as_percent=True,
        )
        y = y_rel.tolist()
        y_se = y_se_rel.tolist()

    # x can be metric for a metric or arm names
    if x_axis_var is None:
        x = arm_names
        x_se = None
    else:
        x_metric_key = "y_hat" if x_axis_var.pred else "y"
        x_sd_key = "se_hat" if x_axis_var.pred else "se"
        x = [getattr(a, x_metric_key).get(x_axis_var.metric, np.nan) for a in arms]
        x_se = [getattr(a, x_sd_key).get(x_axis_var.metric, np.nan) for a in arms]

        if x_axis_var.rel:
            # Delta method if relative to status quo arm
            x_rel, x_se_rel = relativize(
                means_t=x,
                sems_t=x_se,
                mean_c=getattr(status_quo_arm, x_metric_key).get(x_axis_var.metric),
                sem_c=getattr(status_quo_arm, x_sd_key).get(x_axis_var.metric),
                as_percent=True,
            )
            x = x_rel.tolist()
            x_se = x_se_rel.tolist()
    return x, x_se, y, y_se


def _error_scatter_trace(
    arms: Sequence[Union[PlotInSampleArm, PlotOutOfSampleArm]],
    y_axis_var: PlotMetric,
    x_axis_var: Optional[PlotMetric] = None,
    y_axis_label: Optional[str] = None,
    x_axis_label: Optional[str] = None,
    status_quo_arm: Optional[PlotInSampleArm] = None,
    show_CI: bool = True,
    name: str = "In-sample",
    color: Tuple[int] = COLORS.STEELBLUE.value,
    visible: bool = True,
    legendgroup: Optional[str] = None,
    showlegend: bool = True,
    hoverinfo: str = "text",
    show_arm_details_on_hover: bool = True,
    show_context: bool = False,
    arm_noun: str = "arm",
    color_parameter: Optional[str] = None,
    color_metric: Optional[str] = None,
) -> Dict[str, Any]:
    """Plot scatterplot with error bars.

    Args:
        arms (List[Union[PlotInSampleArm, PlotOutOfSampleArm]]):
            a list of in-sample or out-of-sample arms.
            In-sample arms have observed data, while out-of-sample arms
            just have predicted data. As a result,
            when passing out-of-sample arms, pred must be True.
        y_axis_var: name of metric for y-axis, along with whether
            it is observed or predicted.
        x_axis_var: name of metric for x-axis,
            along with whether it is observed or predicted. If None, arm names
            are automatically used.
        y_axis_label: custom label to use for y axis.
            If None, use metric name from `y_axis_var`.
        x_axis_label: custom label to use for x axis.
            If None, use metric name from `x_axis_var` if that is not None.
        status_quo_arm: the status quo
            arm. Necessary for relative metrics.
        show_CI: if True, plot confidence intervals.
        name: name of trace. Default is "In-sample".
        color: color as rgb tuple. Default is
            (128, 177, 211), which corresponds to COLORS.STEELBLUE.
        visible: if True, trace is visible (default).
        legendgroup: group for legends.
        showlegend: if True, legend if rendered.
        hoverinfo: information to show on hover. Default is
            custom text.
        show_arm_details_on_hover: if True, display
            parameterizations of arms on hover. Default is True.
        show_context: if True and show_arm_details_on_hover,
            context will be included in the hover.
        arm_noun: noun to use instead of "arm" (e.g. group)
        color_parameter: color points according to the specified parameter,
            cannot be used together with color_metric.
        color_metric: color points according to the specified metric,
            cannot be used together with color_parameter.
    """
    if color_metric and color_parameter:
        raise RuntimeError(
            "color_metric and color_parameter cannot be used at the same time!"
        )

    if (color_metric or color_parameter) and not all(
        isinstance(arm, PlotInSampleArm) for arm in arms
    ):
        raise RuntimeError("Color coding currently only works with in-sample arms!")

    # Opportunistically sort if arm names are in {trial}_{arm} format
    arms = sorted(arms, key=lambda a: arm_name_to_sort_key(a.name), reverse=True)

    x, x_se, y, y_se = _error_scatter_data(
        arms=arms,
        y_axis_var=y_axis_var,
        x_axis_var=x_axis_var,
        status_quo_arm=status_quo_arm,
    )
    labels = []
    colors = []

    arm_names = [a.name for a in arms]

    # No relativization if no x variable.
    rel_x = x_axis_var.rel if x_axis_var else False
    rel_y = y_axis_var.rel

    for i in range(len(arm_names)):
        heading = f"<b>{arm_noun.title()} {arm_names[i]}</b><br>"
        x_lab = (
            "{name}: {estimate}{perc} {ci}<br>".format(
                name=x_axis_var.metric if x_axis_label is None else x_axis_label,
                estimate=(
                    round(x[i], DECIMALS) if isinstance(x[i], numbers.Number) else x[i]
                ),
                ci="" if x_se is None else _format_CI(x[i], x_se[i], rel_x),
                perc="%" if rel_x else "",
            )
            if x_axis_var is not None
            else ""
        )
        y_lab = "{name}: {estimate}{perc} {ci}<br>".format(
            name=y_axis_var.metric if y_axis_label is None else y_axis_label,
            estimate=(
                round(y[i], DECIMALS) if isinstance(y[i], numbers.Number) else y[i]
            ),
            ci="" if y_se is None else _format_CI(y[i], y_se[i], rel_y),
            perc="%" if rel_y else "",
        )

        parameterization = (
            _format_dict(arms[i].parameters, "Parameterization")
            if show_arm_details_on_hover
            else ""
        )

        if color_parameter:
            colors.append(arms[i].parameters[color_parameter])
        elif color_metric:
            # Must be PlotInSampleArm here if no error raised previously
            # pyre-ignore[16]: `PlotOutOfSampleArm` has no attribute `y`
            colors.append(arms[i].y[color_metric])

        context = (
            # Expected `Dict[str, Optional[Union[bool, float, str]]]` for 1st anonymous
            # parameter to call `ax.plot.helper._format_dict` but got
            # `Optional[Dict[str, Union[float, str]]]`.
            # pyre-fixme[6]:
            _format_dict(arms[i].context_stratum, "Context")
            if show_arm_details_on_hover
            and show_context  # noqa W503
            and arms[i].context_stratum  # noqa W503
            else ""
        )

        labels.append(
            "{arm_name}<br>{xlab}{ylab}{param_blob}{context}".format(
                arm_name=heading,
                xlab=x_lab,
                ylab=y_lab,
                param_blob=parameterization,
                context=context,
            )
        )
        i += 1

    if color_metric or color_parameter:
        rgba_blue_scale = [rgba(c) for c in BLUE_SCALE]
        marker = {
            "color": colors,
            "colorscale": rgba_blue_scale,
            "colorbar": {"title": color_metric or color_parameter},
            "showscale": True,
        }
    else:
        marker = {"color": rgba(color)}

    trace = go.Scatter(
        x=x,
        y=y,
        marker=marker,
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
    model: ModelBridge,
    metric_x: str,
    metric_y: str,
    generator_runs_dict: TNullableGeneratorRunsDict,
    rel_x: bool,
    rel_y: bool,
    fixed_features: Optional[ObservationFeatures] = None,
    data_selector: Optional[Callable[[Observation], bool]] = None,
    color_parameter: Optional[str] = None,
    color_metric: Optional[str] = None,
) -> Traces:
    """Plot traces for multiple metrics given a model and metrics.

    Args:
        model: model to draw predictions from.
        metric_x: metric to plot on the x-axis.
        metric_y: metric to plot on the y-axis.
        generator_runs_dict: a mapping from
            generator run name to generator run.
        rel_x: if True, use relative effects on metric_x.
        rel_y: if True, use relative effects on metric_y.
        fixed_features: Fixed features to use when making model predictions.
        data_selector: Function for selecting observations for plotting.
        color_parameter: color points according to the specified parameter,
            cannot be used together with color_metric.
        color_metric: color points according to the specified metric,
            cannot be used together with color_parameter.
    """
    metric_names = {metric_x, metric_y}
    if color_metric is not None:
        metric_names.add(color_metric)

    plot_data, _, _ = get_plot_data(
        model,
        generator_runs_dict if generator_runs_dict is not None else {},
        metric_names,
        fixed_features=fixed_features,
        data_selector=data_selector,
    )

    status_quo_arm = (
        None
        if plot_data.status_quo_name is None
        else plot_data.in_sample.get(plot_data.status_quo_name)
    )

    traces = [
        _error_scatter_trace(
            list(plot_data.in_sample.values()),
            x_axis_var=PlotMetric(metric_x, pred=False, rel=rel_x),
            y_axis_var=PlotMetric(metric_y, pred=False, rel=rel_y),
            status_quo_arm=status_quo_arm,
            visible=False,
            color_parameter=color_parameter,
            color_metric=color_metric,
        ),
        _error_scatter_trace(
            list(plot_data.in_sample.values()),
            x_axis_var=PlotMetric(metric_x, pred=True, rel=rel_x),
            y_axis_var=PlotMetric(metric_y, pred=True, rel=rel_y),
            status_quo_arm=status_quo_arm,
            visible=True,
            color_parameter=color_parameter,
            color_metric=color_metric,
        ),
    ]

    # TODO: Figure out if there's a better way to color code out-of-sample points
    for i, (generator_run_name, cand_arms) in enumerate(
        (plot_data.out_of_sample or {}).items(), start=1
    ):
        traces.append(
            _error_scatter_trace(
                list(cand_arms.values()),
                x_axis_var=PlotMetric(metric_x, pred=True, rel=rel_x),
                y_axis_var=PlotMetric(metric_y, pred=True, rel=rel_y),
                status_quo_arm=status_quo_arm,
                name=generator_run_name,
                color=DISCRETE_COLOR_SCALE[i],
            )
        )
    return traces


def plot_multiple_metrics(
    model: ModelBridge,
    metric_x: str,
    metric_y: str,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel_x: bool = True,
    rel_y: bool = True,
    fixed_features: Optional[ObservationFeatures] = None,
    data_selector: Optional[Callable[[Observation], bool]] = None,
    color_parameter: Optional[str] = None,
    color_metric: Optional[str] = None,
    **kwargs: Any,
) -> AxPlotConfig:
    """Plot raw values or predictions of two metrics for arms.

    All arms used in the model are included in the plot. Additional
    arms can be passed through the `generator_runs_dict` argument.

    Args:
        model: model to draw predictions from.
        metric_x: metric to plot on the x-axis.
        metric_y: metric to plot on the y-axis.
        generator_runs_dict: a mapping from
            generator run name to generator run.
        rel_x: if True, use relative effects on metric_x.
        rel_y: if True, use relative effects on metric_y.
        data_selector: Function for selecting observations for plotting.
        color_parameter: color points according to the specified parameter,
            cannot be used together with color_metric.
        color_metric: color points according to the specified metric,
            cannot be used together with color_parameter.
    """
    if color_parameter or color_metric:
        layout_offset_x = 0.15
    else:
        layout_offset_x = 0
    rel = checked_cast_optional(bool, kwargs.get("rel"))
    if rel is not None:
        warnings.warn("Use `rel_x` and `rel_y` instead of `rel`.", DeprecationWarning)
        rel_x = rel
        rel_y = rel
    traces = _multiple_metric_traces(
        model,
        metric_x,
        metric_y,
        generator_runs_dict,
        rel_x=rel_x,
        rel_y=rel_y,
        fixed_features=fixed_features,
        data_selector=data_selector,
        color_parameter=color_parameter,
        color_metric=color_metric,
    )
    num_cand_traces = len(generator_runs_dict) if generator_runs_dict is not None else 0
    layout = go.Layout(
        title="Objective Tradeoffs",
        hovermode="closest",
        updatemenus=[
            {
                "x": 1.25 + layout_offset_x,
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
                "x": 1.25 + layout_offset_x,
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
                "x": 1.18 + layout_offset_x,
                "y": 0.7,
                "xref": "paper",
                "yref": "paper",
                "text": "Show CI",
                "showarrow": False,
                "yanchor": "middle",
            },
            {
                "x": 1.18 + layout_offset_x,
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
        legend={"x": 1 + layout_offset_x},
    )
    fig = go.Figure(data=traces, layout=layout)
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)


def plot_objective_vs_constraints(
    model: ModelBridge,
    objective: str,
    subset_metrics: Optional[List[str]] = None,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel: bool = True,
    infer_relative_constraints: Optional[bool] = False,
    fixed_features: Optional[ObservationFeatures] = None,
    data_selector: Optional[Callable[[Observation], bool]] = None,
    color_parameter: Optional[str] = None,
    color_metric: Optional[str] = None,
    label_dict: Optional[Dict[str, str]] = None,
) -> AxPlotConfig:
    """Plot the tradeoff between an objective and all other metrics in a model.

    All arms used in the model are included in the plot. Additional
    arms can be passed through via the `generator_runs_dict` argument.

    Fixed features input can be used to override fields of the insample arms
    when making model predictions.

    Args:
        model: model to draw predictions from.
        objective: metric to optimize. Plotted on the x-axis.
        subset_metrics: list of metrics to plot on the y-axes
            if need a subset of all metrics in the model.
        generator_runs_dict: a mapping from
            generator run name to generator run.
        rel: if True, use relative effects. Default is True.
        infer_relative_constraints: if True, read relative spec from model's
            optimization config. Absolute constraints will not be relativized;
            relative ones will be.
            Objectives will respect the `rel` parameter.
            Metrics that are not constraints will be relativized.
        fixed_features: Fixed features to use when making model predictions.
        data_selector: Function for selecting observations for plotting.
        color_parameter: color points according to the specified parameter,
            cannot be used together with color_metric.
        color_metric: color points according to the specified metric,
            cannot be used together with color_parameter.
        label_dict: A dictionary that maps the label to
            an alias to be used in the plot.
    """
    if color_parameter or color_metric:
        layout_offset_x = 0.15
    else:
        layout_offset_x = 0
    if subset_metrics is not None:
        metrics = subset_metrics
    else:
        metrics = [m for m in model.metric_names if m != objective]
    if not label_dict:
        _check_label_lengths(metrics + [objective])

    metric_dropdown = []

    if infer_relative_constraints:
        rels = infer_is_relative(model, metrics, non_constraint_rel=rel)
        if rel:
            rels[objective] = True
        else:
            rels[objective] = False
    else:
        if rel:
            rels = {metric: True for metric in metrics}
            rels[objective] = True
        else:
            rels = {metric: False for metric in metrics}
            rels[objective] = False

    # set plotted data to the first outcome
    plot_data = _multiple_metric_traces(
        model,
        objective,
        metrics[0],
        generator_runs_dict,
        rel_x=rels[objective],
        rel_y=rels[metrics[0]],
        fixed_features=fixed_features,
        data_selector=data_selector,
        color_parameter=color_parameter,
        color_metric=color_metric,
    )

    for metric in metrics:
        otraces = _multiple_metric_traces(
            model,
            objective,
            metric,
            generator_runs_dict,
            rel_x=rels[objective],
            rel_y=rels[metric],
            fixed_features=fixed_features,
            data_selector=data_selector,
            color_parameter=color_parameter,
            color_metric=color_metric,
        )

        # Current version of Plotly does not allow updating the yaxis label
        # on dropdown (via relayout) simultaneously with restyle
        metric_dropdown.append(
            {
                "args": [
                    {
                        "y": [t["y"] for t in otraces],
                        "error_y.array": [t["error_y"]["array"] for t in otraces],
                        "text": [_replace_str(t["text"], label_dict) for t in otraces],
                    },
                    {
                        "yaxis.title": _replace_str(metric, label_dict)
                        + (" (%)" if rels[metric] else ""),
                    },
                ],
                "label": _replace_str(metric, label_dict),
                "method": "update",
            }
        )

    num_cand_traces = len(generator_runs_dict) if generator_runs_dict is not None else 0

    layout = go.Layout(
        title="Objective Tradeoffs",
        hovermode="closest",
        updatemenus=[
            {
                "x": 1.25 + layout_offset_x,
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
                "x": 1.25 + layout_offset_x,
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
                "x": 1.25 + layout_offset_x,
                "y": 0.72,
                "yanchor": "middle",
                "xanchor": "left",
                "buttons": metric_dropdown,
            },
        ],
        annotations=[
            {
                "x": 1.18 + layout_offset_x,
                "y": 0.72,
                "xref": "paper",
                "yref": "paper",
                "text": "Y-Axis",
                "showarrow": False,
                "yanchor": "middle",
            },
            {
                "x": 1.18 + layout_offset_x,
                "y": 0.62,
                "xref": "paper",
                "yref": "paper",
                "text": "Show CI",
                "showarrow": False,
                "yanchor": "middle",
            },
            {
                "x": 1.18 + layout_offset_x,
                "y": 0.52,
                "xref": "paper",
                "yref": "paper",
                "text": "Type",
                "showarrow": False,
                "yanchor": "middle",
            },
        ],
        xaxis={
            "title": _replace_str(objective, label_dict)
            + (" (%)" if rels[objective] else ""),
            "zeroline": True,
            "zerolinecolor": "red",
        },
        yaxis={
            "title": _replace_str(metrics[0], label_dict)
            + (" (%)" if rels[metrics[0]] else ""),
            "zeroline": True,
            "zerolinecolor": "red",
        },
        width=900,
        height=600,
        font={"size": 10},
        legend={"x": 1 + layout_offset_x},
    )

    fig = go.Figure(data=plot_data, layout=layout)
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)


def _replace_str(input_str: str, str_dict: Optional[Dict[str, str]] = None) -> str:
    """Utility function to replace a string based on a mapping dictionary.

    Args:
        input_str: Input string to map.
        str_dict: Mapping dictionary.
    """
    return str_dict[input_str] if (str_dict and input_str in str_dict) else input_str


def _check_label_lengths(labels: List[str]) -> None:
    """Utility function to check label length and provide a warning for long
       labels pointing to a mapping that can be used to override them.

    Args:
        labels: List of labels to check.
    """
    max_len = 30
    long_labels = [label for label in labels if len(label) > max_len]
    if long_labels:
        logger.info(
            "This plot may be malformed due to long labels. You"
            " can override long labels by passing a label_dict dictionary"
            " to plotting functions that support it.\nHere's a list of labels"
            f" longer than {max_len} characters:\n" + "\n".join(long_labels)
        )


def lattice_multiple_metrics(
    model: ModelBridge,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel: bool = True,
    show_arm_details_on_hover: bool = False,
    data_selector: Optional[Callable[[Observation], bool]] = None,
) -> AxPlotConfig:
    """Plot raw values or predictions of combinations of two metrics for arms.

    Args:
        model: model to draw predictions from.
        generator_runs_dict: a mapping from
            generator run name to generator run.
        rel: if True, use relative effects. Default is True.
        show_arm_details_on_hover: if True, display
            parameterizations of arms on hover. Default is False.
        data_selector: Function for selecting observations for plotting.
    """
    metrics = model.metric_names
    fig = subplots.make_subplots(
        rows=len(metrics),
        cols=len(metrics),
        print_grid=False,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    plot_data, _, _ = get_plot_data(
        model,
        generator_runs_dict if generator_runs_dict is not None else {},
        metrics,
        data_selector=data_selector,
    )
    status_quo_arm = (
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
                    list(plot_data.in_sample.values()),
                    x_axis_var=PlotMetric(o1, pred=False, rel=rel),
                    y_axis_var=PlotMetric(o2, pred=False, rel=rel),
                    status_quo_arm=status_quo_arm,
                    showlegend=(i == 1 and j == 2),
                    legendgroup="In-sample",
                    visible=False,
                    show_arm_details_on_hover=show_arm_details_on_hover,
                )
                predicted_insample_trace = _error_scatter_trace(
                    list(plot_data.in_sample.values()),
                    x_axis_var=PlotMetric(o1, pred=True, rel=rel),
                    y_axis_var=PlotMetric(o2, pred=True, rel=rel),
                    status_quo_arm=status_quo_arm,
                    legendgroup="In-sample",
                    showlegend=(i == 1 and j == 2),
                    visible=True,
                    show_arm_details_on_hover=show_arm_details_on_hover,
                )
                fig.append_trace(obs_insample_trace, j, i)
                fig.append_trace(predicted_insample_trace, j, i)

                # iterate over models here
                for k, (generator_run_name, cand_arms) in enumerate(
                    (plot_data.out_of_sample or {}).items(), start=1
                ):
                    fig.append_trace(
                        _error_scatter_trace(
                            list(cand_arms.values()),
                            x_axis_var=PlotMetric(o1, pred=True, rel=rel),
                            y_axis_var=PlotMetric(o2, pred=True, rel=rel),
                            status_quo_arm=status_quo_arm,
                            name=generator_run_name,
                            color=DISCRETE_COLOR_SCALE[k],
                            showlegend=(i == 1 and j == 2),
                            legendgroup=generator_run_name,
                            show_arm_details_on_hover=show_arm_details_on_hover,
                        ),
                        j,
                        i,
                    )
            else:
                # if diagonal is set to True, add box plots
                fig.append_trace(
                    go.Box(
                        y=[arm.y[o1] for arm in plot_data.in_sample.values()],
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
                    go.Box(
                        y=[arm.y_hat[o1] for arm in plot_data.in_sample.values()],
                        name=None,
                        marker={"color": rgba(COLORS.STEELBLUE.value)},
                        showlegend=False,
                        legendgroup="In-sample",
                        hoverinfo="none",
                    ),
                    j,
                    i,
                )

                for k, (generator_run_name, cand_arms) in enumerate(
                    (plot_data.out_of_sample or {}).items(), start=1
                ):
                    fig.append_trace(
                        go.Box(
                            y=[arm.y_hat[o1] for arm in cand_arms.values()],
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

    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)


# Single metric fitted values
def _single_metric_traces(
    model: ModelBridge,
    metric: str,
    generator_runs_dict: TNullableGeneratorRunsDict,
    rel: bool,
    show_arm_details_on_hover: bool = True,
    showlegend: bool = True,
    show_CI: bool = True,
    arm_noun: str = "arm",
    fixed_features: Optional[ObservationFeatures] = None,
    data_selector: Optional[Callable[[Observation], bool]] = None,
) -> Traces:
    """Plot scatterplots with errors for a single metric (y-axis).

    Arms are plotted on the x-axis.

    Args:
        model: model to draw predictions from.
        metric: name of metric to plot.
        generator_runs_dict: a mapping from
            generator run name to generator run.
        rel: if True, plot relative predictions.
        show_arm_details_on_hover: if True, display
            parameterizations of arms on hover. Default is True.
        show_legend: if True, show legend for trace.
        show_CI: if True, render confidence intervals.
        arm_noun: noun to use instead of "arm" (e.g. group)
        fixed_features: Fixed features to use when making model predictions.
        data_selector: Function for selecting observations for plotting.
    """
    plot_data, _, _ = get_plot_data(
        model,
        generator_runs_dict or {},
        {metric},
        fixed_features=fixed_features,
        data_selector=data_selector,
    )

    status_quo_arm = (
        None
        if plot_data.status_quo_name is None
        else plot_data.in_sample.get(plot_data.status_quo_name)
    )

    traces = [
        _error_scatter_trace(
            list(plot_data.in_sample.values()),
            x_axis_var=None,
            y_axis_var=PlotMetric(metric, pred=True, rel=rel),
            status_quo_arm=status_quo_arm,
            legendgroup="In-sample",
            showlegend=showlegend,
            show_arm_details_on_hover=show_arm_details_on_hover,
            show_CI=show_CI,
            arm_noun=arm_noun,
        )
    ]

    # Candidates
    for i, (generator_run_name, cand_arms) in enumerate(
        (plot_data.out_of_sample or {}).items(), start=1
    ):
        traces.append(
            _error_scatter_trace(
                list(cand_arms.values()),
                x_axis_var=None,
                y_axis_var=PlotMetric(metric, pred=True, rel=rel),
                status_quo_arm=status_quo_arm,
                name=generator_run_name,
                color=DISCRETE_COLOR_SCALE[i],
                legendgroup=generator_run_name,
                showlegend=showlegend,
                show_arm_details_on_hover=show_arm_details_on_hover,
                show_CI=show_CI,
                arm_noun=arm_noun,
            )
        )
    return traces


def plot_fitted(
    model: ModelBridge,
    metric: str,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel: bool = True,
    custom_arm_order: Optional[List[str]] = None,
    custom_arm_order_name: str = "Custom",
    show_CI: bool = True,
    data_selector: Optional[Callable[[Observation], bool]] = None,
) -> AxPlotConfig:
    """Plot fitted metrics.

    Args:
        model: model to use for predictions.
        metric: metric to plot predictions for.
        generator_runs_dict: a mapping from
            generator run name to generator run.
        rel: if True, use relative effects. Default is True.
        custom_arm_order: a list of arm names in the
            order corresponding to how they should be plotted on the x-axis.
            If not None, this is the default ordering.
        custom_arm_order_name: name for custom ordering to
            show in the ordering dropdown. Default is 'Custom'.
        show_CI: if True, render confidence intervals.
        data_selector: Function for selecting observations for plotting.
    """
    traces = _single_metric_traces(
        model,
        metric,
        generator_runs_dict,
        rel,
        show_CI=show_CI,
        data_selector=data_selector,
    )

    # order arm name sorting arm numbers within batch
    names_by_arm = sorted(
        np.unique(np.concatenate([d["x"] for d in traces])),
        key=lambda x: arm_name_to_tuple(x),
    )

    # get arm names sorted by effect size
    names_by_effect = list(
        OrderedDict.fromkeys(
            np.concatenate([d["x"] for d in traces])
            .flatten()
            .take(np.argsort(np.concatenate([d["y"] for d in traces]).flatten()))
        )
    )

    # options for ordering arms (x-axis)
    xaxis_categoryorder = "array"
    xaxis_categoryarray = names_by_arm

    order_options = [
        {
            "args": [
                {"xaxis.categoryorder": "array", "xaxis.categoryarray": names_by_arm}
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
    if custom_arm_order is not None:
        xaxis_categoryorder = "array"
        xaxis_categoryarray = custom_arm_order
        order_options = [
            {
                "args": [
                    {
                        "xaxis.categoryorder": "array",
                        "xaxis.categoryarray": custom_arm_order,
                    }
                ],
                "label": custom_arm_order_name,
                "method": "relayout",
            }
            # Union[List[str...
        ] + order_options

    layout = go.Layout(
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

    fig = go.Figure(data=traces, layout=layout)
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)


def tile_fitted(
    model: ModelBridge,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel: bool = True,
    show_arm_details_on_hover: bool = False,
    show_CI: bool = True,
    arm_noun: str = "arm",
    metrics: Optional[List[str]] = None,
    fixed_features: Optional[ObservationFeatures] = None,
    data_selector: Optional[Callable[[Observation], bool]] = None,
) -> AxPlotConfig:
    """Tile version of fitted outcome plots.

    Args:
        model: model to use for predictions.
        generator_runs_dict: a mapping from
            generator run name to generator run.
        rel: if True, use relative effects. Default is True.
        show_arm_details_on_hover: if True, display
            parameterizations of arms on hover. Default is False.
        show_CI: if True, render confidence intervals.
        arm_noun: noun to use instead of "arm" (e.g. group)
        metrics: List of metric names to restrict to when plotting.
        fixed_features: Fixed features to use when making model predictions.
        data_selector: Function for selecting observations for plotting.
    """
    metrics = metrics or list(model.metric_names)
    nrows = int(np.ceil(len(metrics) / 2))
    ncols = min(len(metrics), 2)

    # make subplots (plot per row)
    fig = subplots.make_subplots(
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
            model,
            metric,
            generator_runs_dict,
            rel,
            showlegend=i == 0,
            show_arm_details_on_hover=show_arm_details_on_hover,
            show_CI=show_CI,
            arm_noun=arm_noun,
            fixed_features=fixed_features,
            data_selector=data_selector,
        )

        # order arm name sorting arm numbers within batch
        names_by_arm = sorted(
            np.unique(np.concatenate([d["x"] for d in data])),
            key=lambda x: arm_name_to_tuple(x),
        )

        # get arm names sorted by effect size
        names_by_effect = list(
            OrderedDict.fromkeys(
                np.concatenate([d["x"] for d in data])
                .flatten()
                .take(np.argsort(np.concatenate([d["y"] for d in data]).flatten()))
            )
        )

        # options for ordering arms (x-axis)
        # Note that xaxes need to be references as xaxis, xaxis2, xaxis3, etc.
        # for the purposes of updatemenus argument (dropdown) in layout.
        # However, when setting the initial ordering layout, the keys should be
        # xaxis1, xaxis2, xaxis3, etc. Note the discrepancy for the initial
        # axis.
        label = "" if i == 0 else i + 1
        name_order_args["xaxis{}.categoryorder".format(label)] = "array"
        name_order_args["xaxis{}.categoryarray".format(label)] = names_by_arm
        effect_order_args["xaxis{}.categoryorder".format(label)] = "array"
        effect_order_args["xaxis{}.categoryarray".format(label)] = names_by_effect
        name_order_axes["xaxis{}".format(i + 1)] = {
            "categoryorder": "array",
            "categoryarray": names_by_arm,
            "type": "category",
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
    # generated by `subplots.make_subplots`
    if len(metrics) % 2 == 1:
        fig["layout"].pop("xaxis{}".format(nrows * ncols))
        fig["layout"].pop("yaxis{}".format(nrows * ncols))

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
        **name_order_axes,
    )

    # append dropdown annotations
    fig["layout"]["annotations"] = fig["layout"]["annotations"] + (
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
    )

    fig = resize_subtitles(figure=fig, size=10)
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)


def interact_fitted_plotly(
    model: ModelBridge,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel: bool = True,
    show_arm_details_on_hover: bool = True,
    show_CI: bool = True,
    arm_noun: str = "arm",
    metrics: Optional[List[str]] = None,
    fixed_features: Optional[ObservationFeatures] = None,
    data_selector: Optional[Callable[[Observation], bool]] = None,
    label_dict: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """Interactive fitted outcome plots for each arm used in fitting the model.

    Choose the outcome to plot using a dropdown.

    Args:
        model: model to use for predictions.
        generator_runs_dict: a mapping from
            generator run name to generator run.
        rel: if True, use relative effects. Default is True.
        show_arm_details_on_hover: if True, display
            parameterizations of arms on hover. Default is True.
        show_CI: if True, render confidence intervals.
        arm_noun: noun to use instead of "arm" (e.g. group)
        metrics: List of metric names to restrict to when plotting.
        fixed_features: Fixed features to use when making model predictions.
        data_selector: Function for selecting observations for plotting.
        label_dict: A dictionary that maps the label to
            an alias to be used in the plot.
    """
    traces_per_metric = (
        1 if generator_runs_dict is None else len(generator_runs_dict) + 1
    )
    metrics = sorted(metrics or model.metric_names)
    if not label_dict:
        _check_label_lengths(metrics)

    traces = []
    dropdown = []

    for i, metric in enumerate(metrics):
        data = _single_metric_traces(
            model,
            metric,
            generator_runs_dict,
            rel,
            showlegend=i == 0,
            show_arm_details_on_hover=show_arm_details_on_hover,
            show_CI=show_CI,
            arm_noun=arm_noun,
            fixed_features=fixed_features,
            data_selector=data_selector,
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
            {
                "args": ["visible", is_visible],
                "label": _replace_str(metric, label_dict),
                "method": "restyle",
            }
        )

    layout = go.Layout(
        xaxis={"title": arm_noun.title(), "zeroline": False, "type": "category"},
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
        layout["annotations"] = layout["annotations"] + (
            {
                "font": {"size": 12},
                "showarrow": False,
                "text": "Arm Source",
                "x": 0.05,
                "xanchor": "right",
                "xref": "paper",
                "y": 1.2,
                "yanchor": "middle",
                "yref": "paper",
            },
        )

    return go.Figure(data=traces, layout=layout)


def interact_fitted(
    model: ModelBridge,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel: bool = True,
    show_arm_details_on_hover: bool = True,
    show_CI: bool = True,
    arm_noun: str = "arm",
    metrics: Optional[List[str]] = None,
    fixed_features: Optional[ObservationFeatures] = None,
    data_selector: Optional[Callable[[Observation], bool]] = None,
    label_dict: Optional[Dict[str, str]] = None,
) -> AxPlotConfig:
    """Interactive fitted outcome plots for each arm used in fitting the model.

    Choose the outcome to plot using a dropdown.

    Args:
        model: model to use for predictions.
        generator_runs_dict: a mapping from
            generator run name to generator run.
        rel: if True, use relative effects. Default is True.
        show_arm_details_on_hover: if True, display
            parameterizations of arms on hover. Default is True.
        show_CI: if True, render confidence intervals.
        arm_noun: noun to use instead of "arm" (e.g. group)
        metrics: List of metric names to restrict to when plotting.
        fixed_features: Fixed features to use when making model predictions.
        data_selector: Function for selecting observations for plotting.
        label_dict: A dictionary that maps the label to
            an alias to be used in the plot.
    """

    return AxPlotConfig(
        data=interact_fitted_plotly(
            model=model,
            generator_runs_dict=generator_runs_dict,
            rel=rel,
            show_arm_details_on_hover=show_arm_details_on_hover,
            show_CI=show_CI,
            arm_noun=arm_noun,
            metrics=metrics,
            fixed_features=fixed_features,
            data_selector=data_selector,
            label_dict=label_dict,
        ),
        plot_type=AxPlotTypes.GENERIC,
    )


def tile_observations(
    experiment: Experiment,
    data: Optional[Data] = None,
    rel: bool = True,
    metrics: Optional[List[str]] = None,
    arm_names: Optional[List[str]] = None,
) -> AxPlotConfig:
    """
    Tiled plot with all observed outcomes.

    Will plot all observed arms. If data is provided will use that, otherwise
    will fetch data from experiment. Will plot all metrics in data unless a
    list is provided in metrics. If arm_names is provided will limit the plot
    to only arms in that list.

    Args:
        experiment: Experiment
        data: Data to use, otherwise will fetch data from experiment.
        rel: Plot relative values, if experiment has status quo.
        metrics: Limit results to this set of metrics.
        arm_names: Limit results to this set of arms.

    Returns: Plot config for the plot.
    """
    if data is None:
        data = experiment.fetch_data()
    if arm_names is not None:
        data = Data(data.df[data.df["arm_name"].isin(arm_names)])
    m_ts = Models.THOMPSON(
        data=data,
        search_space=experiment.search_space,
        experiment=experiment,
    )
    return tile_fitted(
        model=m_ts,
        rel=rel and (experiment.status_quo is not None),
        metrics=metrics,
    )

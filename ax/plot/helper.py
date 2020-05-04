#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, FixedParameter, RangeParameter
from ax.core.types import TParameterization
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.transforms.ivw import IVW
from ax.plot.base import DECIMALS, PlotData, PlotInSampleArm, PlotOutOfSampleArm, Z
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none


logger = get_logger(name="PlotHelper")

# Typing alias
RawData = List[Dict[str, Union[str, float]]]

TNullableGeneratorRunsDict = Optional[Dict[str, GeneratorRun]]


def _format_dict(param_dict: TParameterization, name: str = "Parameterization") -> str:
    """Format a dictionary for labels.

    Args:
        param_dict: Dictionary to be formatted
        name: String name of the thing being formatted.

    Returns: stringified blob.
    """
    if len(param_dict) >= 10:
        blob = "{} has too many items to render on hover ({}).".format(
            name, len(param_dict)
        )
    else:
        blob = "<br><em>{}:</em><br>{}".format(
            name, "<br>".join("{}: {}".format(n, v) for n, v in param_dict.items())
        )
    return blob


def _wrap_metric(metric_name: str) -> str:
    """Put a newline on "::" for metric names.

    Args:
        metric_name: metric name.

    Returns: wrapped metric name.
    """
    if "::" in metric_name:
        return "<br>".join(metric_name.split("::"))
    else:
        return metric_name


def _format_CI(estimate: float, sd: float, relative: bool, zval: float = Z) -> str:
    """Format confidence intervals given estimate and standard deviation.

    Args:
        estimate: point estimate.
        sd: standard deviation of point estimate.
        relative: if True, '%' is appended.
        zval: z-value associated with desired CI (e.g. 1.96 for 95% CIs)

    Returns: formatted confidence interval.
    """
    return "[{lb:.{digits}f}{perc}, {ub:.{digits}f}{perc}]".format(
        lb=estimate - zval * sd,
        ub=estimate + zval * sd,
        digits=DECIMALS,
        perc="%" if relative else "",
    )


def arm_name_to_tuple(arm_name: str) -> Union[Tuple[int, int], Tuple[int]]:
    tup = arm_name.split("_")
    if len(tup) == 2:
        try:
            return (int(tup[0]), int(tup[1]))
        except ValueError:
            return (0,)
    return (0,)


def resize_subtitles(figure: Dict[str, Any], size: int):
    for ant in figure["layout"]["annotations"]:
        ant["font"].update(size=size)
    return figure


def _filter_dict(
    param_dict: TParameterization, subset_keys: List[str]
) -> TParameterization:
    """Filter a dictionary to keys present in a given list."""
    return {k: v for k, v in param_dict.items() if k in subset_keys}


def _get_in_sample_arms(
    model: ModelBridge,
    metric_names: Set[str],
    fixed_features: Optional[ObservationFeatures] = None,
) -> Tuple[Dict[str, PlotInSampleArm], RawData, Dict[str, TParameterization]]:
    """Get in-sample arms from a model with observed and predicted values
    for specified metrics.

    Returns a PlotInSampleArm object in which repeated observations are merged
    with IVW, and a RawData object in which every observation is listed.

    Fixed features input can be used to override fields of the insample arms
    when making model predictions.

    Args:
        model: An instance of the model bridge.
        metric_names: Restrict predictions to these metrics. If None, uses all
            metrics in the model.
        fixed_features: Features that should be fixed in the arms this function
            will obtain predictions for.

    Returns:
        A tuple containing

        - Map from arm name to PlotInSampleArm.
        - List of the data for each observation like::

            {'metric_name': 'likes', 'arm_name': '0_0', 'mean': 1., 'sem': 0.1}

        - Map from arm name to parameters
    """
    observations = model.get_training_data()
    # Calculate raw data
    raw_data = []
    arm_name_to_parameters = {}
    for obs in observations:
        arm_name_to_parameters[obs.arm_name] = obs.features.parameters
        for j, metric_name in enumerate(obs.data.metric_names):
            if metric_name in metric_names:
                raw_data.append(
                    {
                        "metric_name": metric_name,
                        "arm_name": obs.arm_name,
                        "mean": obs.data.means[j],
                        "sem": np.sqrt(obs.data.covariance[j, j]),
                    }
                )

    # Check that we have one ObservationFeatures per arm name since we
    # key by arm name and the model is not Multi-task.
    # If "TrialAsTask" is present, one of the arms is also chosen.
    if ("TrialAsTask" not in model.transforms.keys()) and (
        len(arm_name_to_parameters) != len(observations)
    ):
        logger.error(
            "Have observations of arms with different features but same"
            " name. Arbitrary one will be plotted."
        )

    # Merge multiple measurements within each Observation with IVW to get
    # un-modeled prediction
    t = IVW(None, [], [])
    obs_data = t.transform_observation_data([obs.data for obs in observations], [])
    # Start filling in plot data
    in_sample_plot: Dict[str, PlotInSampleArm] = {}
    for i, obs in enumerate(observations):
        if obs.arm_name is None:
            raise ValueError("Observation must have arm name for plotting.")

        # Extract raw measurement
        obs_y = {}  # Observed metric means.
        obs_se = {}  # Observed metric standard errors.
        # Use the IVW data, not obs.data
        for j, metric_name in enumerate(obs_data[i].metric_names):
            if metric_name in metric_names:
                obs_y[metric_name] = obs_data[i].means[j]
                obs_se[metric_name] = np.sqrt(obs_data[i].covariance[j, j])
        # Make a prediction.
        if model.training_in_design[i]:
            features = obs.features
            if fixed_features is not None:
                features.update_features(fixed_features)
            pred_y, pred_se = _predict_at_point(model, features, metric_names)
        else:
            # Use raw data for out-of-design points
            pred_y = obs_y
            pred_se = obs_se
        in_sample_plot[not_none(obs.arm_name)] = PlotInSampleArm(
            name=not_none(obs.arm_name),
            y=obs_y,
            se=obs_se,
            parameters=obs.features.parameters,
            y_hat=pred_y,
            se_hat=pred_se,
            context_stratum=None,
        )
    return in_sample_plot, raw_data, arm_name_to_parameters


def _predict_at_point(
    model: ModelBridge, obsf: ObservationFeatures, metric_names: Set[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Make a prediction at a point.

    Returns mean and standard deviation in format expected by plotting.

    Args:
        model: ModelBridge
        obsf: ObservationFeatures for which to predict
        metric_names: Limit predictions to these metrics.

    Returns:
        A tuple containing

        - Map from metric name to prediction.
        - Map from metric name to standard error.
    """
    y_hat = {}
    se_hat = {}
    f_pred, cov_pred = model.predict([obsf])
    for metric_name in f_pred:
        if metric_name in metric_names:
            y_hat[metric_name] = f_pred[metric_name][0]
            se_hat[metric_name] = np.sqrt(cov_pred[metric_name][metric_name][0])
    return y_hat, se_hat


def _get_out_of_sample_arms(
    model: ModelBridge,
    generator_runs_dict: Dict[str, GeneratorRun],
    metric_names: Set[str],
    fixed_features: Optional[ObservationFeatures] = None,
) -> Dict[str, Dict[str, PlotOutOfSampleArm]]:
    """Get out-of-sample predictions from a model given a dict of generator runs.

    Fixed features input can be used to override fields of the candidate arms
    when making model predictions.

    Args:
        model: The model.
        generator_runs_dict: a mapping from generator run name to generator run.
        metric_names: metrics to include in the plot.

    Returns:
        A mapping from name to a mapping from arm name to plot.

    """
    out_of_sample_plot: Dict[str, Dict[str, PlotOutOfSampleArm]] = {}
    for generator_run_name, generator_run in generator_runs_dict.items():
        out_of_sample_plot[generator_run_name] = {}
        for arm in generator_run.arms:
            # This assumes context is None
            obsf = ObservationFeatures.from_arm(arm)
            if fixed_features is not None:
                obsf.update_features(fixed_features)

            # Make a prediction
            try:
                pred_y, pred_se = _predict_at_point(model, obsf, metric_names)
            except Exception:
                # Check if it is an out-of-design arm.
                if not model.model_space.check_membership(obsf.parameters):
                    # Skip this point
                    continue
                else:
                    # It should have worked
                    raise
            arm_name = arm.name_or_short_signature
            out_of_sample_plot[generator_run_name][arm_name] = PlotOutOfSampleArm(
                name=arm_name,
                parameters=obsf.parameters,
                y_hat=pred_y,
                se_hat=pred_se,
                context_stratum=None,
            )
    return out_of_sample_plot


def get_plot_data(
    model: ModelBridge,
    generator_runs_dict: Dict[str, GeneratorRun],
    metric_names: Optional[Set[str]] = None,
    fixed_features: Optional[ObservationFeatures] = None,
) -> Tuple[PlotData, RawData, Dict[str, TParameterization]]:
    """Format data object with metrics for in-sample and out-of-sample
    arms.

    Calculate both observed and predicted metrics for in-sample arms.
    Calculate predicted metrics for out-of-sample arms passed via the
    `generator_runs_dict` argument.

    In PlotData, in-sample observations are merged with IVW. In RawData, they
    are left un-merged and given as a list of dictionaries, one for each
    observation and having keys 'arm_name', 'mean', and 'sem'.

    Args:
        model: The model.
        generator_runs_dict: a mapping from generator run name to generator run.
        metric_names: Restrict predictions to this set. If None, all metrics
            in the model will be returned.
        fixed_features: Fixed features to use when making model predictions.

    Returns:
        A tuple containing

        - PlotData object with in-sample and out-of-sample predictions.
        - List of observations like::

            {'metric_name': 'likes', 'arm_name': '0_1', 'mean': 1., 'sem': 0.1}.

        - Mapping from arm name to parameters.
    """
    metrics_plot = model.metric_names if metric_names is None else metric_names
    in_sample_plot, raw_data, cond_name_to_parameters = _get_in_sample_arms(
        model=model, metric_names=metrics_plot, fixed_features=fixed_features
    )
    out_of_sample_plot = _get_out_of_sample_arms(
        model=model,
        generator_runs_dict=generator_runs_dict,
        metric_names=metrics_plot,
        fixed_features=fixed_features,
    )
    # pyre-fixme[16]: `Optional` has no attribute `arm_name`.
    status_quo_name = None if model.status_quo is None else model.status_quo.arm_name
    plot_data = PlotData(
        metrics=list(metrics_plot),
        in_sample=in_sample_plot,
        out_of_sample=out_of_sample_plot,
        status_quo_name=status_quo_name,
    )
    return plot_data, raw_data, cond_name_to_parameters


def get_range_parameter(model: ModelBridge, param_name: str) -> RangeParameter:
    """
    Get the range parameter with the given name from the model.

    Throws if parameter doesn't exist or is not a range parameter.

    Args:
        model: The model.
        param_name: The name of the RangeParameter to be found.

    Returns: The RangeParameter named `param_name`.
    """

    range_param = model.model_space.parameters.get(param_name)
    if range_param is None:
        raise ValueError(f"Parameter `{param_name}` does not exist.")
    if not isinstance(range_param, RangeParameter):
        raise ValueError(f"{param_name} is not a RangeParameter")

    return range_param


def get_range_parameters(model: ModelBridge) -> List[RangeParameter]:
    """
    Get a list of range parameters from a model.

    Args:
        model: The model.

    Returns: List of RangeParameters.
    """
    return [
        parameter
        for parameter in model.model_space.parameters.values()
        if isinstance(parameter, RangeParameter)
    ]


def get_grid_for_parameter(parameter: RangeParameter, density: int) -> np.ndarray:
    """Get a grid of points along the range of the parameter.

    Will be a log-scale grid if parameter is log scale.

    Args:
        parameter: Parameter for which to generate grid.
        density: Number of points in the grid.
    """
    is_log = parameter.log_scale
    if is_log:
        grid = np.linspace(
            np.log10(parameter.lower), np.log10(parameter.upper), density
        )
        grid = 10 ** grid
    else:
        grid = np.linspace(parameter.lower, parameter.upper, density)
    return grid


def get_fixed_values(
    model: ModelBridge,
    slice_values: Optional[Dict[str, Any]] = None,
    trial_index: Optional[int] = None,
) -> TParameterization:
    """Get fixed values for parameters in a slice plot.

    If there is an in-design status quo, those values will be used. Otherwise,
    the mean of RangeParameters or the mode of ChoiceParameters is used.

    Any value in slice_values will override the above.

    Args:
        model: ModelBridge being used for plotting
        slice_values: Map from parameter name to value at which is should be
            fixed.

    Returns: Map from parameter name to fixed value.
    """

    if trial_index is not None:
        if slice_values is None:
            slice_values = {}
        slice_values["TRIAL_PARAM"] = str(trial_index)

    # Check if status_quo is in design
    if model.status_quo is not None and model.model_space.check_membership(
        # pyre-fixme[16]: `Optional` has no attribute `features`.
        model.status_quo.features.parameters
    ):
        setx = model.status_quo.features.parameters
    else:
        observations = model.get_training_data()
        setx = {}
        for p_name, parameter in model.model_space.parameters.items():
            # Exclude out of design status quo (no parameters)
            vals = [
                obs.features.parameters[p_name]
                for obs in observations
                if (
                    len(obs.features.parameters) > 0
                    and parameter.validate(obs.features.parameters[p_name])
                )
            ]
            if isinstance(parameter, FixedParameter):
                setx[p_name] = parameter.value
            elif isinstance(parameter, ChoiceParameter):
                setx[p_name] = Counter(vals).most_common(1)[0][0]
            elif isinstance(parameter, RangeParameter):
                setx[p_name] = parameter.cast(np.mean(vals))

    if slice_values is not None:
        # slice_values has type Dictionary[str, Any]
        setx.update(slice_values)
    return setx


# Utility methods ported from JS
def contour_config_to_trace(config):
    # Load from config
    arm_data = config["arm_data"]
    density = config["density"]
    grid_x = config["grid_x"]
    grid_y = config["grid_y"]
    f = config["f"]
    lower_is_better = config["lower_is_better"]
    metric = config["metric"]
    rel = config["rel"]
    sd = config["sd"]
    xvar = config["xvar"]
    yvar = config["yvar"]

    green_scale = config["green_scale"]
    green_pink_scale = config["green_pink_scale"]
    blue_scale = config["blue_scale"]

    # format data
    res = relativize_data(f, sd, rel, arm_data, metric)
    f_final = res[0]
    sd_final = res[1]

    # calculate max of abs(outcome), used for colorscale
    f_absmax = max(abs(min(f_final)), max(f_final))

    # transform to nested array
    f_plt = []
    for ind in range(0, len(f_final), density):
        f_plt.append(f_final[ind : ind + density])
    sd_plt = []
    for ind in range(0, len(sd_final), density):
        sd_plt.append(sd_final[ind : ind + density])

    CONTOUR_CONFIG = {
        "autocolorscale": False,
        "autocontour": True,
        "contours": {"coloring": "heatmap"},
        "hoverinfo": "x+y+z",
        "ncontours": int(density / 2),
        "type": "contour",
        "x": grid_x,
        "y": grid_y,
    }

    if rel:
        f_scale = reversed(green_pink_scale) if lower_is_better else green_pink_scale
    else:
        f_scale = green_scale

    f_trace = {
        "colorbar": {
            "x": 0.45,
            "y": 0.5,
            "ticksuffix": "%" if rel else "",
            "tickfont": {"size": 8},
        },
        "colorscale": [(i / (len(f_scale) - 1), rgb(v)) for i, v in enumerate(f_scale)],
        "xaxis": "x",
        "yaxis": "y",
        "z": f_plt,
        # zmax and zmin are ignored if zauto is true
        "zauto": not rel,
        "zmax": f_absmax,
        "zmin": -f_absmax,
    }

    sd_trace = {
        "colorbar": {
            "x": 1,
            "y": 0.5,
            "ticksuffix": "%" if rel else "",
            "tickfont": {"size": 8},
        },
        "colorscale": [
            (i / (len(blue_scale) - 1), rgb(v)) for i, v in enumerate(blue_scale)
        ],
        "xaxis": "x2",
        "yaxis": "y2",
        "z": sd_plt,
    }

    f_trace.update(CONTOUR_CONFIG)
    sd_trace.update(CONTOUR_CONFIG)

    # get in-sample arms
    arm_text = list(arm_data["in_sample"].keys())
    arm_x = [
        arm_data["in_sample"][arm_name]["parameters"][xvar] for arm_name in arm_text
    ]
    arm_y = [
        arm_data["in_sample"][arm_name]["parameters"][yvar] for arm_name in arm_text
    ]

    # configs for in-sample arms
    base_in_sample_arm_config = {
        "hoverinfo": "text",
        "legendgroup": "In-sample",
        "marker": {"color": "black", "symbol": 1, "opacity": 0.5},
        "mode": "markers",
        "name": "In-sample",
        "text": arm_text,
        "type": "scatter",
        "x": arm_x,
        "y": arm_y,
    }

    f_in_sample_arm_trace = {"xaxis": "x", "yaxis": "y"}

    sd_in_sample_arm_trace = {"showlegend": False, "xaxis": "x2", "yaxis": "y2"}

    f_in_sample_arm_trace.update(base_in_sample_arm_config)
    sd_in_sample_arm_trace.update(base_in_sample_arm_config)

    traces = [f_trace, sd_trace, f_in_sample_arm_trace, sd_in_sample_arm_trace]

    # iterate over out-of-sample arms
    for i, generator_run_name in enumerate(arm_data["out_of_sample"].keys()):
        symbol = i + 2  # symbols starts from 2 for candidate markers

        ax = []
        ay = []
        atext = []

        for arm_name in arm_data["out_of_sample"][generator_run_name].keys():
            ax.append(
                arm_data["out_of_sample"][generator_run_name][arm_name]["parameters"][
                    xvar
                ]
            )
            ay.append(
                arm_data["out_of_sample"][generator_run_name][arm_name]["parameters"][
                    yvar
                ]
            )
            atext.append("<em>Candidate " + arm_name + "</em>")

        traces.append(
            {
                "hoverinfo": "text",
                "legendgroup": generator_run_name,
                "marker": {"color": "black", "symbol": symbol, "opacity": 0.5},
                "mode": "markers",
                "name": generator_run_name,
                "text": atext,
                "type": "scatter",
                "xaxis": "x",
                "x": ax,
                "yaxis": "y",
                "y": ay,
            }
        )
        traces.append(
            {
                "hoverinfo": "text",
                "legendgroup": generator_run_name,
                "marker": {"color": "black", "symbol": symbol, "opacity": 0.5},
                "mode": "markers",
                "name": "In-sample",
                "showlegend": False,
                "text": atext,
                "type": "scatter",
                "x": ax,
                "xaxis": "x2",
                "y": ay,
                "yaxis": "y2",
            }
        )

    return traces


def axis_range(grid: List[float], is_log: bool) -> List[float]:
    if is_log:
        return [math.log10(min(grid)), math.log10(max(grid))]
    else:
        return [min(grid), max(grid)]


def relativize(m_t: float, sem_t: float, m_c: float, sem_c: float) -> List[float]:
    r_hat = (m_t - m_c) / abs(m_c) - sem_c ** 2 * m_t / abs(m_c) ** 3
    variance = (sem_t ** 2 + (m_t / m_c * sem_c) ** 2) / m_c ** 2
    return [r_hat, math.sqrt(variance)]


def relativize_data(
    f: List[float], sd: List[float], rel: bool, arm_data: Dict[Any, Any], metric: str
) -> List[List[float]]:
    # if relative, extract status quo & compute ratio
    f_final = [] if rel else f
    sd_final = [] if rel else sd

    if rel:
        f_sq = arm_data["in_sample"][arm_data["status_quo_name"]]["y"][metric]
        sd_sq = arm_data["in_sample"][arm_data["status_quo_name"]]["se"][metric]

        for i in range(len(f)):
            res = relativize(f[i], sd[i], f_sq, sd_sq)
            f_final.append(100 * res[0])
            sd_final.append(100 * res[1])

    return [f_final, sd_final]


def rgb(arr: List[int]) -> str:
    return "rgb({},{},{})".format(*arr)


def infer_is_relative(
    model: ModelBridge, metrics: List[str], non_constraint_rel: bool
) -> Dict[str, bool]:
    """Determine whether or not to relativize a metric.

    Metrics that are constraints will get this decision from their `relative` flag.
    Other metrics will use the `default_rel`.

    Args:
        model: model fit on metrics.
        metrics: list of metric names.
        non_constraint_rel: whether or not to relativize non-constraint metrics

    Returns:
        Dict[str, bool] containing whether or not to relativize each input metric.
    """
    relative = {}
    constraint_relativity = {}
    if model._optimization_config:
        constraints = not_none(model._optimization_config).outcome_constraints
        constraint_relativity = {
            constraint.metric.name: constraint.relative for constraint in constraints
        }
    for metric in metrics:
        if metric not in constraint_relativity:
            relative[metric] = non_constraint_rel
        else:
            relative[metric] = constraint_relativity[metric]
    return relative


def slice_config_to_trace(
    arm_data,
    arm_name_to_parameters,
    f,
    fit_data,
    grid,
    metric,
    param,
    rel,
    setx,
    sd,
    is_log,
    visible,
):
    # format data
    res = relativize_data(f, sd, rel, arm_data, metric)
    f_final = res[0]
    sd_final = res[1]

    # get data for standard deviation fill plot
    sd_upper = []
    sd_lower = []
    for i in range(len(sd)):
        sd_upper.append(f_final[i] + 2 * sd_final[i])
        sd_lower.append(f_final[i] - 2 * sd_final[i])
    grid_rev = list(reversed(grid))
    sd_lower_rev = list(reversed(sd_lower))
    sd_x = grid + grid_rev
    sd_y = sd_upper + sd_lower_rev

    # get data for observed arms and error bars
    arm_x = []
    arm_y = []
    arm_sem = []
    for row in fit_data:
        parameters = arm_name_to_parameters[row["arm_name"]]
        plot = True
        for p in setx.keys():
            if p != param and parameters[p] != setx[p]:
                plot = False
        if plot:
            arm_x.append(parameters[param])
            arm_y.append(row["mean"])
            arm_sem.append(row["sem"])

    arm_res = relativize_data(arm_y, arm_sem, rel, arm_data, metric)
    arm_y_final = arm_res[0]
    arm_sem_final = [x * 2 for x in arm_res[1]]

    # create traces
    f_trace = {
        "x": grid,
        "y": f_final,
        "showlegend": False,
        "hoverinfo": "x+y",
        "line": {"color": "rgba(128, 177, 211, 1)"},
        "visible": visible,
    }

    arms_trace = {
        "x": arm_x,
        "y": arm_y_final,
        "mode": "markers",
        "error_y": {
            "type": "data",
            "array": arm_sem_final,
            "visible": True,
            "color": "black",
        },
        "line": {"color": "black"},
        "showlegend": False,
        "hoverinfo": "x+y",
        "visible": visible,
    }

    sd_trace = {
        "x": sd_x,
        "y": sd_y,
        "fill": "toself",
        "fillcolor": "rgba(128, 177, 211, 0.2)",
        "line": {"color": "rgba(128, 177, 211, 0.0)"},
        "showlegend": False,
        "hoverinfo": "none",
        "visible": visible,
    }

    traces = [sd_trace, f_trace, arms_trace]

    # iterate over out-of-sample arms
    for i, generator_run_name in enumerate(arm_data["out_of_sample"].keys()):
        ax = []
        ay = []
        asem = []
        atext = []

        for arm_name in arm_data["out_of_sample"][generator_run_name].keys():
            parameters = arm_data["out_of_sample"][generator_run_name][arm_name][
                "parameters"
            ]
            plot = True
            for p in setx.keys():
                if p != param and parameters[p] != setx[p]:
                    plot = False
            if plot:
                ax.append(parameters[param])
                ay.append(
                    arm_data["out_of_sample"][generator_run_name][arm_name]["y_hat"][
                        metric
                    ]
                )
                asem.append(
                    arm_data["out_of_sample"][generator_run_name][arm_name]["se_hat"][
                        metric
                    ]
                )
                atext.append("<em>Candidate " + arm_name + "</em>")

        out_of_sample_arm_res = relativize_data(ay, asem, rel, arm_data, metric)
        ay_final = out_of_sample_arm_res[0]
        asem_final = [x * 2 for x in out_of_sample_arm_res[1]]

        traces.append(
            {
                "hoverinfo": "text",
                "legendgroup": generator_run_name,
                "marker": {"color": "black", "symbol": i + 1, "opacity": 0.5},
                "mode": "markers",
                "error_y": {
                    "type": "data",
                    "array": asem_final,
                    "visible": True,
                    "color": "black",
                },
                "name": generator_run_name,
                "text": atext,
                "type": "scatter",
                "xaxis": "x",
                "x": ax,
                "yaxis": "y",
                "y": ay_final,
                "visible": visible,
            }
        )

    return traces

#!/usr/bin/env python3

from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from ae.lazarus.ae.core.generator_run import GeneratorRun
from ae.lazarus.ae.core.observation import ObservationFeatures
from ae.lazarus.ae.core.parameter import ChoiceParameter, FixedParameter, RangeParameter
from ae.lazarus.ae.core.types.types import TParameterization
from ae.lazarus.ae.generator.base import Generator
from ae.lazarus.ae.generator.transforms.ivw import IVW
from ae.lazarus.ae.plot.base import (
    DECIMALS,
    PlotData,
    PlotInSampleCondition,
    PlotOutOfSampleCondition,
    Z,
)
from ae.lazarus.ae.utils.common.logger import get_logger


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


def condition_name_to_tuple(condition_name: str) -> Union[Tuple[int, int], Tuple[int]]:
    tup = condition_name.split("_")
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


def _get_in_sample_conditions(
    generator: Generator, metric_names: Set[str]
) -> Tuple[Dict[str, PlotInSampleCondition], RawData, Dict[str, TParameterization]]:
    """Get in-sample conditions from a generator with observed and predicted values
    for specified metrics.

    Returns a PlotInSampleCondition object in which repeated observations are merged
    with IVW, and a RawData object in which every observation is listed.

    Args:
        generator: An instance of the generator.
        metric_names: Restrict predictions to these metrics. If None, uses all
            metrics in the generator.

    Returns:
        in_sample_plot: A map from condition name to PlotInSampleCondition
        raw_data: A list of the data for each observation like
            {'metric_name': 'likes', 'condition_name': '0_0', 'mean': 1., 'sem': 0.1}.
        cond_name_to_params: A mapping from condition name to parameters
    """
    observations = generator.get_training_data()
    # Calculate raw data
    raw_data = []
    cond_name_to_params = {}
    for obs in observations:
        cond_name_to_params[obs.condition_name] = obs.features.parameters
        for j, metric_name in enumerate(obs.data.metric_names):
            raw_data.append(
                {
                    "metric_name": metric_name,
                    "condition_name": obs.condition_name,
                    "mean": obs.data.means[j],
                    "sem": np.sqrt(obs.data.covariance[j, j]),
                }
            )
    # Check that we have one ObservationFeatures per condition name since we
    # key by condition name.
    if len(cond_name_to_params) != len(observations):
        logger.error(
            "Have observations of conditions with different features but same"
            " name. Arbitrary one will be plotted."
        )
    # Merge multiple measurements within each Observation with IVW to get
    # un-modeled prediction
    t = IVW(None, [], [])
    obs_data = t.transform_observation_data([obs.data for obs in observations], [])
    # Start filling in plot data
    in_sample_plot: Dict[str, PlotInSampleCondition] = {}
    for i, obs in enumerate(observations):
        if obs.condition_name is None:
            raise ValueError("Observation must have condition name for plotting.")

        # Extract raw measurement
        obs_y = {}
        obs_se = {}
        # Use the IVW data, not obs.data
        for j, metric_name in enumerate(obs_data[i].metric_names):
            if metric_name in metric_names:
                obs_y[metric_name] = obs_data[i].means[j]
                obs_se[metric_name] = np.sqrt(obs_data[i].covariance[j, j])
        # Make a prediction.
        if generator.training_in_design[i]:
            pred_y, pred_se = _predict_at_point(generator, obs.features, metric_names)
        else:
            # Use raw data for out-of-design points
            pred_y = obs_y
            pred_se = obs_se
        in_sample_plot[obs.condition_name] = PlotInSampleCondition(
            name=obs.condition_name,
            y=obs_y,
            se=obs_se,
            params=obs.features.parameters,
            y_hat=pred_y,
            se_hat=pred_se,
            context_stratum=None,
        )
    # pyre: Expected `typing.Tuple[Dict[str, PlotInSampleCondition],
    # pyre: List[Dict[str, Union[float, str]]], Dict[str, Dict[str,
    # pyre: Optional[Union[bool, float, str]]]]]` but got `typing.
    # pyre: Tuple[Dict[str, PlotInSampleCondition], List[],
    # pyre: Dict[Optional[str], Dict[str, Optional[Union[bool, float,
    # pyre-fixme[7]: str]]]]]`.
    return in_sample_plot, raw_data, cond_name_to_params


def _predict_at_point(
    generator: Generator, obsf: ObservationFeatures, metric_names: Set[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Make a prediction at a point.

    Returns mean and standard deviation in format expected by plotting.

    Args:
        generator: Generator
        obsf: ObservationFeatures for which to predict
        metric_names: Limit predictions to these metrics.

    Returns
        y_hat: map from metric name to prediction
        se_hat: map from metric name to standard error.
    """
    y_hat = {}
    se_hat = {}
    f_pred, cov_pred = generator.predict([obsf])
    for metric_name in f_pred:
        if metric_name in metric_names:
            y_hat[metric_name] = f_pred[metric_name][0]
            se_hat[metric_name] = np.sqrt(cov_pred[metric_name][metric_name][0])
    return y_hat, se_hat


def _get_out_of_sample_conditions(
    generator: Generator,
    generator_runs_dict: Dict[str, GeneratorRun],
    metric_names: Set[str],
) -> Dict[str, Dict[str, PlotOutOfSampleCondition]]:
    """Get out-of-sample predictions from a generator given a dict of generator runs.

    Args:
        generator: The generator.
        generator_runs_dict: a mapping from generator run name to generator run.
        metric_names: metrics to include in the plot.

    Returns:
        A mapping from name to a mapping from condition name to plot.

    """
    out_of_sample_plot: Dict[str, Dict[str, PlotOutOfSampleCondition]] = {}
    for generator_run_name, generator_run in generator_runs_dict.items():
        out_of_sample_plot[generator_run_name] = {}
        for condition in generator_run.conditions:
            # This assumes context is None
            obsf = ObservationFeatures.from_condition(condition)
            # Make a prediction
            try:
                pred_y, pred_se = _predict_at_point(generator, obsf, metric_names)
            except Exception:
                # Check if it is an out-of-design condition.
                if not generator.model_space.validate(obsf.parameters):
                    # Skip this point
                    continue
                else:
                    # It should have worked
                    raise
            condition_name = condition.name_or_short_signature
            out_of_sample_plot[generator_run_name][
                condition_name
            ] = PlotOutOfSampleCondition(
                name=condition_name,
                params=obsf.parameters,
                y_hat=pred_y,
                se_hat=pred_se,
                context_stratum=None,
            )
    return out_of_sample_plot


def get_plot_data(
    generator: Generator,
    generator_runs_dict: Dict[str, GeneratorRun],
    metric_names: Optional[Set[str]] = None,
) -> Tuple[PlotData, RawData, Dict[str, TParameterization]]:
    """Format data object with metrics for in-sample and out-of-sample
    conditions.

    Calculate both observed and predicted metrics for in-sample conditions.
    Calculate predicted metrics for out-of-sample conditions passed via the
    `generator_runs_dict` argument.

    In PlotData, in-sample observations are merged with IVW. In RawData, they
    are left un-merged and given as a list of dictionaries, one for each
    observation and having keys 'condition_name', 'mean', and 'sem'.

    Args:
        generator: The generator.
        generator_runs_dict: a mapping from generator run name to generator run.
        metric_names: Restrict predictions to this set. If None, all metrics
            in the generator will be returned.

    Returns:
        plot_data: a PlotData object with in-sample and out-of-sample
            predictions.
        raw_data: A list of observations like
            {'metric_name': 'likes', 'condition_name': '0_1', 'mean': 1., 'sem': 0.1}.
        cond_name_to_params: A mapping from condition name to parameters.
    """
    metrics_plot = generator.metric_names if metric_names is None else metric_names
    in_sample_plot, raw_data, cond_name_to_params = _get_in_sample_conditions(
        generator=generator, metric_names=metrics_plot
    )
    out_of_sample_plot = _get_out_of_sample_conditions(
        generator=generator,
        generator_runs_dict=generator_runs_dict,
        metric_names=metrics_plot,
    )
    status_quo_name = (
        None if generator.status_quo is None else generator.status_quo.condition_name
    )
    plot_data = PlotData(
        metrics=list(metrics_plot),
        in_sample=in_sample_plot,
        out_of_sample=out_of_sample_plot,
        status_quo_name=status_quo_name,
    )
    return plot_data, raw_data, cond_name_to_params


def get_range_parameter(generator: Generator, param_name: str) -> RangeParameter:
    """
    Get the range parameter with the given name from the generator.

    Throws if parameter doesn't exist or is not a range parameter.

    Args:
        generator: The generator.
        param_name: The name of the RangeParameter to be found.

    Returns: The RangeParameter named `param_name`.
    """

    range_param = generator.model_space.parameters.get(param_name)
    if range_param is None:
        raise ValueError(f"Parameter `{param_name}` does not exist.")
    if not isinstance(range_param, RangeParameter):
        raise ValueError(f"{param_name} is not a RangeParameter")

    return range_param


def get_range_parameters(generator: Generator) -> List[RangeParameter]:
    """
    Get a list of range parameters from a generator.

    Args:
        generator: The generator.

    Returns: List of RangeParameters.
    """
    return [
        parameter
        for parameter in generator.model_space.parameters.values()
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
    generator: Generator, slice_values: Optional[Dict[str, Any]] = None
) -> TParameterization:
    """Get fixed values for parameters in a slice plot.

    If there is an in-design status quo, those values will be used. Otherwise,
    the mean of RangeParameters or the mode of ChoiceParameters is used.

    Any value in slice_values will override the above.

    Args:
        generator: Generator being used for plotting
        slice_values: Map from parameter name to value at which is should be
            fixed.

    Returns: Map from parameter name to fixed value.
    """
    # Check if status_quo is in design
    if generator.status_quo is not None and generator.model_space.validate(
        parameter_dict=generator.status_quo.features.parameters
    ):
        setx = generator.status_quo.features.parameters
    else:
        observations = generator.get_training_data()
        setx = {}
        for p_name, parameter in generator.model_space.parameters.items():
            vals = [
                obs.features.parameters[p_name]
                for obs in observations
                if parameter.validate(obs.features.parameters[p_name])
            ]
            if isinstance(parameter, FixedParameter):
                setx[p_name] = parameter.value
            elif isinstance(parameter, ChoiceParameter):
                setx[p_name] = Counter(vals).most_common(1)[0][1]
            elif isinstance(parameter, RangeParameter):
                setx[p_name] = parameter._cast(np.mean(vals))

    if slice_values is not None:
        # slice_values has type Dictionary[str, Any]
        setx.update(slice_values)
    return setx

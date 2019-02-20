#!/usr/bin/env python3

from typing import Any, Dict, Optional

import numpy as np
from ae.lazarus.ae.core.observation import ObservationFeatures
from ae.lazarus.ae.generator.base import Generator
from ae.lazarus.ae.plot.base import AEPlotConfig, AEPlotTypes
from ae.lazarus.ae.plot.helper import (
    TNullableGeneratorRunsDict,
    get_fixed_values,
    get_grid_for_parameter,
    get_plot_data,
    get_range_parameter,
)


def plot_slice(
    generator: Generator,
    param_name: str,
    metric_name: str,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    relative: bool = False,
    density: int = 50,
    slice_values: Optional[Dict[str, Any]] = None,
) -> AEPlotConfig:
    """Plot predictions for a 1-d slice of the parameter space.

    Args:
        generator: Generator that contains model for predictions
        param_name: Name of parameter that will be sliced
        metric_name: Name of metric to plot
        generator_runs_dict: A dictionary {name: generator run} of generator runs
            whose conditions will be plotted, if they lie in the slice.
        relative: Predictions relative to status quo
        density: Number of points along slice to evaluate predictions.
        slice_values: A dictionary {name: val} for the fixed values of the
            other parameters. If not provided, then the status quo values will
            be used if there is a status quo, otherwise the mean of numeric
            parameters or the mode of choice parameters.
    """
    if generator_runs_dict is None:
        generator_runs_dict = {}

    parameter = get_range_parameter(generator, param_name)
    grid = get_grid_for_parameter(parameter, density)

    plot_data, raw_data, cond_name_to_params = get_plot_data(
        generator=generator,
        generator_runs_dict=generator_runs_dict,
        metric_names={metric_name},
    )

    fixed_values = get_fixed_values(generator, slice_values)

    prediction_features = []
    for x in grid:
        params = fixed_values.copy()
        params[param_name] = x
        # Here we assume context is None
        prediction_features.append(ObservationFeatures(parameters=params))

    f, cov = generator.predict(prediction_features)

    f_plt = f[metric_name]
    sd_plt = np.sqrt(cov[metric_name][metric_name])

    config = {
        "condition_data": plot_data,
        "condition_name_to_params": cond_name_to_params,
        "f": f_plt,
        "fit_data": raw_data,
        "grid": grid,
        "metric": metric_name,
        "param": param_name,
        "rel": relative,
        "setx": fixed_values,
        "sd": sd_plt,
        "is_log": parameter.log_scale,
    }
    return AEPlotConfig(config, plot_type=AEPlotTypes.SLICE)

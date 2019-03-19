#!/usr/bin/env python3

from typing import Any, Dict, Optional, Tuple

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.modelbridge.base import ModelBridge
from ax.plot.base import AEPlotConfig, AEPlotTypes, PlotData
from ax.plot.color import BLUE_SCALE, GREEN_PINK_SCALE, GREEN_SCALE
from ax.plot.helper import (
    TNullableGeneratorRunsDict,
    get_fixed_values,
    get_grid_for_parameter,
    get_plot_data,
    get_range_parameter,
    get_range_parameters,
)


# type aliases
ContourPredictions = Tuple[
    PlotData, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, bool]
]


def _get_contour_predictions(
    model: ModelBridge,
    x_param_name: str,
    y_param_name: str,
    metric: str,
    generator_runs_dict: TNullableGeneratorRunsDict,
    density: int,
    slice_values: Optional[Dict[str, Any]] = None,
) -> ContourPredictions:
    """
    slice_values is a dictionary {param_name: value} for the parameters that
    are being sliced on.
    """
    x_param = get_range_parameter(model, x_param_name)
    y_param = get_range_parameter(model, y_param_name)

    plot_data, _, _ = get_plot_data(model, generator_runs_dict or {}, {metric})

    grid_x = get_grid_for_parameter(x_param, density)
    grid_y = get_grid_for_parameter(y_param, density)
    scales = {"x": x_param.log_scale, "y": y_param.log_scale}

    grid2_x, grid2_y = np.meshgrid(grid_x, grid_y)

    grid2_x = grid2_x.flatten()
    grid2_y = grid2_y.flatten()

    fixed_values = get_fixed_values(model, slice_values)

    param_grid_obsf = []
    for i in range(density ** 2):
        params = fixed_values.copy()
        params[x_param_name] = grid2_x[i]
        params[y_param_name] = grid2_y[i]
        param_grid_obsf.append(ObservationFeatures(params))

    mu, cov = model.predict(param_grid_obsf)

    f_plt = mu[metric]
    sd_plt = np.sqrt(cov[metric][metric])
    return plot_data, f_plt, sd_plt, grid_x, grid_y, scales


def plot_contour(
    model: ModelBridge,
    param_x: str,
    param_y: str,
    metric: str,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel: bool = True,
    density: int = 50,
    slice_values: Optional[Dict[str, Any]] = None,
    lower_is_better: bool = False,
) -> AEPlotConfig:
    if param_x == param_y:
        raise ValueError("Please select different parameters for x- and y-dimensions.")
    data, f_plt, sd_plt, grid_x, grid_y, scales = _get_contour_predictions(
        model=model,
        x_param_name=param_x,
        y_param_name=param_y,
        metric=metric,
        generator_runs_dict=generator_runs_dict,
        density=density,
        slice_values=slice_values,
    )
    config = {
        "arm_data": data,
        "blue_scale": BLUE_SCALE,
        "density": density,
        "f": f_plt,
        "green_scale": GREEN_SCALE,
        "green_pink_scale": GREEN_PINK_SCALE,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "lower_is_better": lower_is_better,
        "metric": metric,
        "rel": rel,
        "sd": sd_plt,
        "xvar": param_x,
        "yvar": param_y,
        "x_is_log": scales["x"],
        "y_is_log": scales["y"],
    }
    return AEPlotConfig(config, plot_type=AEPlotTypes.CONTOUR)


def interact_contour(
    model: ModelBridge,
    metric: str,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    rel: bool = True,
    density: int = 50,
    slice_values: Optional[Dict[str, Any]] = None,
    lower_is_better: bool = False,
) -> AEPlotConfig:
    range_parameters = get_range_parameters(model)
    plot_data, _, _ = get_plot_data(model, generator_runs_dict or {}, {metric})

    # TODO T38563759: Sort parameters by feature importances
    param_names = [parameter.name for parameter in range_parameters]

    is_log_dict: Dict[str, bool] = {}
    grid_dict: Dict[str, np.ndarray] = {}
    for parameter in range_parameters:
        is_log_dict[parameter.name] = parameter.log_scale
        grid_dict[parameter.name] = get_grid_for_parameter(parameter, density)

    # pyre: f_dict is declared to have type `Dict[str, Dict[str, np.ndarray]]`
    # pyre-fixme[9]: but is used as type `Dict[str, Dict[str, typing.List[]]]`.
    f_dict: Dict[str, Dict[str, np.ndarray]] = {
        param1: {param2: [] for param2 in param_names} for param1 in param_names
    }
    # pyre: sd_dict is declared to have type `Dict[str, Dict[str, np.
    # pyre: ndarray]]` but is used as type `Dict[str, Dict[str, typing.
    # pyre-fixme[9]: List[]]]`.
    sd_dict: Dict[str, Dict[str, np.ndarray]] = {
        param1: {param2: [] for param2 in param_names} for param1 in param_names
    }
    for param1 in param_names:
        for param2 in param_names:
            _, f_plt, sd_plt, _, _, _ = _get_contour_predictions(
                model=model,
                x_param_name=param1,
                y_param_name=param2,
                metric=metric,
                generator_runs_dict=generator_runs_dict,
                density=density,
                slice_values=slice_values,
            )
            f_dict[param1][param2] = f_plt
            sd_dict[param1][param2] = sd_plt

    config = {
        "arm_data": plot_data,
        "blue_scale": BLUE_SCALE,
        "density": density,
        "f_dict": f_dict,
        "green_scale": GREEN_SCALE,
        "green_pink_scale": GREEN_PINK_SCALE,
        "grid_dict": grid_dict,
        "lower_is_better": lower_is_better,
        "metric": metric,
        "rel": rel,
        "sd_dict": sd_dict,
        "is_log_dict": is_log_dict,
        "param_names": param_names,
    }
    return AEPlotConfig(config, plot_type=AEPlotTypes.INTERACT_CONTOUR)

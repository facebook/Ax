#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from ax.core.experiment import Experiment
from ax.core.search_space import SearchSpace
from ax.core.trial import BaseTrial, Trial
from ax.modelbridge import ModelBridge
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.random import RandomModelBridge
from ax.plot.contour import interact_contour_plotly
from ax.plot.slice import plot_slice_plotly
from ax.plot.trace import optimization_trace_single_method_plotly
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none


logger: Logger = get_logger(__name__)


def _get_objective_trace_plot(
    trials: Dict[int, BaseTrial],
    metric_name: str,
    model_transitions: List[int],
    optimization_direction: Optional[str] = None,
    # pyre-ignore[11]: Annotation `go.Figure` is not defined as a type.
) -> Optional[go.Figure]:
    best_objectives = np.array(
        [[checked_cast(Trial, t).objective_mean for t in trials.values()]]
    )
    return optimization_trace_single_method_plotly(
        y=best_objectives,
        title="Best objective found vs. # of iterations",
        ylabel=metric_name,
        model_transitions=model_transitions,
        optimization_direction=optimization_direction,
        plot_trial_points=True,
    )


def _get_objective_v_param_plot(
    search_space: SearchSpace,
    model: ModelBridge,
    metric_name: str,
    trials: Dict[int, BaseTrial],
) -> Optional[go.Figure]:
    range_params = list(search_space.range_parameters.keys())
    if len(range_params) == 1:
        # individual parameter slice plot
        output_slice_plot = plot_slice_plotly(
            model=not_none(model),
            param_name=range_params[0],
            metric_name=metric_name,
            generator_runs_dict={
                str(t.index): not_none(checked_cast(Trial, t).generator_run)
                for t in trials.values()
            },
        )
        return output_slice_plot
    if len(range_params) > 1:
        # contour plot
        output_contour_plot = interact_contour_plotly(
            model=not_none(model),
            metric_name=metric_name,
        )
        return output_contour_plot


def get_standard_plots(
    experiment: Experiment, generation_strategy: GenerationStrategy
) -> List[go.Figure]:
    """Extract standard plots for single-objective optimization.

    TODO: Describe specific logic here on what happens depending on search space
    and generation strategy attributes.
    """
    output_plot_list = []
    output_plot_list.append(
        _get_objective_trace_plot(
            trials=experiment.trials,
            metric_name=not_none(experiment.optimization_config).objective.metric.name,
            model_transitions=generation_strategy.model_transitions,
            optimization_direction=(
                "minimize"
                if not_none(experiment.optimization_config).objective.minimize
                else "maximize"
            ),
        )
    )

    # TODO: Replace this check with a check of whether the model
    # implements `predict`.
    if isinstance(generation_strategy.model, RandomModelBridge):
        return output_plot_list

    output_plot_list.append(
        _get_objective_v_param_plot(
            search_space=experiment.search_space,
            model=not_none(generation_strategy.model),
            metric_name=not_none(experiment.optimization_config).objective.metric.name,
            trials=experiment.trials,
        )
    )

    return [plot for plot in output_plot_list if plot is not None]

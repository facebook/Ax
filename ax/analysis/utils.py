# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import itertools

from ax.analysis.analysis import Analysis
from ax.analysis.plotly.cross_validation import CrossValidationPlot
from ax.analysis.plotly.interaction import InteractionPlot
from ax.analysis.plotly.parallel_coordinates import ParallelCoordinatesPlot
from ax.analysis.plotly.scatter import ScatterPlot
from ax.analysis.summary import Summary
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective, ScalarizedObjective


def choose_analyses(experiment: Experiment) -> list[Analysis]:
    """
    Choose a default set of Analyses to compute based on the current state of the
    Experiment.
    """
    if (optimization_config := experiment.optimization_config) is None:
        return []

    if isinstance(optimization_config.objective, MultiObjective) or isinstance(
        optimization_config.objective, ScalarizedObjective
    ):
        # Pareto frontiers for each objective
        objective_plots = [
            *[
                ScatterPlot(x_metric_name=x, y_metric_name=y, show_pareto_frontier=True)
                for x, y in itertools.combinations(
                    optimization_config.objective.metric_names, 2
                )
            ],
        ]

        other_scatters = []

        interactions = [
            InteractionPlot(metric_name=name)
            for name in optimization_config.objective.metric_names
        ]

    else:
        objective_name = optimization_config.objective.metric.name
        # ParallelCoorindates and leave-one-out cross validation
        objective_plots = [
            ParallelCoordinatesPlot(metric_name=objective_name),
        ]

        # Up to six ScatterPlots for other metrics versus the objective,
        # prioritizing optimization config metrics over tracking metrics
        tracking_metric_names = [metric.name for metric in experiment.tracking_metrics]
        other_scatters = [
            ScatterPlot(
                x_metric_name=objective_name,
                y_metric_name=name,
                show_pareto_frontier=False,
            )
            for name in [
                *optimization_config.metrics,
                *tracking_metric_names,
            ]
            if name != objective_name
        ][:6]

        interactions = [InteractionPlot(metric_name=objective_name)]

    # Leave-one-out cross validation for each objective and outcome constraint
    cv_plots = [
        CrossValidationPlot(metric_name=name) for name in optimization_config.metrics
    ]

    return [*objective_plots, *other_scatters, *interactions, *cv_plots, Summary()]

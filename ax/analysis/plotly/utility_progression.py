# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

import numpy as np
import pandas as pd
import plotly.express as px
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.plotly.color_constants import AX_BLUE
from ax.analysis.plotly.plotly_analysis import (
    create_plotly_analysis_card,
    PlotlyAnalysisCard,
)
from ax.analysis.utils import validate_experiment
from ax.core.experiment import Experiment
from ax.exceptions.core import ExperimentNotReadyError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.best_point import get_trace
from pyre_extensions import none_throws, override

# Common subtitle components for reuse across different plot types
_UTILITY_PROGRESSION_TITLE = "Utility Progression"

_TRACE_INDEX_EXPLANATION = (
    "The x-axis shows trial index. Only completed or early-stopped trials with "
    "complete metric data are included, so there may be gaps if some trials "
    "failed, were abandoned, or have incomplete data."
)

_CUMULATIVE_BEST_EXPLANATION = (
    "The y-axis shows cumulative best utility. Only improvements are plotted, so "
    "flat segments indicate trials that didn't surpass the previous best."
)

_INFEASIBLE_TRIALS_EXPLANATION = (
    "Infeasible trials (violating outcome constraints) don't contribute to "
    "the improvements."
)


@final
class UtilityProgressionAnalysis(Analysis):
    """
    Plotly line plot showing the utility progression over completed trial iterations.

    For single-objective experiments, utility is the best observed objective
    value seen so far. For multi-objective experiments, utility is the
    hypervolume of the Pareto frontier.

    The DataFrame computed will contain one row per completed trial and the
    following columns:
        - trial_index: The trial index of each completed/early-stopped trial
            that has complete metric avilability.
        - utility: The cumulative best utility value at that trial
    """

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        UtilityProgressionAnalysis requires an Experiment with completed trials
        and data.
        """
        if (
            experiment_invalid_reason := validate_experiment(
                experiment=experiment,
                require_trials=True,
                require_data=True,
            )
        ) is not None:
            return experiment_invalid_reason

        # Check that optimization config exists
        opt_config = none_throws(experiment).optimization_config
        if opt_config is None:
            return "Experiment must have an optimization config."

        return None

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> PlotlyAnalysisCard:
        experiment = none_throws(experiment)
        opt_config = experiment.optimization_config

        # Compute the utility trace using existing utility
        trace = get_trace(
            experiment=experiment,
            optimization_config=opt_config,
            include_status_quo=True,
        )

        # Check if trace is empty
        if len(trace) == 0:
            raise ExperimentNotReadyError(
                "No utility trace data available. This can happen when there are no "
                "completed trials with valid data, or when all trials violate outcome "
                "constraints."
            )

        # Check if all points are infeasible (inf or -inf values)
        if all(np.isinf(value) for value in trace.values()):
            raise ExperimentNotReadyError(
                "All trials in the utility trace are infeasible i.e., they violate "
                "outcome constraints, so there are no feasible points to plot. During "
                "early, exploratory phases of the experiment (e.g., Sobol), this can "
                "be an expected outcome. However, if your experiment has been running "
                "for a while, consider: (1) adding a feasible point as a baseline or "
                "status quo arm to guide Ax towards feasible regions of the search "
                "space, or (2) relaxing outcome constraints."
            )

        # Create DataFrame with trial indices from the trace
        df = pd.DataFrame(
            {
                "trial_index": list(trace.keys()),
                "utility": list(trace.values()),
            }
        )

        # Subtitle and y-axis label vary by problem type
        title = _UTILITY_PROGRESSION_TITLE

        if experiment.is_bope_problem:
            y_label = "User Preference Score"
            subtitle = (
                "Shows the best user preference score achieved so far across "
                f"completed trials. {_TRACE_INDEX_EXPLANATION} "
                f"{_CUMULATIVE_BEST_EXPLANATION} "
                f"{_INFEASIBLE_TRIALS_EXPLANATION}"
            )
        elif experiment.is_moo_problem:
            y_label = "Hypervolume"
            subtitle = (
                "Shows the hypervolume of the Pareto frontier achieved so far across "
                f"completed trials. {_TRACE_INDEX_EXPLANATION} The y-axis shows "
                "cumulative best hypervolume -- only improvements, so flat "
                "segments indicate trials that didn't improve the frontier. "
                "Hypervolume measures the volume of objective space dominated by the "
                f"Pareto frontier. "
                f"{_INFEASIBLE_TRIALS_EXPLANATION}"
            )
        else:
            objective = none_throws(opt_config).objective

            # Handle ScalarizedObjective vs regular Objective
            if objective.is_scalarized_objective:
                expression = objective.expression
                y_label = f"Best Observed {expression}"
                subtitle = (
                    f"Shows the best scalarized objective value (formula: "
                    f"{expression}) achieved so far across completed trials. "
                    f"{_TRACE_INDEX_EXPLANATION} "
                    f"{_CUMULATIVE_BEST_EXPLANATION} "
                    f"{_INFEASIBLE_TRIALS_EXPLANATION}"
                )
            else:
                minimize = objective.minimize
                direction = "minimize" if minimize else "maximize"
                objective_name = objective.metric_names[0]
                y_label = f"Best Observed {objective_name}"
                subtitle = (
                    f"Shows the best {objective_name} value achieved so far across "
                    f"completed trials (objective is to {direction}). "
                    f"{_TRACE_INDEX_EXPLANATION} {_CUMULATIVE_BEST_EXPLANATION} "
                    f"{_INFEASIBLE_TRIALS_EXPLANATION}"
                )

        # Create the plot
        fig = px.line(
            data_frame=df,
            x="trial_index",
            y="utility",
            markers=True,
            color_discrete_sequence=[AX_BLUE],
        )

        # Update axis labels and format x-axis to show integers only
        fig.update_xaxes(title_text="Trial Index", dtick=1, rangemode="nonnegative")
        fig.update_yaxes(title_text=y_label)

        return create_plotly_analysis_card(
            name=self.__class__.__name__,
            title=title,
            subtitle=subtitle,
            df=df,
            fig=fig,
        )

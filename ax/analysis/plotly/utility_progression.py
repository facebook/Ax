# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

import pandas as pd
import plotly.express as px
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.plotly.plotly_analysis import (
    create_plotly_analysis_card,
    PlotlyAnalysisCard,
)
from ax.analysis.utils import validate_experiment
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.best_point import get_trace
from pyre_extensions import override


@final
class UtilityProgressionAnalysis(Analysis):
    """
    Plotly line plot showing the utility progression over trials.

    For single-objective experiments, utility is the best observed objective
    value seen so far. For multi-objective experiments, utility is the
    hypervolume of the Pareto frontier. For preference learning experiments,
    utility is derived from preference data.

    The DataFrame computed will contain one row per trial and the following columns:
        - trial_index: The trial index
        - utility: The cumulative best utility value at that trial
    """

    def __init__(self) -> None:
        pass

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

        if experiment is None:
            raise UserInputError("UtilityProgressionPlot requires an Experiment")

        # Check that optimization config exists
        if experiment.optimization_config is None:
            return "Experiment must have an optimization config."

        return None

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> PlotlyAnalysisCard:
        if experiment is None:
            raise UserInputError("UtilityProgressionPlot requires an Experiment")

        if experiment.optimization_config is None:
            raise UserInputError("Experiment must have an optimization config.")

        opt_config = experiment.optimization_config

        # Compute the utility trace using existing utility
        trace = get_trace(
            experiment=experiment,
            optimization_config=opt_config,
        )

        # Create DataFrame
        df = pd.DataFrame(
            {
                "trial_index": list(range(len(trace))),
                "utility": trace,
            }
        )

        # Determine the appropriate title, subtitle, and axis labels based on
        # problem type
        if experiment.is_preference_learning_problem:
            title = "Learned Utility Progression"
            y_label = "Learned Utility"
            subtitle = (
                "The learned utility progression plot tracks the cumulative "
                "best utility derived from preference data over the course of "
                "the experiment. This visualization helps monitor optimization "
                "progress and identify convergence or stagnation in preference "
                "learning."
            )
        elif experiment.is_moo_problem:
            title = "Pareto Frontier Hypervolume Progression"
            y_label = "Hypervolume"
            subtitle = (
                "The utility progression plot tracks the hypervolume of "
                "the Pareto frontier over the course of the experiment. This "
                "visualization helps monitor multi-objective optimization "
                "progress and identify convergence or stagnation."
            )
        else:
            objective_name = opt_config.objective.metric.name
            minimize = opt_config.objective.minimize
            direction = "minimized" if minimize else "maximized"
            title = f"Best {objective_name} Progression"
            y_label = f"Best {objective_name} ({direction})"
            subtitle = (
                f"The optimization progression plot tracks the cumulative best "
                f"{objective_name} value over the course of the experiment. "
                f"This visualization helps monitor optimization progress and "
                f"identify convergence or stagnation."
            )

        # Create the plot
        fig = px.line(
            data_frame=df,
            x="trial_index",
            y="utility",
            title=title,
            markers=True,
        )

        # Update axis labels
        fig.update_xaxes(title_text="Trial Index")
        fig.update_yaxes(title_text=y_label)

        return create_plotly_analysis_card(
            name=self.__class__.__name__,
            title=title,
            subtitle=subtitle,
            df=df,
            fig=fig,
        )

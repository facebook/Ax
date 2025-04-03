# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import Sequence

import pandas as pd

from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel
from ax.analysis.plotly.arm_effects.utils import prepare_arm_effects_plot
from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.utils import get_autoset_axis_limits

from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.trial import Trial
from ax.exceptions.core import DataRequiredError, UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from ax.service.utils.best_point import (
    BestPointValueError,
    get_best_raw_objective_point_with_trial_index,
    is_trial_arm_feasible,
)
from ax.utils.common.logger import get_logger
from plotly import express as px

from pyre_extensions import assert_is_instance, none_throws

logger: Logger = get_logger(__name__)

GEN_METHOD_COL_NAME = "generation_method"
FEASIBLE_COL_NAME = "overall_probability_constraints_violated"


class OptimizationTrace(PlotlyAnalysis):
    """Plotly analysis that creates a visualization of optimization progress over
    an experiment's trials.

    The analysis generates a plot showing how the objective value changes across trials,
    including the running best value achieved. This helps track optimization performance
    and convergence over time.

    Visualization elements:
    - Individual trial points colored by feasibility (blue=feasible, grey=infeasible)
    - Running best objective value shown as red step line
    - Hover information showing trial index, arm name, and metric value
    """

    def __init__(
        self,
        hover_data_colnames: list[str] | None = None,
    ) -> None:
        """Initialize OptimizationTrace."""
        self.hover_data_colnames = hover_data_colnames
        super().__init__()

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[PlotlyAnalysisCard]:
        """Compute the optimization trace visualization.

        Args:
            experiment: The experiment to analyze.
            generation_strategy: The generation strategy used (not used in this
                analysis).

        Returns:
            A PlotlyAnalysisCard containing the visualization.
        """
        df, metric_name, minimize, outcome_constraints = self.compute_plot_df(
            experiment=experiment
        )

        scatter = prepare_arm_effects_plot(
            df,
            metric_name=metric_name,
            outcome_constraints=outcome_constraints,
            plot_status_quo_line=False,
        )

        running_optimum_df = compute_running_optimum_df(
            experiment=none_throws(experiment), metric_name=metric_name
        )
        running_optimum_df["Legend"] = "Running optimum"
        line = px.line(
            data_frame=running_optimum_df,
            x="arm_name",
            y="running_optimum",
            color="Legend",
            line_shape="hv",
        )
        fig = scatter.add_trace(line.data[0])
        layout_yaxis_range = get_autoset_axis_limits(
            y=df["mean"].to_numpy(),
            optimization_direction="minimize" if minimize else "maximize",
            force_include_value=running_optimum_df.loc[0, "running_optimum"],
        )
        fig.update_layout(yaxis_range=layout_yaxis_range)

        return [
            self._create_plotly_analysis_card(
                title=(
                    "Optimization Progress "
                    f"({'minimizing' if minimize else 'maximizing'} {metric_name})"
                ),
                subtitle=(
                    "Objective values and running optimum. "
                    "Zoom level omits outliers by default."
                ),
                level=AnalysisCardLevel.HIGH,
                df=df,
                fig=fig,
                category=AnalysisCardCategory.INSIGHT,
            )
        ]

    def compute_plot_df(
        self, experiment: Experiment | None
    ) -> tuple[pd.DataFrame, str, bool, list[OutcomeConstraint]]:
        """Retrieves experiment dataframe and preprocesses it to contain only the
        necessary columns for plotting ObjectiveTrace. This includes:
            - Keeping only necessary columns like trial_index, arm_names, the metric
              being optimized, and any hover data columns.
            - Keeping completed trials only.
            - Computing feasibility of each trial.

        Returns:
            - The dataframe to be used in plotting
            - The name of the objective metric to be plotted
            - A boolean representing whether this objective is being minimized
        """
        exp_df, metric_name, minimize = _validate_experiment_and_get_experiment_df(
            experiment=experiment
        )

        # Experiment has been validated as not-None in the above `_validate` method.
        experiment = none_throws(experiment)
        used_cols = [
            "arm_name",
            metric_name,
        ]
        if self.hover_data_colnames is not None:
            used_cols.extend(self.hover_data_colnames)

        # Use completed trials only.
        completed_trial_indices = [t.index for t in experiment.completed_trials]
        exp_df = exp_df.loc[exp_df["trial_index"].isin(completed_trial_indices)]

        # Compute feasibility of each trial/arm.
        if (
            experiment.optimization_config is not None
            and len(none_throws(experiment.optimization_config).all_constraints) > 0
        ):
            try:
                exp_df[FEASIBLE_COL_NAME] = exp_df.apply(
                    lambda row: not is_trial_arm_feasible(
                        experiment=experiment,
                        trial_index=row["trial_index"],
                        arm_name=row["arm_name"],
                    ),
                    axis=1,
                )
            except (KeyError, ValueError, DataRequiredError) as e:
                logger.warning(
                    f"Feasibility calculation failed with error: {e}. "
                    "Not attaching feasibility column."
                )

            used_cols.append(FEASIBLE_COL_NAME)

        # Keep generation_method column if it's being used.
        if GEN_METHOD_COL_NAME in exp_df.columns:
            exp_df.rename(columns={GEN_METHOD_COL_NAME: "source"}, inplace=True)
            exp_df["source"].fillna("Manual", inplace=True)
            used_cols.append("source")

        # Subset to only used columns.
        exp_df = exp_df[used_cols]

        exp_df.rename(columns={metric_name: "mean"}, inplace=True)

        exp_df["error_margin"] = exp_df["mean"].apply(lambda x: 0.0)

        return (
            assert_is_instance(exp_df, pd.DataFrame),
            metric_name,
            minimize,
            none_throws(experiment.optimization_config).outcome_constraints,
        )


def _validate_experiment_and_get_experiment_df(
    experiment: Experiment | None,
) -> tuple[pd.DataFrame, str, bool]:
    """Validates that the experiment exists, has a single-objective
    OptimizationConfig, and contains some data including trial_index,
    arm_name, and mean values corresponding to the objective metric.

    Returns:
        - The experiment's dataframe obtained by ``experiment.to_df``
        - The name of the objective metric to be plotted
        - A boolean representing whether this objective is being minimized
    """
    if experiment is None:
        raise UserInputError(
            "Experiment cannot be None for OptimizationTrace analysis."
        )

    if experiment.optimization_config is None:
        raise UserInputError(
            "Experiment must have an optimization config for OptimizationTrace "
            "analysis."
        )

    if isinstance(experiment.optimization_config.objective, MultiObjective):
        raise UserInputError(
            "OptimizationTrace does not support multi-objective optimization."
        )

    if not all(isinstance(t, Trial) for t in experiment.trials.values()):
        raise UserInputError("OptimizationTrace supports only single-arm Trials.")

    objective = none_throws(experiment.optimization_config).objective
    metric_name = objective.metric.name
    minimize = objective.minimize

    # Get experiment DataFrame
    exp_df = experiment.to_df()

    if exp_df.empty:
        raise UserInputError("Experiment contains no data")

    # Check required columns
    required_columns = ["trial_index", "arm_name"]
    missing_columns = [col for col in required_columns if col not in exp_df.columns]
    if missing_columns:
        raise UserInputError(f"Missing required columns: {missing_columns}")

    if metric_name not in exp_df.columns:
        raise UserInputError(
            f"Optimization metric '{metric_name}' not found in experiment data"
        )

    return exp_df, metric_name, minimize


def compute_running_optimum_df(
    experiment: Experiment,
    metric_name: str,
) -> pd.DataFrame:
    """Compute the running optimum for an experiment.

    The running optimum is the best objective value achieved across all completed
    trials. This is used to track optimization progress over time.

    For efficiency's sake, this method works backward by computing the overall optimum,
    then iteratively removing the tail of the sequence (starting from the optimum) and
    computing the optimum on that reduced set of trials. This process is repeated until
    the first trial is all that remains.

    Args:
        experiment: The experiment from which to extract the running optimum as a
            function of trial index.

    Returns:
        A DataFrame containing the running optimum for each trial.
    """
    best_trial_index, _, best_metric_value_dict = (
        get_best_raw_objective_point_with_trial_index(experiment=experiment)
    )
    if best_trial_index is None:
        return pd.DataFrame()

    def _fetch_arm_name(trial_index: int) -> str:
        return none_throws(
            assert_is_instance(experiment.trials[trial_index], Trial).arm
        ).name

    records = [
        {
            "trial_index": trial_index,
            "arm_name": _fetch_arm_name(trial_index=trial_index),
            "running_optimum": best_metric_value_dict[metric_name][0],
        }
        for trial_index in [best_trial_index, max(experiment.trials.keys())]
    ]
    while best_trial_index > 0:
        trial_indices = list(range(best_trial_index))
        try:
            best_trial_index, _, best_metric_value_dict = (
                get_best_raw_objective_point_with_trial_index(
                    experiment=experiment, trial_indices=trial_indices
                )
            )
        except BestPointValueError:
            break

        records.append(
            {
                "trial_index": best_trial_index,
                "arm_name": _fetch_arm_name(trial_index=best_trial_index),
                "running_optimum": best_metric_value_dict[metric_name][0],
            }
        )
    return pd.DataFrame.from_records(records).sort_values("trial_index")

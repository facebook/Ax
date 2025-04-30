# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import Mapping, Sequence

import pandas as pd
from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.utils import (
    BEST_LINE_SETTINGS,
    get_arm_tooltip,
    is_predictive,
    trial_status_to_plotly_color,
    truncate_label,
)
from ax.analysis.utils import (
    extract_relevant_adapter,
    POSSIBLE_CONSTRAINT_VIOLATION_THRESHOLD,
    prepare_arm_data,
)
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from ax.modelbridge.registry import Generators
from ax.utils.common.logger import get_logger
from plotly import graph_objects as go
from pyre_extensions import override

logger: Logger = get_logger(__name__)


class ScatterPlot(PlotlyAnalysis):
    """
    Plotly Scatter plot for any two metrics. Each arm is represented by a single point
    with 95% confidence intervals if the data is available. Effects may be either the
    raw observed effects, or the predicted effects using a model. The latter is often
    more trustworthy (and leads to better reproducibility) than using the raw data,
    especially when model fit is good and in high-noise settings.


    The DataFrame computed will contain one row per arm and the following columns:
        - trial_index: The trial index of the arm
        - trial_status: The status of the trial
        - arm_name: The name of the arm
        - generation_node: The name of the ``GenerationNode`` that generated the arm
        - p_feasible: The probability that the arm is feasible (does not violate any
            constraints)
        - **METRIC_NAME_mean: The observed mean of the metric specified
        - **METRIC_NAME_sem: The observed sem of the metric specified
    """

    def __init__(
        self,
        x_metric_name: str,
        y_metric_name: str,
        use_model_predictions: bool = True,
        relativize: bool = False,
        trial_index: int | None = None,
        trial_statuses: Sequence[TrialStatus] | None = None,
        additional_arms: Sequence[Arm] | None = None,
        labels: Mapping[str, str] | None = None,
        show_pareto_frontier: bool = False,
    ) -> None:
        """
        Args:
            x_metric_name: The name of the metric to plot on the x-axis.
            y_metric_name: The name of the metric to plot on the y-axis.
            use_model_predictions: Whether to use model predictions or observed data.
                If ``True``, the plot will show the predicted effects of each arm based
                on the model. If ``False``, the plot will show the observed effects of
                each arm. The latter is often less trustworthy than the former,
                especially when model fit is good and in high-noise settings.
            relativize: Whether to relativize the effects of each arm against the status
                quo arm. If multiple status quo arms are present, relativize each arm
                against the status quo arm from the same trial.
            trial_index: If present, only use arms from the trial with the given index.
            additional_arms: If present, include these arms in the plot in addition to
                the arms in the experiment. These arms will be marked as belonging to a
                trial with index -1.
            labels: A mapping from metric names to labels to use in the plot. If a label
                is not provided for a metric, the metric name will be used.
            show_pareto_frontier: Whether to draw a line representing the Pareto
                frontier for the two metrics on the plot.
        """

        self.x_metric_name = x_metric_name
        self.y_metric_name = y_metric_name
        self.use_model_predictions = use_model_predictions
        self.relativize = relativize
        self.trial_index = trial_index
        self.trial_statuses = trial_statuses
        self.additional_arms = additional_arms
        self.labels: dict[str, str] = {**labels} if labels is not None else {}
        self.show_pareto_frontier = show_pareto_frontier

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[PlotlyAnalysisCard]:
        if experiment is None:
            raise UserInputError("ScatterPlot requires an Experiment")

        if self.use_model_predictions:
            relevant_adapter = extract_relevant_adapter(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )

            if not is_predictive(relevant_adapter):
                logger.warning(
                    f"Adapter {relevant_adapter} cannot make out of sample "
                    "predictions, falling back to EmpiricalBayesThompson."
                )

                data = (
                    experiment.lookup_data(trial_indices=[self.trial_index])
                    if self.trial_index is not None
                    else experiment.lookup_data()
                )
                relevant_adapter = Generators.EMPIRICAL_BAYES_THOMPSON(
                    experiment=experiment, data=data
                )
        else:
            relevant_adapter = None

        df = prepare_arm_data(
            experiment=experiment,
            metric_names=[self.x_metric_name, self.y_metric_name],
            use_model_predictions=self.use_model_predictions,
            adapter=relevant_adapter,
            trial_index=self.trial_index,
            trial_statuses=self.trial_statuses,
            additional_arms=self.additional_arms,
            relativize=self.relativize,
        )

        # Retrieve the metric labels from the mapping provided by the user, defaulting
        # to the metric name if no label is provided, truncated.
        x_metric_label = self.labels.get(
            self.x_metric_name, truncate_label(label=self.x_metric_name)
        )
        y_metric_label = self.labels.get(
            self.y_metric_name, truncate_label(label=self.y_metric_name)
        )

        x_lower_is_better = experiment.metrics[self.x_metric_name].lower_is_better
        y_lower_is_better = experiment.metrics[self.y_metric_name].lower_is_better

        figure = _prepare_figure(
            df=df,
            x_metric_name=self.x_metric_name,
            y_metric_name=self.y_metric_name,
            x_metric_label=x_metric_label,
            y_metric_label=y_metric_label,
            is_relative=self.relativize,
            show_pareto_frontier=self.show_pareto_frontier,
            x_lower_is_better=x_lower_is_better
            if x_lower_is_better is not None
            else False,
            y_lower_is_better=y_lower_is_better
            if y_lower_is_better is not None
            else False,
        )

        return [
            self._create_plotly_analysis_card(
                title=(
                    ("Modeled" if self.use_model_predictions else "Observed")
                    + f" {x_metric_label} vs. {y_metric_label}"
                ),
                subtitle=(
                    "This plot displays the effects of each arm on the two selected "
                    "metrics. It is useful for understanding the trade-off between "
                    "the two metrics and for visualizing the Pareto frontier."
                ),
                level=AnalysisCardLevel.HIGH,
                df=df,
                fig=figure,
                category=AnalysisCardCategory.INSIGHT,
            )
        ]


def compute_scatter_adhoc(
    experiment: Experiment,
    x_metric_name: str,
    y_metric_name: str,
    generation_strategy: GenerationStrategy | None = None,
    adapter: Adapter | None = None,
    use_model_predictions: bool = True,
    relativize: bool = False,
    trial_index: int | None = None,
    trial_statuses: Sequence[TrialStatus] | None = None,
    additional_arms: Sequence[Arm] | None = None,
    labels: Mapping[str, str] | None = None,
) -> list[PlotlyAnalysisCard]:
    """
    Compute ScatterPlot cards for the given experiment and either Adapter or
    GenerationStrategy.

    Note that cards are not saved to the database when computed adhoc -- they are only
    saved when computed as part of a call to ``Client.compute_analyses`` or equivalent.

    Args:
        experiment: The experiment to extract data from.
        x_metric_name: The name of the metric to plot on the x-axis.
        y_metric_name: The name of the metric to plot on the y-axis.
        generation_strategy: The GenerationStrategy to use for predictions if
            use_model_predictions=True.
        adapter: The adapter to use for predictions if use_model_predictions=True.
        use_model_predictions: Whether to use model predictions or observed data.
            If ``True``, the plot will show the predicted effects of each arm based
            on the model. If ``False``, the plot will show the observed effects of
            each arm. The latter is often less trustworthy than the former,
            especially when model fit is good and in high-noise settings.
        relativize: Whether to relativize the effects of each arm against the status
            quo arm. If multiple status quo arms are present, relativize each arm
            against the status quo arm from the same trial.
        trial_index: If present, only use arms from the trial with the given index.
        additional_arms: If present, include these arms in the plot in addition to
            the arms in the experiment. These arms will be marked as belonging to a
            trial with index -1.
        labels: A mapping from metric names to labels to use in the plot. If a label
            is not provided for a metric, the metric name will be used.
    """
    analysis = ScatterPlot(
        x_metric_name=x_metric_name,
        y_metric_name=y_metric_name,
        use_model_predictions=use_model_predictions,
        relativize=relativize,
        trial_index=trial_index,
        trial_statuses=trial_statuses,
        additional_arms=additional_arms,
        labels=labels,
    )

    return [
        *analysis.compute(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )
    ]


def _prepare_figure(
    df: pd.DataFrame,
    x_metric_name: str,
    y_metric_name: str,
    x_metric_label: str,
    y_metric_label: str,
    is_relative: bool,
    show_pareto_frontier: bool,
    x_lower_is_better: bool,
    y_lower_is_better: bool,
) -> go.Figure:
    # Initialize the Scatters one at a time since we cannot specify multiple different
    # error bar colors from within one trace.
    scatters = [
        go.Scatter(
            x=df[
                (df["trial_status"] == trial_status)
                & ~df[f"{x_metric_name}_mean"].isna()
            ][f"{x_metric_name}_mean"],
            y=df[
                (df["trial_status"] == trial_status)
                & ~df[f"{x_metric_name}_mean"].isna()
            ][f"{y_metric_name}_mean"],
            error_x=(
                {
                    "type": "data",
                    "array": df[df["trial_status"] == trial_status][
                        f"{x_metric_name}_sem"
                    ]
                    * 1.96,
                    "color": trial_status_to_plotly_color(
                        trial_status=trial_status, ci_transparency=True
                    ),
                }
                if not df[f"{x_metric_name}_sem"].isna().all()
                else None
            ),
            error_y=(
                {
                    "type": "data",
                    "array": df[df["trial_status"] == trial_status][
                        f"{y_metric_name}_sem"
                    ]
                    * 1.96,
                    "color": trial_status_to_plotly_color(
                        trial_status=trial_status, ci_transparency=True
                    ),
                }
                if not df[f"{y_metric_name}_sem"].isna().all()
                else None
            ),
            mode="markers",
            marker={
                "color": trial_status_to_plotly_color(
                    trial_status=trial_status, ci_transparency=False
                ),
                "line": {
                    "width": df[df["trial_status"] == trial_status].apply(
                        lambda row: 2
                        if row["p_feasible"] < POSSIBLE_CONSTRAINT_VIOLATION_THRESHOLD
                        else 0,
                        axis=1,
                    ),
                    "color": "red",
                },
            },
            # Apply user-friendly name for UNKNOWN_GENERATION_NODE
            name=trial_status,
            hoverinfo="text",
            text=df[df["trial_status"] == trial_status].apply(
                lambda row: get_arm_tooltip(
                    row=row, metric_names=[x_metric_name, y_metric_name]
                ),
                axis=1,
            ),
        )
        for trial_status in df["trial_status"].unique()
    ]

    figure = go.Figure(data=scatters)
    figure.update_layout(
        xaxis_title=x_metric_label,
        yaxis_title=y_metric_label,
        xaxis_tickformat=".2%" if is_relative else None,
        yaxis_tickformat=".2%" if is_relative else None,
    )

    # Add a red circle with no fill if any arms are marked as possibly infeasible.
    if (df["p_feasible"] < POSSIBLE_CONSTRAINT_VIOLATION_THRESHOLD).any():
        legend_trace = go.Scatter(
            # None here allows us to place a legend item without corresponding points
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "color": "rgba(0, 0, 0, 0)",
                "line": {"width": 2, "color": "red"},
            },
            name="Possible Constraint Violation",
        )

        figure.add_trace(legend_trace)

    # Add horizontal and vertical lines for the status quo.
    if "status_quo" in df["arm_name"].values:
        x = df[df["arm_name"] == "status_quo"][f"{x_metric_name}_mean"].iloc[0]
        y = df[df["arm_name"] == "status_quo"][f"{y_metric_name}_mean"].iloc[0]

        figure.add_shape(
            type="line",
            yref="paper",
            x0=x,
            y0=0,
            x1=x,
            y1=1,
            line={"color": "gray", "dash": "dot"},
        )

        figure.add_shape(
            type="line",
            xref="paper",
            x0=0,
            y0=y,
            x1=1,
            y1=y,
            line={"color": "gray", "dash": "dot"},
        )

    if show_pareto_frontier:
        # Infeasible arms are not included in the Pareto frontier
        eligable_arms = df[df["p_feasible"] >= POSSIBLE_CONSTRAINT_VIOLATION_THRESHOLD]

        # If there are no arms which are not likely to violate constraints, return the
        # figure as is, without adding a Pareto frontier line.
        if len(eligable_arms) == 0:
            return figure

        sorted_df = eligable_arms.sort_values(
            by=f"{x_metric_name}_mean", ascending=x_lower_is_better
        )

        pareto_x = [sorted_df[f"{x_metric_name}_mean"].iloc[0]]
        pareto_y = [sorted_df[f"{y_metric_name}_mean"].iloc[0]]
        for i in range(1, len(sorted_df)):
            if not y_lower_is_better and sorted_df[f"{y_metric_name}_mean"].iloc[
                i
            ] > max(sorted_df[f"{y_metric_name}_mean"].iloc[:i]):
                pareto_x.append(sorted_df[f"{x_metric_name}_mean"].iloc[i])
                pareto_y.append(sorted_df[f"{y_metric_name}_mean"].iloc[i])
            elif y_lower_is_better and sorted_df[f"{y_metric_name}_mean"].iloc[i] < min(
                sorted_df[f"{y_metric_name}_mean"].iloc[:i]
            ):
                pareto_x.append(sorted_df[f"{x_metric_name}_mean"].iloc[i])
                pareto_y.append(sorted_df[f"{y_metric_name}_mean"].iloc[i])

        pareto_trace = go.Scatter(x=pareto_x, y=pareto_y, **BEST_LINE_SETTINGS)

        figure.add_trace(pareto_trace)

    return figure

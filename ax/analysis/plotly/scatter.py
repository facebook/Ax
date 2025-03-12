# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.utils import (
    CANDIDATE_CI_RED,
    CANDIDATE_RED,
    CONFIDENCE_INTERVAL_BLUE,
    MARKER_BLUE,
)
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.observation import ObservationFeatures
from ax.exceptions.core import DataRequiredError, UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from ax.modelbridge.prediction_utils import predict_at_point
from plotly import graph_objects as go


class ScatterPlot(PlotlyAnalysis):
    """
    Plotly Scatter plot for any two metrics. Each arm is represented by a single point.
    Only completed trials are shown.

    Optionally, the Pareto frontier can be shown. This plot is useful for understanding
    the relationship and/or tradeoff between two metrics.

    The DataFrame computed will contain one row per arm and the following columns:
        - trial_index: The trial index of the arm
        - arm_name: The name of the arm
        - X_METRIC_NAME: The observed mean of the metric specified
        - Y_METRIC_NAME: The observed mean of the metric specified
        - is_optimal: Whether the arm is on the Pareto frontier
    """

    def __init__(
        self,
        x_metric_name: str,
        y_metric_name: str,
        show_pareto_frontier: bool = False,
        trial_index: int | None = None,
        arms_to_predict_with_adapter: list[Arm] | None = None,
        fixed_features: ObservationFeatures | None = None,
        metric_name_mapping: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            x_metric_name: The name of the metric to plot on the x-axis.
            y_metric_name: The name of the metric to plot on the y-axis.
            show_pareto_frontier: Whether to show the Pareto frontier for the two
                metrics. Optimization direction is inferred from the Experiment.
            trial_index: Optional trial index to filter the data to. If not specified,
                all trials will be included.
            arms_to_predict_with_adapter: Optional list of Arm objects that were used
                to generate predictions. If specified, these points will be plotted
                alongside insample points
            fixed_features: Optional fixed features to project to trials (e.g.
                relativization), and offers further adhoc flexibility.
            metric_name_mapping: Optional mapping from default metric names to more
                readable metric names.
        """

        self.x_metric_name = x_metric_name
        self.y_metric_name = y_metric_name
        self.show_pareto_frontier = show_pareto_frontier
        self.trial_index = trial_index
        self._arms_to_predict_with_adapter = arms_to_predict_with_adapter
        self._fixed_features = fixed_features
        self._metric_name_mapping = metric_name_mapping

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> PlotlyAnalysisCard:
        if experiment is None:
            raise UserInputError("ScatterPlot requires an Experiment")

        df = _prepare_data(
            experiment=experiment,
            x_metric_name=self.x_metric_name,
            y_metric_name=self.y_metric_name,
            trial_index=self.trial_index,
            adapter=adapter,
            arms_to_predict_with_adapter=self._arms_to_predict_with_adapter,
            fixed_features=self._fixed_features,
            metric_name_mapping=self._metric_name_mapping,
        )
        # replace metric name with human readable names if provided
        x_metric_name = (
            self._metric_name_mapping.get(self.x_metric_name, self.x_metric_name)
            if self._metric_name_mapping is not None
            else self.x_metric_name
        )
        y_metric_name = (
            self._metric_name_mapping.get(self.y_metric_name, self.y_metric_name)
            if self._metric_name_mapping is not None
            else self.y_metric_name
        )
        fig = _prepare_plot(
            df=df,
            x_metric_name=x_metric_name,
            y_metric_name=y_metric_name,
            show_pareto_frontier=self.show_pareto_frontier,
            x_lower_is_better=experiment.metrics[self.x_metric_name].lower_is_better
            or False,
        )

        return self._create_plotly_analysis_card(
            title=f"Observed {x_metric_name} vs. {y_metric_name}",
            subtitle="Compare arms by their observed metric values",
            level=AnalysisCardLevel.HIGH,
            df=df,
            fig=fig,
            category=AnalysisCardCategory.INSIGHT,
        )


def scatter_plot(
    adapter: Adapter,
    experiment: Experiment,
    x_metric_name: str,
    y_metric_name: str,
    trial_index: int | None = None,
    arms_to_predict_with_adapter: list[Arm] | None = None,
    fixed_features: ObservationFeatures | None = None,
    metric_name_mapping: dict[str, str] | None = None,
) -> list[PlotlyAnalysisCard]:
    """
    Exposes scatter plot for adhoc functionality, only to be used in notebook
    setting, and not within Ax code.

    Args:
        adapter: The adapter that will be assessed during cross validation.
        experiment: Experiment associated with this analysis. Used to extract
            the data.
        x_metric_name: The name of the metric to plot on the x-axis.
        y_metric_name: The name of the metric to plot on the y-axis.
        trial_index: Optional trial index to filter the data to. If not specified,
            all trials will be included.
        arms_to_predict_with_adapter: Optional list of Arm objects that were used
                to generate predictions. If specified, these points will be plotted
                alongside insample points
        fixed_features: Optional fixed features to pass with custom arms during
            generation of predictions. This is useful for relativization, etc.
        metric_name_mapping: Optional mapping from default metric names to more
            readable metric names.
    """
    # returning as a list enables easier UX for displaying the cards in a notebook
    return [
        ScatterPlot(
            x_metric_name=x_metric_name,
            y_metric_name=y_metric_name,
            show_pareto_frontier=False,  # Not supported for adhoc use
            trial_index=trial_index,
            arms_to_predict_with_adapter=arms_to_predict_with_adapter,
            fixed_features=fixed_features,
            metric_name_mapping=metric_name_mapping,
        ).compute(
            experiment=experiment,
            adapter=adapter,
        )
    ]


def _prepare_data(
    experiment: Experiment,
    x_metric_name: str,
    y_metric_name: str,
    trial_index: int | None = None,
    adapter: Adapter | None = None,
    arms_to_predict_with_adapter: list[Arm] | None = None,
    fixed_features: ObservationFeatures | None = None,
    metric_name_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Extract the relevant data from the experiment and prepare it into a dataframe
    formatted in the way expected by _prepare_plot.

    Args:
        experiment: The experiment to extract data from.
        x_metric_name: The name of the metric to plot on the x-axis.
        y_metric_name: The name of the metric to plot on the y-axis.
        trial_index: Optional trial index to filter the data to. If not specified,
            all trials will be included.
        arms_to_predict_with_adapter: Optional list of Arm objects that were used
                to generate predictions. If specified, these points will be plotted
                alongside insample points
        fixed_features: Optional fixed features to pass with custom arms during
            generation of predictions. This is useful for relativization, etc.
        metric_name_mapping: Optional mapping from default metric names to more
            readable metric names.

    """
    # Lookup the data that has already been fetched and attached to the experiment
    data = experiment.lookup_data().df

    # Filter for only rows with the relevant metric names and only completed trials
    metric_name_mask = data["metric_name"].isin([x_metric_name, y_metric_name])
    status_mask = data["trial_index"].apply(
        lambda trial_index: experiment.trials[trial_index].status.is_completed
    )
    filtered = data[metric_name_mask & status_mask][
        ["trial_index", "arm_name", "metric_name", "mean", "sem"]
    ]

    # filter data to trial index if specified
    if trial_index is not None:
        filtered = filtered[filtered["trial_index"] == trial_index]

    # If custom adapter and custom GeneratorRun are provided, plot the
    # predicted values for metric x and y given the custom adapter alongside
    # the observed values.
    if adapter is not None and arms_to_predict_with_adapter is not None:
        new_rows = []
        for arm in arms_to_predict_with_adapter:
            obsf = ObservationFeatures.from_arm(arm)
            if fixed_features is not None:
                obsf.update_features(fixed_features)
            # Make a prediction
            try:
                pred_y, pred_se = predict_at_point(
                    adapter, obsf, {x_metric_name, y_metric_name}
                )
            except Exception:
                # Check if it is an out-of-design arm.
                if not adapter.model_space.check_membership(obsf.parameters):
                    # Skip this point
                    continue
                else:
                    # It should have worked
                    raise
            for metric in pred_y:
                new_rows.append(
                    {
                        "trial_index": -1,  # indidcates this is a candidate point
                        "arm_name": arm.name_or_short_signature,
                        "metric_name": metric,
                        "mean": pred_y[metric],
                        "sem": pred_se[metric],
                    }
                )
        new_rows_df = pd.DataFrame(new_rows)
        filtered = pd.concat([filtered, new_rows_df], ignore_index=True)

    # Pivot the data so that each row is an arm and the columns are the metric names
    # and the SEMs for each metric.
    pivoted_mean: pd.DataFrame = filtered.pivot_table(
        index=["trial_index", "arm_name"], columns="metric_name", values="mean"
    ).dropna()
    pivoted_sem: pd.DataFrame = filtered.pivot_table(
        index=["trial_index", "arm_name"], columns="metric_name", values="sem"
    ).dropna()
    pivoted = pivoted_mean.join(pivoted_sem, rsuffix="_sem")
    pivoted.reset_index(inplace=True)
    pivoted.columns.name = None

    if pivoted.empty:
        raise DataRequiredError(
            f"No observations have data for both {x_metric_name} and {y_metric_name}. "
            "Please ensure that the data has been fetched and attached to the "
            "experiment."
        )

    # Add a column indicating whether the arm is on the Pareto frontier. This is
    # calculated by comparing each arm to all other arms in the experiment and
    # creating a mask.
    # If directional guidance is not specified, assume higher is better
    x_lower_is_better: bool = experiment.metrics[x_metric_name].lower_is_better or False
    y_lower_is_better: bool = experiment.metrics[y_metric_name].lower_is_better or False

    pivoted["is_optimal"] = pivoted.apply(
        lambda row: not (
            (
                (pivoted[x_metric_name] < row[x_metric_name])
                if x_lower_is_better
                else (pivoted[x_metric_name] > row[x_metric_name])
            )
            & (
                (pivoted[y_metric_name] < row[y_metric_name])
                if y_lower_is_better
                else (pivoted[y_metric_name] > row[y_metric_name])
            )
        ).any(),
        axis=1,
    )

    # replace metric name with more human readable name if provided
    if metric_name_mapping is not None:
        new_x_name = metric_name_mapping.get(x_metric_name, x_metric_name)
        new_y_name = metric_name_mapping.get(y_metric_name, y_metric_name)
        pivoted = pivoted.rename(
            columns={
                x_metric_name: new_x_name,
                y_metric_name: new_y_name,
                f"{x_metric_name}_sem": f"{new_x_name}_sem",
                f"{y_metric_name}_sem": f"{new_y_name}_sem",
            }
        )

    return pivoted


def _prepare_plot(
    df: pd.DataFrame,
    x_metric_name: str,
    y_metric_name: str,
    show_pareto_frontier: bool,
    x_lower_is_better: bool,
) -> go.Figure:
    """
    Prepare a scatter plot for the given DataFrame.

    Args:
        df: The DataFrame to plot. Must contain the following columns:
            - trial_index: The trial index of the arm
            - arm_name: The name of the arm
            - X_METRIC_NAME: The observed mean of some metric to plot on the x-axis
            - Y_METRIC_NAME: The observed mean of the metric to plot on the y-axis
            - X_METRIC_NAME_SEM: The SEM of the observed mean of the x-axis metric
            - Y_METRIC_NAME_SEM: The SEM of the observed mean of the y-axis metric
            - is_optimal: Whether the arm is on the Pareto frontier (this can be
                omitted if show_pareto_frontier=False)
        x_metric_name: The name of the metric to plot on the x-axis
        y_metric_name: The name of the metric to plot on the y-axis
        show_pareto_frontier: Whether to draw the Pareto frontier for the two metrics
        x_lower_is_better: Whether the metric on the x-axis is being minimized (only
            relevant if show_pareto_frontier=True)
    """
    # Define hover text with conditional trial index display
    hover_text = df.apply(
        lambda row: (
            "Trial: Candidate<br>"
            if row["trial_index"] == -1
            else f"Trial: {row['trial_index']}<br>"
        )
        + f"Arm: {row['arm_name']}<br>"
        + f"{x_metric_name}: {row[x_metric_name]}"
        + f" ±{1.96 * row[f'{x_metric_name}_sem']:.2f}<br>"
        + f"{y_metric_name}: {row[y_metric_name]}"
        + f" ±{1.96 * row[f'{y_metric_name}_sem']:.2f}",
        axis=1,
    )
    # Create a mask for each category
    candidate_mask = df["trial_index"] == -1
    insample_mask = ~candidate_mask

    fig = go.Figure(
        # always plot insample points
        _create_scatter_trace(
            df=df,
            mask=insample_mask,
            x_metric_name=x_metric_name,
            y_metric_name=y_metric_name,
            marker_color=MARKER_BLUE,
            ci_color=CONFIDENCE_INTERVAL_BLUE,
            hover_text=hover_text,
        )
    )

    if show_pareto_frontier:
        # Must sort to ensure we draw the line through optimal points in the correct
        # order.
        frontier_df = df[df["is_optimal"]].sort_values(by=x_metric_name)
        fig.add_trace(
            go.Scatter(
                x=frontier_df[x_metric_name],
                y=frontier_df[y_metric_name],
                mode="lines",
                line_shape="hv" if x_lower_is_better else "vh",
                showlegend=False,
            )
        )

    # Add trace for candidate points (red), will only add points if they exist
    fig.add_trace(
        _create_scatter_trace(
            df=df,
            mask=candidate_mask,
            x_metric_name=x_metric_name,
            y_metric_name=y_metric_name,
            marker_color=CANDIDATE_RED,
            ci_color=CANDIDATE_CI_RED,
            hover_text=hover_text,
            name_for_legend="Candidate",
        )
    )
    # Update layout to add axis labels
    fig.update_layout(xaxis_title=x_metric_name, yaxis_title=y_metric_name)

    return fig


def _create_scatter_trace(
    df: pd.DataFrame,
    mask: pd.DataFrame,
    x_metric_name: str,
    y_metric_name: str,
    marker_color: str,
    ci_color: str,
    hover_text: str,
    name_for_legend: str | None = None,
) -> go.Scatter:
    """
    Helper to create scatter traces for the scatter plot. Allows for customization
    of multiple types of data (insample and candidate) with standardization of the
    look across data types.
    """
    return go.Scatter(
        x=df.loc[mask, x_metric_name],
        y=df.loc[mask, y_metric_name],
        mode="markers",
        marker={"color": marker_color},
        error_x={
            "type": "data",
            "array": df.loc[mask, f"{x_metric_name}_sem"] * 1.96,
            "visible": True,
            "color": ci_color,
        },
        error_y={
            "type": "data",
            "array": df.loc[mask, f"{y_metric_name}_sem"] * 1.96,
            "visible": True,
            "color": ci_color,
        },
        hoverlabel={"bgcolor": ci_color, "font": {"color": "black"}},
        hoverinfo="text",
        text=hover_text[mask],
        name=name_for_legend,
        showlegend=True if name_for_legend is not None else False,
    )

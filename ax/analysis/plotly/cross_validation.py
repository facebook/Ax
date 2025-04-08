# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import Sequence

import pandas as pd
from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.utils import (
    CONFIDENCE_INTERVAL_BLUE,
    get_nudge_value,
    MARKER_BLUE,
    select_metric,
)
from ax.analysis.utils import extract_relevant_adapter
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from ax.modelbridge.cross_validation import cross_validate
from plotly import graph_objects as go


class CrossValidationPlot(PlotlyAnalysis):
    """
    Plotly Scatter plot for cross validation for model predictions using the current
    model on the GenerationStrategy. This plot is useful for understanding how well
    the model is able to predict out-of-sample which in turn is indicative of its
    ability to suggest valuable candidates.

    Splits the model's training data into train/test folds and makes
    out-of-sample predictions on the test folds.

    A well fit model will have points clustered around the y=x line, and a model with
    poor fit may have points in a horizontal band in the center of the plot
    indicating a tendency to predict the observed mean of the specificed metric for
    all arms.

    The DataFrame computed will contain one row per arm and the following columns:
        - arm_name: The name of the arm
        - observed: The observed mean of the metric specified
        - observed_sem: The SEM of the observed mean of the metric specified
        - predicted: The predicted mean of the metric specified
        - predicted_sem: The SEM of the predicted mean of the metric specified
    """

    def __init__(
        self,
        metric_name: str | None = None,
        folds: int = -1,
        untransform: bool = True,
        trial_index: int | None = None,
        refined_metric_name: str | None = None,
    ) -> None:
        """
        Args:
            metric_name: The name of the metric to plot. If not specified the objective
                will be used. Note that the metric cannot be inferred for
                multi-objective or scalarized-objective experiments.
            folds: Number of subsamples to partition observations into. Use -1 for
                leave-one-out cross validation.
            untransform: Whether to untransform the model predictions before cross
                validating. Generators are trained on transformed data, and candidate
                generation is performed in the transformed space. Computing the model
                quality metric based on the cross-validation results in the
                untransformed space may not be representative of the model that
                is actually used for candidate generation in case of non-invertible
                transforms, e.g., Winsorize or LogY. While the model in the
                transformed space may not be representative of the original data in
                regions where outliers have been removed, we have found it to better
                reflect the how good the model used for candidate generation actually
                is.
            trial_index: Optional trial index that the model from generation_strategy
                was used to generate. Useful card attribute to filter to only specific
                trial.
        """

        self.metric_name = metric_name
        self.folds = folds
        self.untransform = untransform
        self.trial_index = trial_index
        self._refined_metric_name = refined_metric_name

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[PlotlyAnalysisCard]:
        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        # If metric name is not provided, try to infer it from the experiment
        metric_name = self.metric_name
        if metric_name is None:
            if experiment is None:
                raise UserInputError(
                    "No metric name is provided, attempting to infer metric name "
                    "from the Experiment object, but no Experiment is provided."
                )
            metric_name = select_metric(experiment=experiment)

        df = _prepare_data(
            adapter=relevant_adapter,
            metric_name=metric_name,
            folds=self.folds,
            untransform=self.untransform,
        )

        fig = _prepare_plot(df=df)
        k_folds_substring = f"{self.folds}-fold" if self.folds > 0 else "leave-one-out"
        nudge = get_nudge_value(metric_name=metric_name, experiment=experiment)

        # If a human readable metric name is provided, use it in the title
        metric_title = self._refined_metric_name or metric_name

        # Define the cross-validation description based on the number of folds
        cv_description = (
            (
                f"the data is split into {self.folds} subsets and the model is "
                f"trained on {self.folds - 1} subsets while the remaining subset "
                "is used for validation"
            )
            if self.folds > 0
            else (
                "the model is trained on all data except one sample, which is "
                "used for validation"
            )
        )
        return [
            self._create_plotly_analysis_card(
                title=f"Cross Validation for {metric_title}",
                subtitle=(
                    "The cross-validation plot displays the model fit for each "
                    f"metric in the experiment. It employs a {k_folds_substring} "
                    f"approach, where {cv_description}. The plot shows the "
                    "predicted outcome for the validation set on the y-axis against "
                    "its actual value on the x-axis. Points that align closely with "
                    "the dotted diagonal line indicate a strong model fit, signifying "
                    "accurate predictions. Additionally, the plot includes 95% "
                    "confidence intervals that provide insight into the noise in "
                    "observations and the uncertainty in model predictions. A "
                    "horizontal, flat line of predictions indicates that the model "
                    "has not picked up on sufficient signal in the data, and instead "
                    "is just predicting the mean."
                ),
                level=AnalysisCardLevel.LOW.value + nudge,
                df=df,
                fig=fig,
                category=AnalysisCardCategory.INSIGHT,
            )
        ]


def cross_validation_adhoc_compute(
    adapter: Adapter,
    data: Data,
    experiment: Experiment | None = None,
    folds: int = -1,
    untransform: bool = True,
    metric_name_mapping: dict[str, str] | None = None,
) -> list[PlotlyAnalysisCard]:
    """
    Helper method to expose adhoc cross validation plotting. This overrides the
    default assumption that the adapter from the generation strategy should be
    used. Only for advanced users in a notebook setting.

    Args:
        adapter: The adapter that will be assessed during cross validation.
        data: The Data that was used to fit the model. Will be used in this
            adhoc cross validation call to compute the cross validation for all
            metrics in the Data object.
        experiment: Experiment associated with this analysis. Used to determine
            the priority of the analysis based on the metric importance in the
            optimization config.
        folds: Number of subsamples to partition observations into. Use -1 for
            leave-one-out cross validation.
        untransform: Whether to untransform the model predictions before cross
            validating. Generators are trained on transformed data, and candidate
            generation is performed in the transformed space. Computing the model
            quality metric based on the cross-validation results in the
            untransformed space may not be representative of the model that
            is actually used for candidate generation in case of non-invertible
            transforms, e.g., Winsorize or LogY. While the model in the
            transformed space may not be representative of the original data in
            regions where outliers have been removed, we have found it to better
            reflect the how good the model used for candidate generation actually
            is.
        metric_name_mapping: Optional mapping from default metric names to more
            readable metric names.
    """
    plots = []
    # Get all unique metric names in the data object, CVs will be computed for
    # all metrics in the data object
    metric_names = list(data.df["metric_name"].unique())
    for metric_name in metric_names:
        # replace metric name with human readable name if mapping is provided
        refined_metric_name = (
            metric_name_mapping.get(metric_name, metric_name)
            if metric_name_mapping
            else metric_name
        )
        plots.append(
            *CrossValidationPlot(
                metric_name=metric_name,
                folds=folds,
                untransform=untransform,
                refined_metric_name=refined_metric_name,
            ).compute(experiment=experiment, generation_strategy=None, adapter=adapter)
        )
    return plots


def _prepare_data(
    adapter: Adapter,
    metric_name: str,
    folds: int,
    untransform: bool,
) -> pd.DataFrame:
    cv_results = cross_validate(
        model=adapter,
        folds=folds,
        untransform=untransform,
    )

    records = []
    for observed, predicted in cv_results:
        # Find the index of the metric in observed and predicted
        observed_i = next(
            (
                i
                for i, name in enumerate(observed.data.metric_names)
                if name == metric_name
            ),
            None,
        )
        predicted_i = next(
            (i for i, name in enumerate(predicted.metric_names) if name == metric_name),
            None,
        )
        # Check if both indices are found
        if observed_i is not None and predicted_i is not None:
            record = {
                "arm_name": observed.arm_name,
                "observed": observed.data.means[observed_i],
                "predicted": predicted.means[predicted_i],
                # Compute the 95% confidence intervals for plotting purposes
                "observed_95_ci": observed.data.covariance[observed_i][observed_i]
                ** 0.5
                * 1.96,
                "predicted_95_ci": predicted.covariance[predicted_i][predicted_i] ** 0.5
                * 1.96,
            }
            records.append(record)
    return pd.DataFrame.from_records(records)


def _prepare_plot(
    df: pd.DataFrame,
) -> go.Figure:
    # Create a scatter plot using Plotly Graph Objects for more control
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["observed"],
            y=df["predicted"],
            mode="markers",
            marker={
                "color": MARKER_BLUE,
            },
            error_x={
                "type": "data",
                "array": df["observed_95_ci"],
                "visible": True,
                "color": CONFIDENCE_INTERVAL_BLUE,
            },
            error_y={
                "type": "data",
                "array": df["predicted_95_ci"],
                "visible": True,
                "color": CONFIDENCE_INTERVAL_BLUE,
            },
            text=df["arm_name"],
            hovertemplate=(
                "<b>Arm Name: %{text}</b><br>"
                + "Predicted: %{y}<br>"
                + "Observed: %{x}<br>"
                + "<extra></extra>"  # Removes the trace name from the hover
            ),
            hoverlabel={
                "bgcolor": CONFIDENCE_INTERVAL_BLUE,
                "font": {"color": "black"},
            },
        )
    )

    # Add a gray dashed line at y=x starting and ending just outside of the region of
    # interest for reference. A well fit model should have points clustered around
    # this line.
    lower_bound = (
        min(
            (df["observed"] - df["observed_95_ci"].fillna(0)).min(),
            (df["predicted"] - df["predicted_95_ci"].fillna(0)).min(),
        )
        * 0.999  # tight autozoom
    )
    upper_bound = (
        max(
            (df["observed"] + df["observed_95_ci"].fillna(0)).max(),
            (df["predicted"] + df["predicted_95_ci"].fillna(0)).max(),
        )
        * 1.001  # tight autozoom
    )
    fig.add_shape(
        type="line",
        x0=lower_bound,
        y0=lower_bound,
        x1=upper_bound,
        y1=upper_bound,
        line={"color": "gray", "dash": "dot"},
    )

    # Update axes with tight autozoom that remains square
    fig.update_xaxes(
        range=[lower_bound, upper_bound], constrain="domain", title="Actual Outcome"
    )
    fig.update_yaxes(
        range=[lower_bound, upper_bound],
        scaleanchor="x",
        scaleratio=1,
        title="Predicted Outcome",
    )
    return fig

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.utils import select_metric
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.exceptions.core import UserInputError
from ax.modelbridge.cross_validation import cross_validate
from ax.modelbridge.generation_strategy import GenerationStrategy
from plotly import express as px, graph_objects as go
from pyre_extensions import assert_is_instance, none_throws


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
    ) -> None:
        """
        Args:
            metric_name: The name of the metric to plot. If not specified the objective
                will be used. Note that the metric cannot be inferred for
                multi-objective or scalarized-objective experiments.
            folds: Number of subsamples to partition observations into. Use -1 for
                leave-one-out cross validation.
            untransform: Whether to untransform the model predictions before cross
                validating. Models are trained on transformed data, and candidate
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
                was used to generate.  We should therefore only have observations from
                trials prior to this trial index in our plot.  If this is not True, we
                should error out.
        """

        self.metric_name = metric_name
        self.folds = folds
        self.untransform = untransform
        self.trial_index = trial_index

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> PlotlyAnalysisCard:
        if generation_strategy is None:
            raise UserInputError("CrossValidation requires a GenerationStrategy")

        metric_name = self.metric_name or select_metric(
            experiment=generation_strategy.experiment
        )

        df = _prepare_data(
            # CrossValidationPlot requires a native Ax GenerationStrategy and cannot be
            # used with a GenerationStrategyInterface.
            generation_strategy=assert_is_instance(
                generation_strategy, GenerationStrategy
            ),
            metric_name=metric_name,
            folds=self.folds,
            untransform=self.untransform,
            trial_index=self.trial_index,
        )
        fig = _prepare_plot(df=df)

        k_folds_substring = f"{self.folds}-fold" if self.folds > 0 else "leave-one-out"

        # Nudge the priority if the metric is important to the experiment
        if (
            experiment is not None
            and (optimization_config := experiment.optimization_config) is not None
            and (objective := optimization_config.objective) is not None
            and metric_name in objective.metric_names
        ):
            nudge = 2
        elif (
            experiment is not None
            and (optimization_config := experiment.optimization_config) is not None
            and metric_name in optimization_config.outcome_constraints
        ):
            nudge = 1
        else:
            nudge = 0

        return self._create_plotly_analysis_card(
            title=f"Cross Validation for {metric_name}",
            subtitle=f"Out-of-sample predictions using {k_folds_substring} CV",
            level=AnalysisCardLevel.LOW.value + nudge,
            df=df,
            fig=fig,
        )


def _prepare_data(
    generation_strategy: GenerationStrategy,
    metric_name: str,
    folds: int,
    untransform: bool,
    trial_index: int | None,
) -> pd.DataFrame:
    # If model is not fit already, fit it
    if generation_strategy.model is None:
        generation_strategy._fit_current_model(None)

    cv_results = cross_validate(
        model=none_throws(generation_strategy.model),
        folds=folds,
        untransform=untransform,
    )

    records = []
    for observed, predicted in cv_results:
        if trial_index is not None:
            if (
                observed.features.trial_index is not None
                and observed.features.trial_index >= trial_index
            ):
                raise UserInputError(
                    "CrossValidationPlot was specified to be for the generation of "
                    f"trial {trial_index}, but has observations from trial "
                    f"{observed.features.trial_index}."
                )
        for i in range(len(observed.data.metric_names)):
            # Find the index of the metric we want to plot
            if not (
                observed.data.metric_names[i] == metric_name
                and predicted.metric_names[i] == metric_name
            ):
                continue

            record = {
                "arm_name": observed.arm_name,
                "observed": observed.data.means[i],
                "predicted": predicted.means[i],
                # Take the square root of the the SEM to get the standard deviation
                "observed_sem": observed.data.covariance[i][i] ** 0.5,
                "predicted_sem": predicted.covariance[i][i] ** 0.5,
            }
            records.append(record)
            break

    return pd.DataFrame.from_records(records)


def _prepare_plot(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="observed",
        y="predicted",
        error_x="observed_sem",
        error_y="predicted_sem",
        hover_data=["arm_name", "observed", "predicted"],
    )

    # Add a gray dashed line at y=x starting and ending just outside of the region of
    # interest for reference. A well fit model should have points clustered around this
    # line.
    lower_bound = (
        min(
            (df["observed"] - df["observed_sem"].fillna(0)).min(),
            (df["predicted"] - df["predicted_sem"].fillna(0)).min(),
        )
        * 0.99
    )
    upper_bound = (
        max(
            (df["observed"] + df["observed_sem"].fillna(0)).max(),
            (df["predicted"] + df["predicted_sem"].fillna(0)).max(),
        )
        * 1.01
    )

    fig.add_shape(
        type="line",
        x0=lower_bound,
        y0=lower_bound,
        x1=upper_bound,
        y1=upper_bound,
        line={"color": "gray", "dash": "dot"},
    )

    # Force plot to display as a square
    fig.update_xaxes(range=[lower_bound, upper_bound], constrain="domain")
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig

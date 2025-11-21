# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import final, Mapping, Sequence

import pandas as pd
from ax.adapter.base import Adapter
from ax.adapter.cross_validation import cross_validate, CVResult
from ax.analysis.analysis import Analysis
from ax.analysis.analysis_card import AnalysisCardBase
from ax.analysis.plotly.color_constants import AX_BLUE
from ax.analysis.plotly.plotly_analysis import create_plotly_analysis_card
from ax.analysis.plotly.utils import get_scatter_point_color, Z_SCORE_95_CI
from ax.analysis.utils import extract_relevant_adapter, validate_adapter_can_predict
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from plotly import graph_objects as go
from pyre_extensions import override

CV_CARDGROUP_TITLE = "Cross Validation: Assessing model fit"

CV_CARDGROUP_SUBTITLE = (
    "Cross-validation plots display the model fit for each metric in the "
    "experiment. The model is trained on a subset of the data and then predicts the "
    "outcome for the remaining subset. The plots show the predicted outcome for the "
    "validation set on the y-axis against its actual value on the x-axis. Points "
    "that align closely with the dotted diagonal line indicate a strong model fit, "
    "signifying accurate predictions. Additionally, the plots include "
    "confidence intervals that provide insight into the noise in observations and "
    "the uncertainty in model predictions. <br><br>"
    "NOTE: A horizontal, flat line of predictions "
    "indicates that the model has not picked up on sufficient signal in the data, "
    "and instead is just predicting the mean."
)


@final
class CrossValidationPlot(Analysis):
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
        metric_names: Sequence[str] | None = None,
        folds: int = -1,
        untransform: bool = False,
        trial_index: int | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """
        Args:
            metric_names: The name of the metric to plot. If not specified all metrics
                available on the underyling model will be used.
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
            labels: Optional dictionary of labels for the plot. Useful for when metric
                names are too long or otherwise challenging to read.
        """

        self.metric_names = metric_names
        self.folds = folds
        self.untransform = untransform
        self.trial_index = trial_index
        self.labels: dict[str, str] = {**labels} if labels is not None else {}

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        CrossValidationPlot requires only an Adapter which can predict.
        """
        return validate_adapter_can_predict(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
            required_metric_names=None,
        )

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardBase:
        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        cards = []
        cv_results = cross_validate(
            model=relevant_adapter, folds=self.folds, untransform=self.untransform
        )
        relevant_adapter_metric_names = [
            relevant_adapter._experiment.signature_to_metric[signature].name
            for signature in relevant_adapter._metric_signatures
        ]
        for metric_name in self.metric_names or relevant_adapter_metric_names:
            df = _prepare_data(
                metric_name=metric_name, cv_results=cv_results, adapter=relevant_adapter
            )

            fig = _prepare_plot(df=df)

            k_folds_substring = (
                f"{self.folds}-fold" if self.folds > 0 else "leave-one-out"
            )

            # If a human readable metric name is provided, use it in the title
            metric_title = self.labels.get(metric_name, metric_name)

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

            card = create_plotly_analysis_card(
                name=self.__class__.__name__,
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
                df=df,
                fig=fig,
            )

            cards.append(card)

        return self._create_analysis_card_group(
            title=CV_CARDGROUP_TITLE,
            subtitle=CV_CARDGROUP_SUBTITLE,
            children=cards,
        )


def compute_cross_validation_adhoc(
    metric_names: Sequence[str] | None = None,
    folds: int = -1,
    untransform: bool = True,
    labels: Mapping[str, str] | None = None,
    experiment: Experiment | None = None,
    generation_strategy: GenerationStrategy | None = None,
    adapter: Adapter | None = None,
) -> AnalysisCardBase:
    """
    Helper method to expose adhoc cross validation plotting. Only for advanced users in
    a notebook setting.

    Args:
        metric_names: The name of the metric to plot. If not specified all metrics
            available on the underyling model will be used.
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
        labels: Optional dictionary of labels for the plot. Useful for when metric
            names are too long or otherwise challenging to read.
        experiment: Optional. The experiment to extract data from.
        generation_strategy: Optional. The generation strategy to extract the adapter
            from.
        adapter: Optional. The adapter to cross validate. If provided, this adapter
            will be used instead of the current adapter on the ``GenerationStrategy``
    """
    relevant_adapter = extract_relevant_adapter(
        experiment=experiment,
        generation_strategy=generation_strategy,
        adapter=adapter,
    )

    analysis = CrossValidationPlot(
        metric_names=metric_names,
        folds=folds,
        untransform=untransform,
        labels=labels,
    )

    return analysis.compute(
        experiment=experiment,
        adapter=relevant_adapter,
    )


def _prepare_data(
    metric_name: str, cv_results: list[CVResult], adapter: Adapter
) -> pd.DataFrame:
    records = []
    for observed, predicted in cv_results:
        observed_metric_names = []
        predicted_metric_names = []
        for signature in observed.data.metric_signatures:
            observed_metric_names.append(
                adapter._experiment.signature_to_metric[signature].name
            )
        for signature in predicted.metric_signatures:
            predicted_metric_names.append(
                adapter._experiment.signature_to_metric[signature].name
            )

        # Find the index of the metric in observed and predicted
        observed_i = next(
            (i for i, name in enumerate(observed_metric_names) if name == metric_name),
            None,
        )
        predicted_i = next(
            (i for i, name in enumerate(predicted_metric_names) if name == metric_name),
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
                * Z_SCORE_95_CI,
                "predicted_95_ci": predicted.covariance[predicted_i][predicted_i] ** 0.5
                * Z_SCORE_95_CI,
            }
            records.append(record)
    return pd.DataFrame.from_records(records)


def _prepare_plot(
    df: pd.DataFrame,
) -> go.Figure:
    # Create a scatter plot using Plotly Graph Objects for more control
    fig = go.Figure()
    TRANSPARENT_AX_BLUE: str = get_scatter_point_color(
        hex_color=AX_BLUE,
        ci_transparency=True,
    )
    FILLED_AX_BLUE: str = get_scatter_point_color(
        hex_color=AX_BLUE,
        ci_transparency=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["observed"],
            y=df["predicted"],
            mode="markers",
            marker={
                "color": FILLED_AX_BLUE,
            },
            error_x={
                "type": "data",
                "array": df["observed_95_ci"],
                "visible": True,
                "color": TRANSPARENT_AX_BLUE,
            },
            error_y={
                "type": "data",
                "array": df["predicted_95_ci"],
                "visible": True,
                "color": TRANSPARENT_AX_BLUE,
            },
            text=df["arm_name"],
            hovertemplate=(
                "<b>Arm Name: %{text}</b><br>"
                + "Predicted: %{y}<br>"
                + "Observed: %{x}<br>"
                + "<extra></extra>"  # Removes the trace name from the hover
            ),
            hoverlabel={
                "bgcolor": TRANSPARENT_AX_BLUE,
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

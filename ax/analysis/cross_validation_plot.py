# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from typing import Any, Dict, List, Optional

import pandas as pd

from ax.analysis.base_plotly_visualization import BasePlotlyVisualization

from ax.analysis.helpers.cross_validation_helpers import (
    cv_results_to_df,
    diagonal_trace,
    get_plotting_limit_ignore_outliers,
)

from ax.analysis.helpers.layout_helpers import layout_format, updatemenus_format

from ax.analysis.helpers.scatter_helpers import (
    error_scatter_trace_from_df,
    extract_mean_and_error_from_df,
)

from ax.core.experiment import Experiment
from ax.modelbridge import ModelBridge

from ax.modelbridge.cross_validation import cross_validate, CVResult

from plotly import graph_objs as go


class CrossValidationPlot(BasePlotlyVisualization):
    CROSS_VALIDATION_CAPTION = (
        "<b>NOTE:</b> We have tried our best to only plot the region of interest.<br>"
        "This may hide outliers. You can autoscale the axes to see all trials."
    )

    def __init__(
        self,
        experiment: Experiment,
        model: ModelBridge,
        label_dict: Optional[Dict[str, str]] = None,
        caption: str = CROSS_VALIDATION_CAPTION,
    ) -> None:
        """
        Args:
        experiment: Experiment containing trials to plot
        model: ModelBridge to cross validate against
        label_dict: optional map from real metric names to shortened names
        caption: text to display below the plot
        """
        self.model = model
        self.cv: List[CVResult] = cross_validate(model=model)

        self.label_dict: Optional[Dict[str, str]] = label_dict
        if self.label_dict:
            self.cv = self.remap_label(cv_results=self.cv, label_dict=self.label_dict)

        self.metric_names: List[str] = list(
            set().union(*(cv_result.predicted.metric_names for cv_result in self.cv))
        )
        self.caption = caption

        super().__init__(experiment=experiment)

    def get_df(self) -> pd.DataFrame:
        """
        Overrides BaseAnalysis.get_df()

        Returns:
            df representation of the cross validation results.
            columns:
            {
                "arm_name": name of the arm in the cross validation result
                "metric_name": name of the observed/predicted metric
                "x": value observed for the metric for this arm
                "x_se": standard error of observed metric (0 for observations)
                "y": value predicted for the metric for this arm
                "y_se": standard error of predicted metric for this arm
                "arm_parameters": Parametrization of the arm
            }
        """

        df = pd.concat(
            [
                cv_results_to_df(
                    cv_results=self.cv,
                    metric_name=metric,
                )
                for metric in self.metric_names
            ]
        )

        return df

    @staticmethod
    def compose_annotation(
        caption: str, x: float = 0.0, y: float = -0.15
    ) -> List[Dict[str, Any]]:
        """Composes an annotation dict for use in Plotly figure.
        args:
            caption: str to use for dropdown text
            x: x position of the annotation
            y: y position of the annotation

        returns:
            Annotation dict for use in Plotly figure.
        """
        return [
            {
                "showarrow": False,
                "text": caption,
                "x": x,
                "xanchor": "left",
                "xref": "paper",
                "y": y,
                "yanchor": "top",
                "yref": "paper",
                "align": "left",
            },
        ]

    @staticmethod
    def remap_label(
        cv_results: List[CVResult], label_dict: Dict[str, str]
    ) -> List[CVResult]:
        """Remaps labels in cv_results according to label_dict.

        Args:
            cv_results: A CVResult for each observation in the training data.
            label_dict: optional map from real metric names to shortened names

        Returns:
            A CVResult with metric names mapped from label_dict.
        """
        cv_results = deepcopy(cv_results)  # Copy and edit in-place
        for cv_i in cv_results:
            cv_i.observed.data.metric_names = [
                label_dict.get(m, m) for m in cv_i.observed.data.metric_names
            ]
            cv_i.predicted.metric_names = [
                label_dict.get(m, m) for m in cv_i.predicted.metric_names
            ]
        return cv_results

    def obs_vs_pred_dropdown_plot(
        self,
        xlabel: str = "Actual Outcome",
        ylabel: str = "Predicted Outcome",
    ) -> go.Figure:
        """Plot a dropdown plot of observed vs. predicted values from the
        cross validation results.

        Args:
            xlabel: Label for x-axis.
            ylabel: Label for y-axis.
        """
        traces = []
        metric_dropdown = []
        layout_axis_range = []

        # Get the union of all metric_names seen in predictions
        metric_names = self.metric_names
        df = self.get_df()

        for i, metric in enumerate(metric_names):
            metric_filtered_df = df.loc[df["metric_name"] == metric]

            y_raw, se_raw, y_hat, se_hat = extract_mean_and_error_from_df(
                metric_filtered_df
            )

            # Use the min/max of the limits
            layout_range, diagonal_trace_range = get_plotting_limit_ignore_outliers(
                x=y_raw, y=y_hat, se_x=se_raw, se_y=se_hat
            )
            layout_axis_range.append(layout_range)

            # add a diagonal dotted line to plot
            traces.append(
                diagonal_trace(
                    diagonal_trace_range[0],
                    diagonal_trace_range[1],
                    visible=(i == 0),
                )
            )

            traces.append(
                error_scatter_trace_from_df(
                    df=metric_filtered_df,
                    show_CI=True,
                    visible=(i == 0),
                    x_axis_label="Actual Outcome",
                    y_axis_label="Predicted Outcome",
                )
            )

            # only the first two traces are visible (corresponding to first outcome
            # in dropdown)
            is_visible = [False] * (len(metric_names) * 2)
            is_visible[2 * i] = True
            is_visible[2 * i + 1] = True

            # on dropdown change, restyle
            metric_dropdown.append(
                {
                    "args": [
                        {"visible": is_visible},
                        {
                            "xaxis.range": layout_axis_range[-1],
                            "yaxis.range": layout_axis_range[-1],
                        },
                    ],
                    "label": metric,
                    "method": "update",
                }
            )

        updatemenus = updatemenus_format(metric_dropdown=metric_dropdown)
        layout = layout_format(
            layout_axis_range_value=layout_axis_range[0],
            xlabel=xlabel,
            ylabel=ylabel,
            updatemenus=updatemenus,
        )

        return go.Figure(data=traces, layout=layout)

    def get_fig(self) -> go.Figure:
        """
        Interactive cross-validation (CV) plotting; select metric via dropdown.
        Note: uses the Plotly version of dropdown (which means that all data is
        stored within the notebook).

        Returns:
            go.Figure: Plotly figure with cross validation plot
        """
        caption = self.caption

        fig = self.obs_vs_pred_dropdown_plot()

        current_bmargin = fig["layout"]["margin"].b or 90
        caption_height = 100 * (len(caption) > 0)
        fig["layout"]["margin"].b = current_bmargin + caption_height
        fig["layout"]["height"] += caption_height
        fig["layout"]["annotations"] += tuple(self.compose_annotation(caption))
        fig["layout"]["title"] = "Cross-validation"
        return fig

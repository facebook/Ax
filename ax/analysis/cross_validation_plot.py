# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import pandas as pd

from ax.analysis.base_plotly_visualization import BasePlotlyVisualization

from ax.analysis.helpers.cross_validation_helpers import (
    get_cv_plot_data,
    obs_vs_pred_dropdown_plot,
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
    ) -> None:
        """
        Args:
        experiment: Experiment containing trials to plot
        objective_name: Objective name used to color lines between
            parallel plots

        # potential future args
            label_dict: optional map from real metric names to shortened names
        """
        self.model = model
        self.cv: List[CVResult] = cross_validate(model=model)

        # potential args
        self.label_dict: Optional[Dict[str, str]] = None

        super().__init__(experiment=experiment)

    def get_df(self) -> pd.DataFrame:
        """
        Return a df representation of the cross validation results.
        return pd.DataFrame(self.cv)
        """
        cv_dict = [cv.observed.__dict__ | cv.predicted.__dict__ for cv in self.cv]
        cv_results_dataframe = pd.DataFrame.from_records(cv_dict)
        return cv_results_dataframe

    @staticmethod
    def compose_annotation(
        caption: str, x: float = 0.0, y: float = -0.15
    ) -> List[Dict[str, Any]]:
        if not caption:
            return []
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

    def get_fig(self) -> go.Figure:
        """Plot trials as a parallel coordinates graph"""

        """Interactive cross-validation (CV) plotting; select metric via dropdown.

        Note: uses the Plotly version of dropdown (which means that all data is
        stored within the notebook).

        Args:
            cv_results: cross-validation results.
            show_context: if True, show context on hover.

        Returns:
            go.Figure: Parellel coordinates plot of all experiment trials
        """
        caption = self.CROSS_VALIDATION_CAPTION

        data = get_cv_plot_data(self.cv, label_dict=self.label_dict)
        fig = obs_vs_pred_dropdown_plot(
            data=data,
        )
        current_bmargin = fig["layout"]["margin"].b or 90
        caption_height = 100 * (len(caption) > 0)
        fig["layout"]["margin"].b = current_bmargin + caption_height
        fig["layout"]["height"] += caption_height
        fig["layout"]["annotations"] += tuple(self.compose_annotation(caption))
        fig["layout"]["title"] = "Cross-validation"
        return fig

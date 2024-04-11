# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, Set

import numpy as np

import pandas as pd

from ax.analysis.base_plotly_visualization import BasePlotlyVisualization

from ax.analysis.helpers.layout_helpers import updatemenus_format

from ax.analysis.helpers.plot_data_df_helpers import get_plot_data_in_sample_arms_df
from ax.analysis.helpers.plot_helpers import arm_name_to_sort_key, resize_subtitles

from ax.analysis.helpers.scatter_helpers import (
    error_dot_plot_trace_from_df,
    relativize_dataframe,
)

from ax.core.experiment import Experiment

from ax.exceptions.core import UnsupportedPlotError

from ax.modelbridge import ModelBridge

from plotly import graph_objs as go


class PredictedOutcomesDotPlot(BasePlotlyVisualization):
    def __init__(
        self,
        experiment: Experiment,
        model: ModelBridge,
    ) -> None:
        """
        Args:
            experiment: The experiment associated with this plot
            model: model which is used to fetch the plotting data.
        """

        self.model = model
        self.metrics: Set[str] = model.metric_names
        if model.status_quo is None or model.status_quo.arm_name is None:
            raise UnsupportedPlotError(
                "status quo must be specified for PredictedOutcomesDotPlot"
            )
        self.status_quo_name: str = model.status_quo.arm_name

        super().__init__(experiment=experiment)

    def get_df(self) -> pd.DataFrame:
        """
        Returns:
            A dataframe containing:
            {
                "arm_name": name of the arm in the cross validation result
                "metric_name": name of the observed/predicted metric
                "x": value of the observation for the metric for this arm
                "x_se": standard error of the observation for the metric of this arm
                "y": value predicted for the metric for this arm
                "y_se": standard error of predicted metric for this arm
                "arm_parameters": Parametrization of the arm
                "rel": whether the data is relativized with respect to status quo
            }"""
        return relativize_dataframe(
            get_plot_data_in_sample_arms_df(
                model=self.model, metric_names=self.metrics
            ),
            status_quo_name=self.status_quo_name,
        )

    def get_fig(
        self,
    ) -> go.Figure:
        """
        For each metric, we plot the predicted values for each arm along with its CI
        These values are relativized with respect to the status quo.
        """
        name_order_axes: Dict[str, Dict[str, Any]] = {}

        in_sample_df = self.get_df()
        traces = []
        metric_dropdown = []

        for i, metric in enumerate(self.metrics):
            filtered_df = in_sample_df.loc[in_sample_df["metric_name"] == metric]
            data_single: Dict[str, Any] = error_dot_plot_trace_from_df(
                df=filtered_df, show_CI=True, visible=(i == 0)
            )

            # order arm name sorting arm numbers within batch
            names_by_arm = sorted(
                np.unique(data_single["x"]),
                key=lambda x: arm_name_to_sort_key(x),
                reverse=True,
            )

            name_order_axes["xaxis{}".format(i + 1)] = {
                "categoryorder": "array",
                "categoryarray": names_by_arm,
                "type": "category",
            }
            name_order_axes["yaxis{}".format(i + 1)] = {
                "ticksuffix": "%",
                "zerolinecolor": "red",
            }

            traces.append(data_single)

            is_visible = [False] * (len(metric))
            is_visible[i] = True
            metric_dropdown.append(
                {
                    "args": [
                        {
                            "visible": is_visible,
                        },
                    ],
                    "label": metric,
                    "method": "update",
                }
            )

        updatemenus = updatemenus_format(metric_dropdown=metric_dropdown)

        fig = go.Figure(data=traces)

        fig["layout"].update(
            updatemenus=updatemenus,
            width=1030,
            height=500,
            **name_order_axes,
        )

        fig = resize_subtitles(figure=fig, size=10)
        fig["layout"]["title"] = "Predicted Outcomes by Metric"
        return fig

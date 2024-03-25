# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, Set

import numpy as np

import pandas as pd

from ax.analysis.base_plotly_visualization import BasePlotlyVisualization

from ax.analysis.helpers.plot_data_df_helpers import get_plot_data_in_sample_arms_df

from ax.analysis.helpers.scatter_helpers import error_scatter_trace_from_df

from ax.core.experiment import Experiment

from ax.modelbridge import ModelBridge

from plotly import graph_objs as go, subplots


class TileFittedPlot(BasePlotlyVisualization):
    def __init__(
        self,
        experiment: Experiment,
        model: ModelBridge,
    ) -> None:
        """ """

        self.model = model
        self.metrics: Set[str] = model.metric_names

        super().__init__(experiment=experiment)

    def get_df(self) -> pd.DataFrame:
        """ """
        return get_plot_data_in_sample_arms_df(
            model=self.model, metric_names=self.metrics
        )

    def get_fig(
        self,
    ) -> go.Figure:
        """Tile version of fitted outcome plots."""
        metrics = self.metrics
        nrows = int(np.ceil(len(metrics) / 2))
        ncols = min(len(metrics), 2)

        subplot_titles = metrics

        fig = subplots.make_subplots(
            rows=nrows,
            cols=ncols,
            print_grid=False,
            shared_xaxes=False,
            shared_yaxes=False,
            subplot_titles=tuple(subplot_titles),
            horizontal_spacing=0.05,
            vertical_spacing=0.30 / nrows,
        )

        name_order_args: Dict[str, Any] = {}
        name_order_axes: Dict[str, Dict[str, Any]] = {}
        effect_order_args: Dict[str, Any] = {}

        in_sample_df = self.get_df()

        for i, metric in enumerate(metrics):
            filtered_df = in_sample_df.loc[in_sample_df["metric_name"] == metric]
            data: Dict[str, Any] = error_scatter_trace_from_df(
                df=filtered_df,
                show_CI=True,
            )

            # order arm name sorting arm numbers within batch
            """names_by_arm = sorted(
                np.unique(np.concatenate([d["x"] for d in data])),
                key=lambda x: arm_name_to_sort_key(x),
            )"""
            names_by_arm = []

            # get arm names sorted by effect size
            """names_by_effect = list(
                OrderedDict.fromkeys(
                    np.concatenate([d["x"] for d in data])
                    .flatten()
                    .take(np.argsort(np.concatenate([d["y"] for d in data]).flatten()))
                )
            )"""
            names_by_effect = []

            # options for ordering arms (x-axis)
            # Note that xaxes need to be references as xaxis, xaxis2, xaxis3, etc.
            # for the purposes of updatemenus argument (dropdown) in layout.
            # However, when setting the initial ordering layout, the keys should be
            # xaxis1, xaxis2, xaxis3, etc. Note the discrepancy for the initial
            # axis.
            label = "" if i == 0 else i + 1
            name_order_args["xaxis{}.categoryorder".format(label)] = "array"
            name_order_args["xaxis{}.categoryarray".format(label)] = names_by_arm
            effect_order_args["xaxis{}.categoryorder".format(label)] = "array"
            effect_order_args["xaxis{}.categoryarray".format(label)] = names_by_effect
            name_order_axes["xaxis{}".format(i + 1)] = {
                "categoryorder": "array",
                "categoryarray": names_by_arm,
                "type": "category",
            }
            name_order_axes["yaxis{}".format(i + 1)] = {
                "ticksuffix": "",
                "zerolinecolor": "red",
            }

            fig.append_trace(  # pyre-ignore[16]
                data, int(np.floor(i / ncols)) + 1, i % ncols + 1
            )

        order_options = [
            {"args": [name_order_args], "label": "Name", "method": "relayout"},
            {"args": [effect_order_args], "label": "Effect Size", "method": "relayout"},
        ]

        # if odd number of plots, need to manually remove the last blank subplot
        # generated by `subplots.make_subplots`
        if len(metrics) % 2 == 1:
            fig["layout"].pop("xaxis{}".format(nrows * ncols))
            fig["layout"].pop("yaxis{}".format(nrows * ncols))

        # allocate 400 px per plot
        fig["layout"].update(
            margin={"t": 0},
            hovermode="closest",
            updatemenus=[
                {
                    "x": 0.15,
                    "y": 1 + 0.40 / nrows,
                    "buttons": order_options,
                    "xanchor": "left",
                    "yanchor": "middle",
                }
            ],
            font={"size": 10},
            width=650 if ncols == 1 else 950,
            height=300 * nrows,
            legend={
                "orientation": "h",
                "x": 0,
                "y": 1 + 0.20 / nrows,
                "xanchor": "left",
                "yanchor": "middle",
            },
            **name_order_axes,
        )

        # append dropdown annotations
        fig["layout"]["annotations"] = fig["layout"]["annotations"] + (
            {
                "x": 0.5,
                "y": 1 + 0.40 / nrows,
                "xref": "paper",
                "yref": "paper",
                "font": {"size": 14},
                "text": "Predicted Outcomes",
                "showarrow": False,
                "xanchor": "center",
                "yanchor": "middle",
            },
            {
                "x": 0.05,
                "y": 1 + 0.40 / nrows,
                "xref": "paper",
                "yref": "paper",
                "text": "Sort By",
                "showarrow": False,
                "xanchor": "left",
                "yanchor": "middle",
            },
        )

        # fig = resize_subtitles(figure=fig, size=10)
        return fig

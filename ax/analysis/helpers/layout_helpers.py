# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, List, Tuple, Type

import plotly.graph_objs as go


def updatemenus_format(metric_dropdown: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Formats for use in the cross validation plot
    """
    return [
        {
            "x": 0,
            "y": 1.125,
            "yanchor": "top",
            "xanchor": "left",
            "buttons": metric_dropdown,
        },
        {
            "buttons": [
                {
                    "args": [
                        {
                            "error_x.width": 4,
                            "error_x.thickness": 2,
                            "error_y.width": 4,
                            "error_y.thickness": 2,
                        }
                    ],
                    "label": "Yes",
                    "method": "restyle",
                },
                {
                    "args": [
                        {
                            "error_x.width": 0,
                            "error_x.thickness": 0,
                            "error_y.width": 0,
                            "error_y.thickness": 0,
                        }
                    ],
                    "label": "No",
                    "method": "restyle",
                },
            ],
            "x": 1.125,
            "xanchor": "left",
            "y": 0.8,
            "yanchor": "middle",
        },
    ]


def layout_format(
    layout_axis_range_value: Tuple[float, float],
    xlabel: str,
    ylabel: str,
    updatemenus: List[Dict[str, Any]],
) -> Type[go.Figure]:
    """
    Constructs a layout object for a CrossValidation figure.
    args:
        layout_axis_range_value: A tuple containing the range of values
            for the x-axis and y-axis.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        updatemenus: A list of dictionaries containing information to use on update.
    """
    layout = go.Layout(
        annotations=[
            {
                "showarrow": False,
                "text": "Show CI",
                "x": 1.125,
                "xanchor": "left",
                "xref": "paper",
                "y": 0.9,
                "yanchor": "middle",
                "yref": "paper",
            }
        ],
        xaxis={
            "range": layout_axis_range_value,
            "title": xlabel,
            "zeroline": False,
            "mirror": True,
            "linecolor": "black",
            "linewidth": 0.5,
        },
        yaxis={
            "range": layout_axis_range_value,
            "title": ylabel,
            "zeroline": False,
            "mirror": True,
            "linecolor": "black",
            "linewidth": 0.5,
        },
        showlegend=False,
        hovermode="closest",
        updatemenus=updatemenus,
        width=530,
        height=500,
    )
    return layout

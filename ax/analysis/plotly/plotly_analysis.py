# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import pandas as pd
from ax.core.analysis_card import AnalysisCard
from plotly import graph_objects as go, io as pio

# Body HTML template for Plotly figures with a couple tricks for rendering in Jupyter.
# 1. It is necessary to use the UTF-8 encoding with plotly graphics to get e.g.
# negativesigns to render correctly
# 2. Need to define a fixed height for the content so that the plotly figure doesn't
# get squished.
body_html_template = """
<meta charset="utf-8" />
<style>
.content {{
    overflow-x: auto;
    overflow-y: auto;
    height: 500px;
}}
</style>
<div class="content">
    {figure_html}
</div>
"""


class PlotlyAnalysisCard(AnalysisCard):
    def get_figure(self) -> go.Figure:
        return pio.from_json(self.blob)

    def _body_html(self, depth: int) -> str:
        """
        Return the standalone HTML of the Plotly figure that can be rendered in an
        IPython environment (ex. Jupyter).
        """

        return body_html_template.format(
            figure_html=self.get_figure().to_html(
                full_html=False, include_plotlyjs=False
            )
        )

    def _body_papermill(self) -> go.Figure:
        """
        Return the Plotly figure directly to use the default renderer.
        """
        return self.get_figure()


def create_plotly_analysis_card(
    name: str,
    title: str,
    subtitle: str,
    df: pd.DataFrame,
    fig: go.Figure,
) -> PlotlyAnalysisCard:
    return PlotlyAnalysisCard(
        name=name,
        title=title,
        subtitle=subtitle,
        df=df,
        blob=pio.to_json(fig),
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import Sequence

import pandas as pd
from ax.analysis.analysis import Analysis, AnalysisBlobAnnotation, AnalysisCard
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from plotly import graph_objects as go, io as pio
from pyre_extensions import override

# Body HTML template for Plotly figures with a couple tricks for rendering in Jupyter.
# 1. It is necessary to use the UTF-8 encoding with plotly graphics to get e.g.
# negativesigns to render correctly
# 2. Need to define a fixed height for the content so that the plotly figure doesn't
# get squished.
# 3. require.js is not compatible with ES6 import used by plotly.js so we null out
# define
body_html_template = """
<meta charset="utf-8" />
<style>
.content {{
    overflow-x: auto;
    overflow-y: auto;
    height: 500px;
}}
</style>
<script>define = null;</script>
<div class="content">
    {figure_html}
</div>
"""


class PlotlyAnalysisCard(AnalysisCard):
    blob_annotation: AnalysisBlobAnnotation = AnalysisBlobAnnotation.PLOTLY

    def get_figure(self) -> go.Figure:
        return pio.from_json(self.blob)

    def _body_html(self) -> str:
        """
        Return the standalone HTML of the Plotly figure that can be rendered in an
        IPython environment (ex. Jupyter).
        """

        return body_html_template.format(
            figure_html=self.get_figure().to_html(
                full_html=False, include_plotlyjs=True
            )
        )

    def _body_papermill(self) -> go.Figure:
        """
        Return the Plotly figure directly to use the default renderer.
        """
        return self.get_figure()


class PlotlyAnalysis(Analysis):
    """
    An Analysis that computes a Plotly figure.
    """

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[PlotlyAnalysisCard]: ...

    def _create_plotly_analysis_card(
        self,
        title: str,
        subtitle: str,
        level: int,
        df: pd.DataFrame,
        fig: go.Figure,
        category: int,
    ) -> PlotlyAnalysisCard:
        """
        Make a PlotlyAnalysisCard from this Analysis using provided fields and
        details about the Analysis class.
        """
        return PlotlyAnalysisCard(
            name=self.name,
            attributes=self.attributes,
            title=title,
            subtitle=subtitle,
            level=level,
            df=df,
            blob=pio.to_json(fig),
            category=category,
        )

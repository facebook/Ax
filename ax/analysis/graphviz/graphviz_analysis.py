# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.analysis.analysis_card import AnalysisCard
from graphviz import Digraph, Source


class GraphvizAnalysisCard(AnalysisCard):
    def get_digraph(self) -> Digraph:
        return Source(self.blob)

    def _body_html(self, depth: int) -> str:
        """
        Return the a HTML div with the Graphviz figure as an SVG.
        """
        svg = self.get_digraph().pipe(format="svg")

        return f"<div>{svg}</div>"


def create_graphviz_analysis_card(
    name: str,
    title: str,
    subtitle: str,
    df: pd.DataFrame,
    dot: Digraph,
) -> GraphvizAnalysisCard:
    return GraphvizAnalysisCard(
        name=name,
        title=title,
        subtitle=subtitle,
        df=df,
        blob=dot.source,
    )

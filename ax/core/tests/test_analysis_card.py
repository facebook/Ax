# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.analysis.analysis import AnalysisCard
from ax.analysis.markdown.markdown_analysis import MarkdownAnalysisCard
from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from ax.core.analysis_card import AnalysisCardGroup
from ax.utils.common.testutils import TestCase
from plotly import graph_objects as go, io as pio


class TestAnalysisCard(TestCase):
    def test_hierarchy_str(self) -> None:
        test_df = pd.DataFrame(
            columns=["a", "b"],
            data=[
                [1, 2],
                [3, 4],
            ],
        )

        base_analysis_card = AnalysisCard(
            name="test_base_analysis_card",
            title="test_base_analysis_card_title",
            subtitle="test_subtitle",
            df=test_df,
            blob="test blob",
        )
        markdown_analysis_card = MarkdownAnalysisCard(
            name="test_markdown_analysis_card",
            title="test_markdown_analysis_card_title",
            subtitle="test_subtitle",
            df=test_df,
            blob="This is some **really cool** markdown",
        )
        plotly_analysis_card = PlotlyAnalysisCard(
            name="test_plotly_analysis_card",
            title="test_plotly_analysis_card_title",
            subtitle="test_subtitle",
            df=test_df,
            blob=pio.to_json(go.Figure()),
        )

        # Create two groups which hold the leaf cards
        small_group = AnalysisCardGroup(
            name="small_group",
            title="Small Group",
            subtitle="This is a small group with just a few cards",
            children=[base_analysis_card, markdown_analysis_card],
        )
        big_group = AnalysisCardGroup(
            name="big_group",
            title="Big Group",
            subtitle="This is a big group with a lot of cards",
            children=[plotly_analysis_card, small_group],
        )

        expected = """big_group
    test_plotly_analysis_card_title
    small_group
        test_base_analysis_card_title
        test_markdown_analysis_card_title"""

        self.assertEqual(big_group.hierarchy_str(), expected)

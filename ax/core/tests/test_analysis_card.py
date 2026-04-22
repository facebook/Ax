# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.analysis.analysis import AnalysisCard
from ax.analysis.markdown.markdown_analysis import MarkdownAnalysisCard
from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from ax.core.analysis_card import AnalysisCardGroup, NotApplicableStateAnalysisCard
from ax.utils.common.testutils import TestCase
from plotly import graph_objects as go, io as pio


DUMMY_DF: pd.DataFrame = pd.DataFrame(
    columns=["a", "b"],
    data=[[1, 2], [3, 4]],
)


class TestAnalysisCard(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.base_card = AnalysisCard(
            name="test_base_analysis_card",
            title="test_base_analysis_card_title",
            subtitle="test_subtitle",
            df=DUMMY_DF,
            blob="test blob",
        )

    def test_hierarchy_str(self) -> None:
        markdown_analysis_card = MarkdownAnalysisCard(
            name="test_markdown_analysis_card",
            title="test_markdown_analysis_card_title",
            subtitle="test_subtitle",
            df=DUMMY_DF,
            blob="This is some **really cool** markdown",
        )
        plotly_analysis_card = PlotlyAnalysisCard(
            name="test_plotly_analysis_card",
            title="test_plotly_analysis_card_title",
            subtitle="test_subtitle",
            df=DUMMY_DF,
            blob=pio.to_json(go.Figure()),
        )

        # Create two groups which hold the leaf cards
        small_group = AnalysisCardGroup(
            name="small_group",
            title="Small Group",
            subtitle="This is a small group with just a few cards",
            children=[self.base_card, markdown_analysis_card],
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

    def test_not_applicable_card(self) -> None:
        """Test NotApplicableStateAnalysisCard._body_html renders blob content."""
        card = NotApplicableStateAnalysisCard(
            name="Test",
            title="Test",
            subtitle="",
            df=pd.DataFrame(),
            blob="Explanation text.",
        )
        self.assertIn("Explanation text.", card._body_html(depth=0))

    def test_subtitle_toggle_label_rendering(self) -> None:
        """Verify subtitle_toggle_label controls toggle button text in HTML."""
        for label, expected_text in (
            ("", "See more"),
            (
                "Expand to see annotated parameters.",
                "Expand to see annotated parameters.",
            ),
        ):
            with self.subTest(label=label):
                card = AnalysisCard(
                    name="Test",
                    title="Title",
                    subtitle="A long subtitle",
                    df=pd.DataFrame(),
                    blob="blob",
                    subtitle_toggle_label=label,
                )
                self.assertEqual(card.subtitle_toggle_label, label)
                html = card._repr_html_()
                self.assertIn(expected_text, html)

    def test_analysis_card_group_html_does_not_render_toggle(self) -> None:
        """AnalysisCardGroup._to_html uses html_group_card_template which renders
        the subtitle as a plain <p> tag (no collapsible toggle). Verify the group's
        own subtitle_toggle_label is stored but not rendered in the group header."""

        group = AnalysisCardGroup(
            name="G",
            title="GT",
            subtitle="GS",
            children=[self.base_card],
            subtitle_toggle_label="Custom toggle.",
        )
        self.assertEqual(group.subtitle_toggle_label, "Custom toggle.")

        html = group._to_html(depth=0)

        # The group template uses a plain <p> for subtitles, not the
        # collapsible card-subtitle + toggle-button pattern.
        self.assertNotIn("Custom toggle.", html)
        self.assertIn('<p class="group-subtitle">', html)
        self.assertIn("GS", html)

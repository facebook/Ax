# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import markdown
import pandas as pd
from ax.analysis.analysis_card import AnalysisCard
from IPython.display import Markdown


class MarkdownAnalysisCard(AnalysisCard):
    def get_markdown(self) -> str:
        return self.blob

    def _body_html(self, depth: int) -> str:
        return f"<div class='content'>{markdown.markdown(self.get_markdown())}<div>"

    def _body_papermill(self) -> Markdown:
        return Markdown(self.get_markdown())


def create_markdown_analysis_card(
    name: str,
    title: str,
    subtitle: str,
    df: pd.DataFrame,
    message: str,
) -> MarkdownAnalysisCard:
    return MarkdownAnalysisCard(
        name=name,
        title=title,
        subtitle=subtitle,
        df=df,
        blob=message,
    )

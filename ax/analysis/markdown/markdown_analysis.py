# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import pandas as pd
from ax.analysis.analysis import Analysis, AnalysisCard, AnalysisCardLevel
from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStrategy


class MarkdownAnalysisCard(AnalysisCard):
    name: str

    title: str
    subtitle: str
    level: AnalysisCardLevel

    df: pd.DataFrame
    blob: str
    blob_annotation = "markdown"

    def get_markdown(self) -> str:
        return self.blob


class MarkdownAnalysis(Analysis):
    """
    An Analysis that computes a paragraph of Markdown formatted text.
    """

    def compute(
        self,
        experiment: Optional[Experiment] = None,
        generation_strategy: Optional[GenerationStrategy] = None,
    ) -> MarkdownAnalysisCard: ...

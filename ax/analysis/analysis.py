# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Optional, Protocol

import pandas as pd
from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStrategy


class AnalysisCardLevel(Enum):
    DEBUG = 0
    LOW = 1
    MID = 2
    HIGH = 3
    CRITICAL = 4


class AnalysisCard:
    # Name of the analysis computed, usually the class name of the Analysis which
    # produced the card. Useful for grouping by when querying a large collection of
    # cards.
    name: str

    title: str
    subtitle: str
    level: AnalysisCardLevel

    df: pd.DataFrame  # Raw data produced by the Analysis

    # pyre-ignore[4] We explicitly want to allow any type here, blob is narrowed in
    # AnalysisCard's subclasses
    blob: Any  # Data processed and ready for end-user consumption

    # How to interpret the blob (ex. "dataframe", "plotly", "markdown")
    blob_annotation = "dataframe"

    def __init__(
        self,
        name: str,
        title: str,
        subtitle: str,
        level: AnalysisCardLevel,
        df: pd.DataFrame,
        # pyre-ignore[2] We explicitly want to allow any type here, blob is narrowed in
        # AnalysisCard's subclasses
        blob: Any,
    ) -> None:
        self.name = name
        self.title = title
        self.subtitle = subtitle
        self.level = level
        self.df = df
        self.blob = blob


class Analysis(Protocol):
    """
    An Analysis is a class that given either and Experiment, a GenerationStrategy, or
    both can compute some data intended for end-user consumption. The data is returned
    to the user in the form of an AnalysisCard which contains the raw data, a blob (the
    data processed for end-user consumption), and miscellaneous metadata that can be
    useful for rendering the card or a collection of cards.

    The AnalysisCard is a thin wrapper around the raw data and the processed blob;
    Analyses impose structure on their blob should subclass Analysis. See
    PlotlyAnalysis for an example which produces cards where the blob is always a
    Plotly Figure object.

    A good pattern to follow when implementing your own Analyses is to configure
    "settings" (like which parameter or metrics to operate on, or whether to use
    observed or modeled effects) in your Analyses' __init__ methods, then to consume
    these settings in the compute method.
    """

    def compute(
        self,
        experiment: Optional[Experiment] = None,
        generation_strategy: Optional[GenerationStrategy] = None,
    ) -> AnalysisCard:
        # Note: when implementing compute always prefer experiment.lookup_data() to
        # experiment.fetch_data() to avoid unintential data fetching within the report
        # generation.
        ...

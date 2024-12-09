# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
from collections.abc import Iterable
from enum import IntEnum
from logging import Logger
from typing import Any, Protocol

import pandas as pd
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.utils.common.base import Base
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Err, ExceptionE, Ok, Result
from IPython.display import display, Markdown

logger: Logger = get_logger(__name__)


class AnalysisCardLevel(IntEnum):
    DEBUG = 0
    LOW = 10
    MID = 20
    HIGH = 30
    CRITICAL = 40


class AnalysisCard(Base):
    # Name of the analysis computed, usually the class name of the Analysis which
    # produced the card. Useful for grouping by when querying a large collection of
    # cards.
    name: str
    # Arguments passed to the Analysis which produced the card, or their eventual
    # values if they were inferred.
    attributes: dict[str, Any]

    title: str
    subtitle: str

    level: int

    df: pd.DataFrame  # Raw data produced by the Analysis

    # Blob is the data processed for end-user consumption, encoded as a string,
    # typically JSON. Subclasses of Analysis can define their own methods for consuming
    # the blob and presenting it to the user (ex. PlotlyAnalysisCard.get_figure()
    # decodes the blob into a go.Figure object).
    blob: str
    # How to interpret the blob (ex. "dataframe", "plotly", "markdown")
    blob_annotation = "dataframe"

    def __init__(
        self,
        name: str,
        title: str,
        subtitle: str,
        level: int,
        df: pd.DataFrame,
        blob: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.title = title
        self.subtitle = subtitle
        self.level = level
        self.df = df
        self.blob = blob
        self.attributes = {} if attributes is None else attributes

    def _ipython_display_(self) -> None:
        """
        IPython display hook. This is called when the AnalysisCard is printed in an
        IPython environment (ex. Jupyter). This method should be implemented by
        subclasses of Analysis to display the AnalysisCard in a useful way.

        By default, this method displays the raw data in a pandas DataFrame.
        """
        display(Markdown(f"## {self.title}\n\n### {self.subtitle}"))
        display(self.df)


def display_cards(
    cards: Iterable[AnalysisCard], minimum_level: int = AnalysisCardLevel.LOW
) -> None:
    """
    Display a collection of AnalysisCards in IPython environments (ex. Jupyter).

    Args:
        cards: Collection of AnalysisCards to display.
        minimum_level: Minimum level of cards to display.
    """
    for card in sorted(cards, key=lambda x: x.level, reverse=True):
        if card.level >= minimum_level:
            display(card)


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
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> AnalysisCard:
        # Note: when implementing compute always prefer experiment.lookup_data() to
        # experiment.fetch_data() to avoid unintential data fetching within the report
        # generation.
        ...

    def compute_result(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> Result[AnalysisCard, AnalysisE]:
        """
        Utility method to compute an AnalysisCard as a Result. This can be useful for
        computing many Analyses at once and handling Exceptions later.
        """

        try:
            card = self.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )
            return Ok(value=card)
        except Exception as e:
            logger.error(f"Failed to compute {self.__class__.__name__}: {e}")

            return Err(
                value=AnalysisE(
                    message=f"Failed to compute {self.__class__.__name__}",
                    exception=e,
                    analysis=self,
                )
            )

    def _create_analysis_card(
        self,
        title: str,
        subtitle: str,
        level: int,
        df: pd.DataFrame,
    ) -> AnalysisCard:
        """
        Make an AnalysisCard from this Analysis using provided fields and
        details about the Analysis class.
        """
        return AnalysisCard(
            name=self.name,
            attributes=self.attributes,
            title=title,
            subtitle=subtitle,
            level=level,
            df=df,
            blob=df.to_json(),
        )

    @property
    def name(self) -> str:
        """The name the AnalysisCard will be given in compute."""
        return self.__class__.__name__

    @property
    def attributes(self) -> dict[str, Any]:
        """The attributes the AnalysisCard will be given in compute."""
        return self.__dict__

    def __repr__(self) -> str:
        try:
            return (
                f"{self.__class__.__name__}(name={self.name}, "
                f"attributes={json.dumps(self.attributes)})"
            )
        # in case there is logic in name or attributes that throws a json error
        except Exception:
            return self.__class__.__name__


class AnalysisE(ExceptionE):
    analysis: Analysis

    def __init__(
        self,
        message: str,
        exception: Exception,
        analysis: Analysis,
    ) -> None:
        super().__init__(message, exception)
        self.analysis = analysis

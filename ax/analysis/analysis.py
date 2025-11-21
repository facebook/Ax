# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import traceback
from logging import Logger
from typing import Protocol, Sequence

import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis_card import (
    AnalysisCard,
    AnalysisCardBase,
    AnalysisCardGroup,
    ErrorAnalysisCard,
)
from ax.core.experiment import Experiment
from ax.exceptions.analysis import AnalysisNotApplicableStateError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Err, ExceptionE, Ok, Result
from IPython.display import display

logger: Logger = get_logger(__name__)


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
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardBase:
        # Note: when implementing compute always prefer experiment.lookup_data() to
        # experiment.fetch_data() to avoid unintential data fetching within the report
        # generation.
        ...

    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        Validates that the Experiment, GenerationStrategy, and/or Adapter are in an
        applicable state to compute this Analysis for its given settings. If the state
        is not applicable, returns a string describing why; if the state is applicable,
        returns None.

        Example: if ArmEffectsPlot(metric_name="foo").validate_applicable_state(...) is
        called on an Experiment with no data for metric "foo", it will return a string
        clearly stating that the Experiment is still waiting for data for "foo".
        """
        ...

    def compute_result(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Result[AnalysisCardBase, AnalysisE]:
        """
        Utility method to compute an AnalysisCard as a Result. This can be useful for
        computing many Analyses at once and handling Exceptions later.
        """
        not_applicable_explanation = self.validate_applicable_state(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )
        if not_applicable_explanation is not None:
            return Err(
                value=AnalysisE(
                    message="Analysis is not applicable to given state",
                    exception=AnalysisNotApplicableStateError(
                        not_applicable_explanation
                    ),
                    analysis=self,
                )
            )

        try:
            card = self.compute(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )
            return Ok(value=card)

        except Exception as e:
            logger.error(f"Failed to compute {self.__class__.__name__}")
            logger.error(traceback.format_exc())

            return Err(
                value=AnalysisE(
                    message=f"Failed to compute {self.__class__.__name__}",
                    exception=e,
                    analysis=self,
                )
            )

    def compute_or_error_card(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardBase:
        """
        Utility method to compute an AnalysisCard or an ErrorAnalysisCard if an
        exception is raised.
        """
        return self.compute_result(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        ).unwrap_or_else(error_card_from_analysis_e)

    def _create_analysis_card(
        self,
        title: str,
        subtitle: str,
        df: pd.DataFrame,
    ) -> AnalysisCard:
        """
        Make an AnalysisCard from this Analysis using provided fields and
        details about the Analysis class.
        """
        return AnalysisCard(
            name=self.__class__.__name__,
            title=title,
            subtitle=subtitle,
            df=df,
            blob=df.to_json(),
        )

    def _create_analysis_card_group(
        self,
        title: str,
        subtitle: str | None,
        children: Sequence[AnalysisCardBase],
    ) -> AnalysisCardGroup:
        """
        Make an AnalysisCardGroup from this Analysis using provided fields and
        details about the Analysis class.
        """
        return AnalysisCardGroup(
            name=self.__class__.__name__,
            title=title,
            subtitle=subtitle if subtitle is not None else "",
            children=children,
        )


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


def display_cards(
    cards: Sequence[AnalysisCardBase],
) -> None:
    """
    Helper method for displaying a sequence of AnalysisCards (as is returned by adhoc
    compute_ methods and by Client.compute_analyses).
    """
    for card in cards:
        display(card)


def error_card_from_analysis_e(
    analysis_e: AnalysisE,
) -> ErrorAnalysisCard:
    analysis_name = analysis_e.analysis.__class__.__name__
    exception_name = analysis_e.exception.__class__.__name__

    return ErrorAnalysisCard(
        name=analysis_name,
        title=f"{analysis_name} Error",
        subtitle=f"{exception_name} encountered while computing {analysis_name}.",
        df=pd.DataFrame(),
        blob=analysis_e.tb_str() or "",
    )

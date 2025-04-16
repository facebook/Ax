# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterable

from ax.analysis.analysis import Analysis, AnalysisCard, AnalysisCardCategory
from ax.analysis.plotly.parallel_coordinates import ParallelCoordinatesPlot
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.with_db_settings_base import WithDBSettingsBase


class AnalysisBase(WithDBSettingsBase):
    """
    Base class for analysis functionality shared between AxClient and Scheduler.
    """

    # pyre-fixme[13]: Attribute `experiment` is declared in class
    # `AnalysisBase` to have type `Experiment` but is never initialized
    experiment: Experiment
    # pyre-fixme[13]: Attribute `generation_strategy` is declared in class
    # `AnalysisBase` to have type `GenerationStrategy` but
    # is never initialized
    generation_strategy: GenerationStrategy

    def _choose_analyses(
        self, categories: list[AnalysisCardCategory] | None = None
    ) -> list[Analysis]:
        """
        Choose Analyses to compute based on the Experiment, GenerationStrategy, etc.
        """

        # TODO Create a useful heuristic for choosing analyses
        return [ParallelCoordinatesPlot()]

    def compute_analyses(
        self,
        analyses: Iterable[Analysis] | None = None,
    ) -> list[AnalysisCard]:
        """
        Compute AnalysisCards (data about the optimization for end-user consumption)
        using the Experiment and GenerationStrategy. If no analyses are provided use
        some heuristic to determine which analyses to run. If some analyses fail, log
        failure and continue to compute the rest.

        Note that the Analysis class is NOT part of the API and its methods are subject
        to change incompatibly between minor versions. Users are encouraged to use the
        provided analyses or leave this argument as None to use the default analyses.

        Saves to database on completion if storage_config is present.

        Args:
            analyses: A list of Analysis classes to run. If None Ax will choose which
                analyses to run based on the state of the experiment.
        Returns:
            A list of AnalysisCards.
        """

        analyses = analyses if analyses is not None else self._choose_analyses()

        # Compute Analyses one by one and accumulate Results holding either the
        # AnalysisCard or an Exception and some metadata
        results = [
            analysis.compute_result(
                experiment=self.experiment,
                generation_strategy=self.generation_strategy,
            )
            for analysis in analyses
        ]

        # Turn Exceptions into MarkdownAnalysisCards with the traceback as the message
        cards = [
            card
            for result in results
            for card in result.unwrap_or_else(lambda e: e.error_card())
        ]

        # Save the AnalysisCards to the database if possible
        self._save_analysis_cards_to_db_if_possible(
            experiment=self.experiment, analysis_cards=cards
        )

        return cards

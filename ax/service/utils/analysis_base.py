# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import traceback
from typing import Iterable

import pandas as pd

from ax.analysis.analysis import Analysis, AnalysisCard, AnalysisCardLevel, AnalysisE
from ax.analysis.markdown.markdown_analysis import MarkdownAnalysisCard
from ax.analysis.plotly.parallel_coordinates import ParallelCoordinatesPlot
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service.utils.with_db_settings_base import WithDBSettingsBase
from ax.utils.common.typeutils import checked_cast
from pyre_extensions import none_throws


class AnalysisBase(WithDBSettingsBase):
    """
    Base class for analysis functionality shared between AxClient and Scheduler.
    It also manages the experiment and generation strategy associated with the
    instance.
    """

    # pyre-fixme[13]: Attribute `experiment` is declared in class
    # `AnalysisBase` to have type `Experiment` but is never initialized
    _experiment: Experiment | None
    # pyre-fixme[13]: Attribute `generation_strategy` is declared in class
    # `AnalysisBase` to have type `GenerationStrategyInterface` but
    # is never initialized
    _generation_strategy: GenerationStrategyInterface | None

    def _choose_analyses(self) -> list[Analysis]:
        """
        Choose Analyses to compute based on the Experiment, GenerationStrategy, etc.
        """

        # TODO Create a useful heuristic for choosing analyses
        return [ParallelCoordinatesPlot()]

    def compute_analyses(
        self, analyses: Iterable[Analysis] | None = None
    ) -> list[AnalysisCard]:
        """
        Compute Analyses for the Experiment and GenerationStrategy associated with this
        Scheduler instance and save them to the DB if possible. If an Analysis fails to
        compute (e.g. due to a missing metric), it will be skipped and a warning will
        be logged.

        Args:
            analyses: Analyses to compute. If None, the Scheduler will choose a set of
                Analyses to compute based on the Experiment and GenerationStrategy.
        """
        analyses = analyses if analyses is not None else self._choose_analyses()

        results = [
            analysis.compute_result(
                experiment=self.experiment,
                generation_strategy=self.generation_strategy,
            )
            for analysis in analyses
        ]

        # TODO Accumulate Es into their own card, perhaps via unwrap_or_else
        cards = [result.unwrap() for result in results if result.is_ok()]

        for result in results:
            if result.is_err():
                e = checked_cast(AnalysisE, result.err)
                traceback_str = "".join(
                    traceback.format_exception(
                        type(result.err.exception),
                        e.exception,
                        e.exception.__traceback__,
                    )
                )
                cards.append(
                    MarkdownAnalysisCard(
                        name=e.analysis.name,
                        # It would be better if we could reliably compute the title
                        # without risking another error
                        title=f"{e.analysis.name} Error",
                        subtitle=f"An error occurred while computing {e.analysis}",
                        attributes=e.analysis.attributes,
                        blob=traceback_str,
                        df=pd.DataFrame(),
                        level=AnalysisCardLevel.DEBUG,
                    )
                )

        self._save_analysis_cards_to_db_if_possible(
            analysis_cards=cards,
            experiment=self.experiment,
        )

        return cards

    @property
    def experiment(self) -> Experiment:
        """Returns the experiment set on this instance."""
        return none_throws(
            self._experiment,
            (
                f"Experiment not set on {self.__class__.__name__}. Must first "
                "call load_experiment or create_experiment to use handler functions."
            ),
        )

    @experiment.setter
    def experiment(self, experiment: Experiment) -> None:
        """Sets the experiment on this instance."""
        self._experiment = experiment

    @property
    def generation_strategy(self) -> GenerationStrategyInterface:
        """Returns the generation strategy, set on this experiment."""
        return none_throws(
            self._generation_strategy,
            "No generation strategy has been set on this optimization yet.",
        )

    @generation_strategy.setter
    def generation_strategy(
        self, generation_strategy: GenerationStrategyInterface
    ) -> None:
        """Sets the generation strategy on this instance."""
        self._generation_strategy = generation_strategy

    @property
    def standard_generation_strategy(self) -> GenerationStrategy:
        """Used for operations in the scheduler that can only be done with
        and instance of ``GenerationStrategy``.
        """
        gs = self.generation_strategy
        if not isinstance(gs, GenerationStrategy):
            raise NotImplementedError(
                "This functionality is only supported with instances of "
                "`GenerationStrategy` (one that uses `GenerationStrategy` "
                "class) and not yet with other types of "
                "`GenerationStrategyInterface`."
            )
        return gs

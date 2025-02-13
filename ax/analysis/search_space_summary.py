# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.analysis import Analysis, AnalysisCard, AnalysisCardLevel
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy


class SearchSpaceSummary(Analysis):
    """
    Creates a dataframe with information about each parameter in the given
    search space. The resulting dataframe has one row per parameter, and the
    following columns:
        - Name: the name of the parameter.
        - Type: the parameter subclass (Fixed, Range, Choice).
        - Domain: the parameter's domain (e.g., "range=[0, 1]" or
            "values=['a', 'b']").
        - Datatype: the datatype of the parameter (int, float, str, bool).
        - Flags: flags associated with the parameter, if any.
        - Target Value: the target value of the parameter, if applicable.
        - Dependent Parameters: for parameters in hierarchical search spaces,
        mapping from parameter value -> list of dependent parameter names.
    """

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
    ) -> AnalysisCard:
        if experiment is None:
            raise UserInputError(
                "`SearchSpaceSummary` analysis requires an `Experiment` input"
            )
        return self._create_analysis_card(
            title=f"SearchSpaceSummary for `{experiment.name}`",
            subtitle="High-level summary of the `Parameter`-s in this `Experiment`",
            level=AnalysisCardLevel.MID,
            df=experiment.search_space.summary_df,
        )

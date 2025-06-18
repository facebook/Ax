# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.analysis_card import AnalysisCardGroup
from ax.analysis.plotly.cross_validation import CrossValidationPlot
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import none_throws, override


class DiagnosticAnalysis(Analysis):
    """
    An Analysis that provides diagnostic information about the optimization process.
    This includes information about the quality of the model fit, such as the results
    of leave-one-out cross validation.
    """

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardGroup:
        if experiment is None:
            raise UserInputError("DiagnosticAnalysis requires an Experiment.")

        # Extract all metric names from the OptimizationConfig.
        metric_names = [*none_throws(experiment.optimization_config).metrics.keys()]

        return self._create_analysis_card_group(
            children=[
                *CrossValidationPlot(metric_names=metric_names)
                .compute_or_error_card(
                    experiment=experiment,
                    generation_strategy=generation_strategy,
                    adapter=adapter,
                )
                .flatten()
            ]
        )

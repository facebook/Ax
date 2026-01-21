# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.graphviz.generation_strategy_graph import GenerationStrategyGraph
from ax.analysis.plotly.cross_validation import CrossValidationPlot
from ax.analysis.utils import validate_experiment
from ax.core.analysis_card import AnalysisCardGroup
from ax.core.experiment import Experiment
from ax.core.utils import is_bandit_experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import none_throws, override

DIAGNOSTICS_CARDGROUP_TITLE = "Diagnostic Analysis"

DIAGNOSTICS_CARDGROUP_SUBTITLE = (
    "Diagnostic Analyses provide information about the optimization process and "
    "the quality of the model fit. You can use this information to understand "
    "if the experimental design should be adjusted to improve optimization quality."
)


@final
class DiagnosticAnalysis(Analysis):
    """
    An Analysis that provides diagnostic information about the optimization process.
    This includes information about the quality of the model fit, such as the results
    of leave-one-out cross validation.
    """

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        return validate_experiment(
            experiment=experiment,
            require_trials=False,
            require_data=False,
        )

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardGroup:
        experiment = none_throws(experiment)

        # Extract all metric names from the OptimizationConfig.
        metric_names = [*none_throws(experiment.optimization_config).metrics.keys()]

        is_bandit = generation_strategy and is_bandit_experiment(
            generation_strategy_name=generation_strategy.name
        )

        cross_validation_plots = (
            [
                CrossValidationPlot(metric_names=metric_names).compute_or_error_card(
                    experiment=experiment,
                    generation_strategy=generation_strategy,
                    adapter=adapter,
                )
            ]
            if not is_bandit
            else []
        )

        generation_strategy_graph = (
            [
                GenerationStrategyGraph().compute_or_error_card(
                    experiment=experiment,
                    generation_strategy=generation_strategy,
                    adapter=adapter,
                )
            ]
            if generation_strategy is not None
            else []
        )

        return self._create_analysis_card_group(
            title=DIAGNOSTICS_CARDGROUP_TITLE,
            subtitle=DIAGNOSTICS_CARDGROUP_SUBTITLE,
            children=[*cross_validation_plots, *generation_strategy_graph],
        )
